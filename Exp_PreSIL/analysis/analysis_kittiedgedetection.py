from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from torch.utils.data import DataLoader

# Resolve Tensorbard Confliction across pytorch version
import torch
import warnings

from Exp_PreSIL.dataloader_kitti import KittiDataset

import networks


from layers import *
from networks import *

import argparse

default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",                  type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                    type=str,                               help="path to kitti gt file")
parser.add_argument("--semanticspred_path",         type=str,   default='None',             help="path to kitti semantics prediction file")
parser.add_argument("--instancepred_path",          type=str,   default='None',             help="path to kitti instance prediction file")
parser.add_argument("--split",                      type=str,                               help="train/val split to use")
parser.add_argument("--num_layers",                 type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                     type=int,   default=320,                help="input image height")
parser.add_argument("--width",                      type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                       type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                       type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",                  type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",                  type=float, default=100.0,              help="maximum depth")
parser.add_argument("--vlsfold",                    type=str)

# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=1,                 help="batch size")
parser.add_argument("--load_weights_folder_depth",  type=str,   default=None,               help="name of models to load")
parser.add_argument("--load_weights_folder_norm",   type=str,   default=None,               help="name of models to load")

class Trainer:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = "cuda"

        self.models["encoder_depth"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder_depth"].to(self.device)
        self.models["depth"] = DepthDecoder(self.models["encoder_depth"].num_ch_enc, num_output_channels=1)
        self.models["depth"].to(self.device)

        self.models["encoder_norm"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder_norm"].to(self.device)
        self.models["norm"] = DepthDecoder(self.models["encoder_norm"].num_ch_enc, num_output_channels=3)
        self.models["norm"].to(self.device)

        self.set_dataset()

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.load_model()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.minabsrel = 1e10
        self.maxa1 = -1e10

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.getcwd(), "splits", self.opt.split, "{}_files.txt")
        val_filenames = readlines(fpath.format("train"))

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, semanticspred_path=self.opt.semanticspred_path
        )

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True)

        self.val_num = val_dataset.__len__()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        # Define Gradient Kernel
        weightsx = torch.Tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0).expand([3, -1, -1, -1])
        weightsx = weightsx / 4 / 2

        weightsy = torch.Tensor([[-1., -2., -1.],
                                 [0., 0., 0.],
                                 [1., 2., 1.]]).unsqueeze(0).unsqueeze(0).expand([3, -1, -1, -1])
        weightsy = weightsy / 4 / 2
        self.diffx = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, groups=3)
        self.diffy = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, groups=3)

        self.diffx.weight = nn.Parameter(weightsx, requires_grad=False)
        self.diffy.weight = nn.Parameter(weightsy, requires_grad=False)

        self.diffx = self.diffx.cuda()
        self.diffy = self.diffy.cuda()

        intensitymeasurew = torch.Tensor([[1., 1., 1.],
                                          [1., 1., 1.],
                                          [1., 1., 1.]]).unsqueeze(0).unsqueeze(0) / 9
        intensitymeasurew = intensitymeasurew.expand([-1, 3, -1, -1]) / 3
        self.intensityconv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.intensityconv.weight = nn.Parameter(intensitymeasurew, requires_grad=False)
        self.intensityconv = self.intensityconv.cuda()

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                gradx = self.diffx(inputs['color'])
                gradx = torch.mean(torch.abs(gradx), dim=1, keepdim=True)
                grady = self.diffy(inputs['color'])
                grady = torch.mean(torch.abs(grady), dim=1, keepdim=True)
                intensity = self.intensityconv(inputs['color'])

                figname = inputs['tag'][0].split(' ')[0].split('/')[1] + '_' + inputs['tag'][0].split(' ')[1]

                fig1 = tensor2disp((gradx + grady) / (intensity + 1e-3), vmax=1, ind=0)
                fig2 = tensor2rgb(inputs['color'], ind=0)
                figcombined = np.concatenate([np.array(fig1), np.array(fig2)], axis=0)
                pil.fromarray(figcombined).save(os.path.join(self.opt.vlsfold, "{}.png".format(figname)))

    def load_model(self):
        """Load model(s) from disk
        """
        load_depth_folder = os.path.expanduser(self.opt.load_weights_folder_depth)
        load_norm_folder = os.path.expanduser(self.opt.load_weights_folder_norm)

        assert os.path.isdir(load_depth_folder), "Cannot find folder {}".format(self.opt.load_weights_folder_depth)
        assert os.path.isdir(load_norm_folder), "Cannot find folder {}".format(self.opt.load_weights_folder_norm)

        models_to_load = ['encoder_depth', 'depth']
        pthfilemapping = {'encoder_depth': 'encoder', 'depth': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder_depth, "{}.pth".format(pthfilemapping[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        models_to_load = ['encoder_norm', 'norm']
        pthfilemapping = {'encoder_norm': 'encoder', 'norm': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder_norm, "{}.pth".format(pthfilemapping[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
