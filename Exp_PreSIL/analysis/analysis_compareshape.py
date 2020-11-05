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
parser.add_argument("--load_angweights_folder", type=str,   default=None,               help="name of models to load")

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

        self.models["angencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["angencoder"].to(self.device)

        self.models["angdecoder"] = DepthDecoder(self.models["encoder_depth"].num_ch_enc, num_output_channels=2)
        self.models["angdecoder"].to(self.device)
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
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, instancepred_path=self.opt.instancepred_path)

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
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                _, _, gt_height, gt_width = inputs['depthgt'].shape

                outputs_depth = self.models['depth'](self.models['encoder_depth'](inputs['color']))
                _, pred_depth = disp_to_depth(outputs_depth[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_depth = pred_depth * self.STEREO_SCALE_FACTOR
                pred_depth = F.interpolate(pred_depth, [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=False)

                outputs_ang = self.models['angdecoder'](self.models['angencoder'](inputs['color']))
                angfromang = (outputs_ang[("disp", 0)] - 0.5) * 2 * np.pi
                angfromang = F.interpolate(angfromang, [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=False)
                angfromdepth = self.sfnormOptimizer.depth2ang_log(depthMap=pred_depth, intrinsic=inputs['K'])

                logfromang = self.sfnormOptimizer.ang2log(intrinsic=inputs["K"], ang=angfromang)
                logfromdepth = self.sfnormOptimizer.ang2log(intrinsic=inputs["K"], ang=angfromdepth)

                mask = torch.zeros_like(pred_depth, dtype=torch.int)
                mask[:, :, int(0.40810811 * self.opt.crph): -1, :] = 1
                mask = mask.contiguous()
                mask = (mask * inputs['instancepred'] > 0).int()
                inputs['instancepred'] = inputs['instancepred'].int().contiguous()

                totflops = 0
                for k in torch.unique(inputs['instancepred']):
                    if k > 0:
                        totflops += torch.sum(inputs['instancepred'] == k) ** 2
                totflops / float(self.opt.crph) / float(self.opt.crpw)

                itnum = 1
                lamb = 0.05
                depth_optedin = pred_depth.clone()
                import shapeintegration_cuda
                for i in range(itnum):
                    depth_optedout = torch.zeros_like(depth_optedin)
                    shapeintegration_cuda.shapeIntegration_crf_forward(logfromang, inputs['instancepred'], mask, pred_depth, depth_optedin, depth_optedout, self.opt.crph, self.opt.crpw, 1, lamb)
                    depth_optedin = depth_optedout.clone()

                sfnorm_opted = self.sfnormOptimizer.depth2norm(depth_optedout, intrinsic=inputs['K'])

                minang = - np.pi / 3 * 2
                maxang = 2 * np.pi - np.pi / 3 * 2
                vind = 0
                tensor2disp(angfromang[:, 0:1, :, :] - minang, vmax=maxang, ind=vind).show()
                tensor2disp(angfromdepth[:, 0:1, :, :] - minang, vmax=maxang, ind=vind).show()
                tensor2disp(angfromang[:, 1:2, :, :] - minang, vmax=maxang, ind=vind).show()
                tensor2disp(angfromdepth[:, 1:2, :, :] - minang, vmax=maxang, ind=vind).show()
                tensor2disp(mask, vmax=1, ind=vind).show()
                tensor2disp(1 / depth_optedout, vmax=0.15, ind=vind).show()
                tensor2rgb((sfnorm_opted + 1) / 2, ind=vind).show()
                figname = inputs['tag'][0].split(' ')[0].split('/')[1] + '_' + inputs['tag'][0].split(' ')[1]
                tensor2disp(inputs['instancepred'] > 0, vmax=1, ind=0).show()
                tensor2rgb(inputs['color'], ind=0).show()

    def load_model(self):
        """Load model(s) from disk
        """
        load_depth_folder = os.path.expanduser(self.opt.load_weights_folder_depth)
        load_weights_folder = os.path.expanduser(self.opt.load_angweights_folder)

        assert os.path.isdir(load_depth_folder), "Cannot find folder {}".format(self.opt.load_weights_folder_depth)
        assert os.path.isdir(load_weights_folder), "Cannot find folder {}".format(load_weights_folder)

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

        models_to_load = ['angencoder', 'angdecoder']
        modelnameMap = {'angencoder': 'encoder', 'angdecoder': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}.pth".format(modelnameMap[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
