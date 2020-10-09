from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from torch.utils.data import DataLoader

# Resolve Tensorbard Confliction across pytorch version
import torch
import warnings
version_num = torch.__version__
version_num = ''.join(i for i in version_num if i.isdigit())
version_num = int(version_num.ljust(10, '0'))
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    if version_num > 1100000000:
        from torch.utils.tensorboard import SummaryWriter
    else:
        from tensorboardX import SummaryWriter

from Exp_PreSIL.dataloader_kitti import KittiDataset

import networks

import time
import json

from layers import *
from networks import *

import argparse

default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",                  type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                    type=str,                               help="path to kitti gt file")
parser.add_argument("--predang_path",               type=str,                               help="path to kitti gt file")
parser.add_argument("--semanticspred_path",         type=str,   default='None',             help="path to kitti semantics prediction file")
parser.add_argument("--val_gt_path",                type=str,                               help="path to validation gt file")
parser.add_argument("--split",                      type=str,                               help="train/val split to use")
parser.add_argument("--num_layers",                 type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                     type=int,   default=320,                help="input image height")
parser.add_argument("--width",                      type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                       type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                       type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",                  type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",                  type=float, default=100.0,              help="maximum depth")
parser.add_argument("--load_depthweights_folder",   type=str,                               help="path to kitti gt file")

# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=12,                 help="batch size")
parser.add_argument("--num_workers",                type=int,   default=6,                  help="number of dataloader workers")

class Trainer:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.device = "cuda"

        self.depthmodels = {}
        self.depthmodels["depthencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=False, num_input_channels=3)
        self.depthmodels["depthdecoder"] = DepthDecoder(self.depthmodels["depthencoder"].num_ch_enc, num_output_channels=1)
        self.depthmodels["depthencoder"].to(self.device)
        self.depthmodels["depthdecoder"].to(self.device)
        self.load_model(weightFolder=self.opt.load_depthweights_folder, encoderName='depthencoder',
                        decoderName='depthdecoder', encoder=self.depthmodels["depthencoder"],
                        decoder=self.depthmodels["depthdecoder"])
        for m in self.depthmodels.values():
            m.eval()

        print("Training is using:\t", self.device)

        self.set_dataset()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.STEREO_SCALE_FACTOR = 5.4

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "{}_files.txt")
        test_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "test_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(test_fpath)

        train_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, train_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=True, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path
        )

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.val_gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path
        )

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        for batch_idx, inputs in enumerate(self.val_loader):
            for key, ipt in inputs.items():
                if not key == 'tag':
                    inputs[key] = ipt.to(self.device)

            rgb_wgrad = torch.clone(inputs['color'])
            rgb_wgrad.requires_grad = True
            zeroer = optim.SGD([rgb_wgrad], lr=1)
            outputs_depth = self.depthmodels['depthdecoder'](self.depthmodels['depthencoder'](rgb_wgrad))

            _, pred_depth = disp_to_depth(outputs_depth['disp', 0], min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
            pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

            xx, yy = np.meshgrid(range(self.opt.width), range(self.opt.height), indexing='xy')
            xx = xx.flatten()
            yy = yy.flatten()
            rndind = np.random.randint(xx.shape[0])

            rndxx = xx[rndind]
            rndyy = yy[rndind]

            loss = pred_depth[0,0,rndyy,rndxx] / pred_depth[0,0,rndyy,rndxx].detach()
            zeroer.zero_grad()
            loss.backward()
            sampleGrad = rgb_wgrad.grad
            sampleGrad = torch.mean(torch.abs(sampleGrad), dim=1, keepdim=True)
            overlaycolor = torch.clone(inputs['color'])

            overlaycolor[:, 0:1, :, :] = overlaycolor[:, 0:1, :, :] + sampleGrad * 50
            overlaycolor = torch.clamp(overlaycolor, max=1)

            fig, axs = plt.subplots(2, figsize=(16, 8))
            axs[0].imshow(tensor2rgb(inputs['color'], ind=0))
            axs[1].imshow(tensor2rgb(overlaycolor, ind=0))
            axs[1].scatter([rndxx], [rndyy], s=3, c='b')
            plt.savefig(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/analysis_receptivefieldReal', '{}.png'.format(batch_idx)))
            plt.close(fig)
    def load_model(self, weightFolder, encoderName, decoderName, encoder, decoder):
        """Load model(s) from disk
        """
        assert os.path.isdir(weightFolder), "Cannot find folder {}".format(weightFolder)
        print("loading model from folder {}".format(weightFolder))

        path = os.path.join(weightFolder, "encoder.pth")
        print("Loading {} weights...".format(encoderName))
        model_dict = encoder.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        encoder.load_state_dict(model_dict)

        path = os.path.join(weightFolder, "depth.pth")
        print("Loading {} weights...".format(decoderName))
        model_dict = decoder.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        decoder.load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
