from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

from torch.utils.data import DataLoader

from Exp_PreSIL.dataloader_PreSIL import PreSILDataset

import networks

import time
import json

from layers import *
from networks import *

import argparse

parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",              type=str,                               help="path to dataset")
parser.add_argument("--split",                  type=str,                               help="train/val split to use")
parser.add_argument("--height",                 type=int,   default=192,                help="input image height")
parser.add_argument("--width",                  type=int,   default=640,                help="input image width")
parser.add_argument("--num_workers",            type=int,   default=0,                  help="Dataloader Process Number")


# OPTIMIZATION options
parser.add_argument("--batch_size",             type=int,   default=12,                 help="batch size")


class Developer:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.set_dataset()

        print("Using split:\t  ", self.opt.split)
        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.height, width=self.opt.width, batch_size=self.opt.batch_size).cuda()

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))

        train_dataset = PreSILDataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            is_train=True
        )

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.train_num = train_dataset.__len__()

    def develop_dynamicReceptiveField(self):
        """Pass a minibatch through the network and generate images and losses
        """

        for batch_idx, inputs in enumerate(self.train_loader):
            for key, ipt in inputs.items():
                if not key == 'tag':
                    inputs[key] = ipt.to(torch.device("cuda"))

            rnddepth = torch.rand_like(inputs['depthgt']) * 30
            anggt = self.sfnormOptimizer.depth2ang_log(rnddepth, inputs['K'])
            loggt = self.sfnormOptimizer.ang2log(ang=anggt, intrinsic=inputs['K'])

            loghnp = loggt[0, 0, :, :].detach().cpu().numpy()
            logvnp = loggt[0, 1, :, :].detach().cpu().numpy()
            depthnp = rnddepth[0, 0, :, :].detach().cpu().numpy()
            dynamicReceptiveField(height=self.opt.height, width=self.opt.width, logh=loghnp, logv=logvnp, depth=depthnp, sr=20)
        return

def dynamicReceptiveField(height, width, logh, logv, depth, sr):
    for m in range(height):
        for n in range(width):
            # Left Direction
            intl = 0
            for lx in range(sr):
                ckx = n - (lx + 1)
                if ckx >= 0:
                    intl = intl + logh[m, ckx]
            refl = np.log(depth[m, n]) - np.log(depth[m, n - (lx + 1)])

            intr = 0
            for rx in range(sr):
                ckx = n + (rx + 1)
                if ckx < width:
                    intr = intr - logh[m, ckx - 1]
            refr = np.log(depth[m, n]) - np.log(depth[m, n + (rx + 1)])

            intu = 0
            for ru in range(sr):
                cky = m - (ru + 1)
                if cky >= 0:
                    intu = intu + logv[cky, n]
            refu = np.log(depth[m, n]) - np.log(depth[m - (ru + 1), n])

            intd = 0
            for rd in range(sr):
                cky = m + (rd + 1)
                if cky < height:
                    intd = intd - logv[cky - 1, n]
            refr = np.log(depth[m, n]) - np.log(depth[m + (rd + 1), n])

if __name__ == "__main__":
    developer = Developer(parser.parse_args())
    developer.develop_dynamicReceptiveField()
