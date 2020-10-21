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
            anghnp = anggt[0,0,:,:].detach().cpu().numpy()
            angvnp = anggt[0,1,:,:].detach().cpu().numpy()
            evaluateReceptiveFieldVariance(height=self.opt.height, width=self.opt.width, logh=loghnp, logv=logvnp,
                                           depthpredl=np.log(depthnp), depthgtl=np.log(depthnp), shapeh=anghnp, shapev=angvnp)
        return

def evaluateReceptiveFieldVariance(height, width, logh, logv, depthpredl, depthgtl, shapeh, shapev):
    m = np.random.randint(0, height)
    n = np.random.randint(0, width)

    inthr = 0
    inthl = 0
    counth = 1
    squaresumh = shapeh[m, n] ** 2
    sumh = shapeh[m, n]
    breakl = False
    breakr = False
    le = n
    re = n
    for ih in range(width * 2):
        if ih % 2 == 0:
            if breakr:
                continue
            sn = n + int(ih / 2) + 1
            if sn >= width:
                breakr = True
                continue
            else:
                inthr = inthr + logh[m, sn-1]
                squaresumh = squaresumh + shapeh[m, sn]**2
                sumh = sumh + shapeh[m, sn]
                counth = counth + 1
                varh = (squaresumh / counth - (sumh / counth) ** 2)
                gth = depthgtl[m, sn] - depthgtl[m, n]
                re = sn
                assert np.abs(inthr - gth) < 1e-1
                assert np.abs(np.var(shapeh[m, le:re+1]) - varh) < 1e-2
        else:
            if breakl:
                continue
            sn = n - int(ih / 2) - 1
            if sn < 0:
                breakl = True
                continue
            else:
                inthl = inthl - logh[m, sn]
                squaresumh = squaresumh + shapeh[m, sn]**2
                sumh = sumh + shapeh[m, sn]
                counth = counth + 1
                varh = (squaresumh / counth - (sumh / counth) ** 2)
                gth = depthgtl[m, sn] - depthgtl[m, n]
                le = sn
                assert np.abs(inthl - gth) < 1e-1
                assert np.abs(np.var(shapeh[m, le:re+1]) - varh) < 1e-2

        intvu = 0
        intvd = 0
        countv = 1
        squaresumv = shapev[m, n] ** 2
        sumv = shapev[m, n]
        breaku = False
        breakd = False
        ue = m
        de = m
        for iv in range(height * 2):
            if iv % 2 == 0:
                if breakd:
                    continue
                sm = m + int(iv / 2) + 1
                if sm >= height:
                    breakd = True
                    continue
                else:
                    intvd = intvd + logv[sm-1, n]
                    squaresumv = squaresumv + shapev[sm, n]**2
                    sumv = sumv + shapev[sm, n]
                    countv = countv + 1
                    varv = (squaresumv / countv - (sumv / countv) ** 2)
                    gtv = depthgtl[sm, n] - depthgtl[m, n]
                    de = sm

                    assert np.abs(intvd - gtv) < 1e-1
                    assert np.abs(np.var(shapev[ue:de+1, n]) - varv) < 1e-2

            else:
                if breaku:
                    continue
                sm = m - int(iv / 2) - 1
                if sm < 0:
                    breaku = True
                    continue
                else:
                    intvu = intvu - logv[sm, n]
                    squaresumv = squaresumv + shapev[sm, n]**2
                    sumv = sumv + shapev[sm, n]
                    countv = countv + 1
                    varv = (squaresumv / countv - (sumv / countv) ** 2)
                    gtv = depthgtl[sm, n] - depthgtl[m, n]
                    ue = sm
                    assert np.abs(intvu - gtv) < 1e-1
                    assert np.abs(np.var(shapev[ue:de+1, n]) - varv) < 1e-2
    return

if __name__ == "__main__":
    developer = Developer(parser.parse_args())
    developer.develop_dynamicReceptiveField()
