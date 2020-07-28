from __future__ import absolute_import, division, print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from torch.utils.data import DataLoader
from layers import *
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

import glob
import torch.optim as optim
import cv2
import numpy as np
import torch


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


def cvt_png2depth_PreSIL(tsv_depth):
    maxM = 1000
    sMax = 255 ** 3 - 1

    tsv_depth = tsv_depth.astype(np.float)
    depthIm = (tsv_depth[:, :, 0] * 255 * 255 + tsv_depth[:, :, 1] * 255 + tsv_depth[:, :, 2]) / sMax * maxM
    return depthIm

if __name__ == "__main__":
    """Evaluates a pretrained model using a specified test set
    """

    import matlab
    import matlab.engine
    eng = matlab.engine.start_matlab()

    preSILroot = '/home/shengjie/Documents/Data/PreSIL_organized'
    seq = '000000'
    frameind = 164

    intrinsic = np.array([
        [960, 0, 960],
        [0, 960, 540],
        [0, 0, 1]
    ])

    height = 448
    width = 1024
    localGeomDesp = LocalThetaDesp(height=height, width=width, batch_size=1, intrinsic=intrinsic).cuda()

    rgb = pil.open(os.path.join(preSILroot, seq, 'rgb', str(frameind).zfill(6) + '.png'))

    depth = pil.open(os.path.join(preSILroot, seq, 'depth', str(frameind).zfill(6) + '.png'))
    depth = cvt_png2depth_PreSIL(np.array(depth))

    rgb_torch = torch.from_numpy(np.array(rgb)).unsqueeze(0).permute([0,3,1,2]).float().cuda().contiguous() / 255.0
    depth_torch = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float().cuda()
    depth_torch = torch.clamp(depth_torch, min=0, max=120)

    # tensor2disp(1/depth_torch, vmax=0.5, ind=0).show()
    htheta, vtheta = localGeomDesp.get_theta(depth_torch)
    localGeomDesp.jointConstrainLoss(depth_torch, htheta, vtheta, rgb_torch, eng=eng)

