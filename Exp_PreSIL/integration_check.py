from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from Exp_PreSIL.dataloader_kitti import KittiDataset
import networks
from layers import *
from networks import *
import argparse
import shapeintegration_cuda

import torch
from kitti_utils import variancebar
from integrationModule import IntegrationFunction

integrationFunction = IntegrationFunction.apply
height = 320
width = 1024
bz = 2

sfnormOptimizer = SurfaceNormalOptimizer(height=height, width=width, batch_size=bz).cuda()
intrinsic = np.array([[7.2154e+02, 0.0000e+00, 5.9856e+02, 4.4857e+01],
                      [0.0000e+00, 7.2154e+02, 1.6785e+02, 2.1638e-01],
                      [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.7459e-03],
                      [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]])
intrinsic = torch.from_numpy(intrinsic).float().cuda()
intrinsic = intrinsic.unsqueeze(0).repeat([bz, 1, 1]).contiguous()

depth = torch.rand([bz, 1, height, width], dtype=torch.float, device=torch.device('cuda')) * 50
mask = torch.ones_like(depth).int().cuda()
semantics = torch.ones_like(depth).int().cuda()
variancebarcuda = torch.from_numpy(variancebar).cuda().float()
confidence = torch.ones_like(depth).cuda()

ang = sfnormOptimizer.depth2ang_log(depthMap=depth, intrinsic=intrinsic).cuda()
log = sfnormOptimizer.ang2log(intrinsic=intrinsic, ang=ang).cuda()

opteddepth = integrationFunction(ang, log, confidence, semantics, mask, depth, variancebarcuda, height, width, bz)

