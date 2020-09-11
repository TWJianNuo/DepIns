from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfaceNormalOptimizer(nn.Module):
    def __init__(self, height, width, batch_size):
        super(SurfaceNormalOptimizer, self).__init__()
        # intrinsic: (batch_size, 4, 4)
        self.height = height
        self.width = width
        self.batch_size = batch_size
        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')

        self.xx = nn.Parameter(torch.from_numpy(np.copy(xx)).unsqueeze(0).repeat([self.batch_size, 1, 1]).float(), requires_grad=False)
        self.yy = nn.Parameter(torch.from_numpy(np.copy(yy)).unsqueeze(0).repeat([self.batch_size, 1, 1]).float(), requires_grad=False)

        xx = xx.flatten().astype(np.float32)
        yy = yy.flatten().astype(np.float32)
        pix_coords = np.expand_dims(np.stack([xx, yy, np.ones(self.width * self.height).astype(np.float32)], axis=1), axis=0).repeat(self.batch_size, axis=0)
        self.pix_coords = nn.Parameter(torch.from_numpy(pix_coords).permute(0, 2, 1), requires_grad=False)
        self.ones = nn.Parameter(torch.ones([self.batch_size, 1, self.height * self.width]), requires_grad=False)
        self.init_gradconv()

    def init_gradconv(self):
        weightsx = torch.Tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        weightsx = weightsx / 4 / 2

        weightsy = torch.Tensor([[-1., -2., -1.],
                                 [0., 0., 0.],
                                 [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        weightsy = weightsy / 4 / 2
        self.diffx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.diffy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.diffx.weight = nn.Parameter(weightsx, requires_grad=False)
        self.diffy.weight = nn.Parameter(weightsy, requires_grad=False)
