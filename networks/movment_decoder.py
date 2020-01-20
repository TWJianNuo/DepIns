# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
from layers import *


class MovementDecoder(nn.Module):
    def __init__(self, batch_size, imageHeight):
        super(MovementDecoder, self).__init__()

        self.pooler = Pooler(batch_size, shrinkScale=16, imageHeight=imageHeight)

        self.movConv = torch.nn.Sequential()
        self.movConv.add_module('Conv1', nn.Conv2d(512, 512, 3, stride=1, padding=1))
        self.movConv.add_module('Sigmoid1', nn.Sigmoid())
        self.movFcn = torch.nn.Sequential()
        self.movFcn.add_module('Fcn1', nn.Linear(512, 6, bias=True))

    def forward(self, pose_features, rois, insfeture):
        # Pooling
        pooled_feature, obj_ind = self.pooler(pose_features, rois)
        # Concat with ins feature
        pooled_feature = torch.cat([pooled_feature, insfeture], dim=1)

        pooled_feature = self.movConv(pooled_feature)
        pooled_feature = pooled_feature.mean(2).mean(2)
        pooled_feature = self.movFcn(pooled_feature)

        pooled_feature = 0.01 * pooled_feature.view(-1, 1, 1, 6)
        axisangle = pooled_feature[..., :3]
        translation = pooled_feature[..., 3:]

        return axisangle, translation








