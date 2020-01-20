# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *
import copy


class DynamicDecoder(nn.Module):
    def __init__(self, resnet, batch_size, imageHeight, num_inputs = 512, dim_reduced = 256):
        super(DynamicDecoder, self).__init__()
        # Init for Resnet Layer 5
        self.pooler = Pooler(batch_size, shrinkScale = 16, imageHeight = imageHeight)
        self.res5head = copy.deepcopy(resnet.encoder.layer4)
        # self.init_resnet_head(resnet.encoder.layer4)

        self.num_inputs = num_inputs
        self.dim_reduced = dim_reduced
        self.num_classes = 1 # We only predict whether or not is moving instance

        self.maskhead = torch.nn.Sequential()
        self.maskhead.add_module('TransConv1', ConvTranspose2d(self.num_inputs, self.dim_reduced, 2, 2, 0))
        self.maskhead.add_module('Sigmoid1', nn.Sigmoid())
        self.maskhead.add_module('TransConv2', ConvTranspose2d(self.dim_reduced, int(self.dim_reduced / 2), 2, 2, 0))
        self.maskhead.add_module('Sigmoid2', nn.Sigmoid())
        self.maskhead.add_module('Conv3', Conv2d(int(self.dim_reduced / 2), self.num_classes, 1, 1, 0))
        self.maskhead.add_module('Sigmoid3', nn.Sigmoid())

        # Check for the deepcopy function
        # list(self.layer4)[0].conv1.weight.data = list(self.layer4)[0].conv1.weight.data * 0
        # print(torch.sum(list(self.layer4)[0].conv1.weight.data))
        # print(torch.sum(list(resnet.encoder.layer4)[0].conv1.weight.data))

    # def init_resnet_head(self, resnetHead):
    #     self.head = torch.nn.Sequential()
    #     self.head.add_module('conv1', copy.deepcopy(resnetHead[0].conv1))
    #     self.head.add_module('bn1', copy.deepcopy(resnetHead[0].bn1))
    #     self.head.add_module('relu', copy.deepcopy(resnetHead[0].relu))
    #     self.head.add_module('conv2', copy.deepcopy(resnetHead[0].conv2))
    #     self.head.add_module('bn2', copy.deepcopy(resnetHead[0].bn2))
    #     # Check
    #     # torch.sum(resnetHead[0].conv1.weight) - torch.sum(self.head.conv1.weight)

    def forward(self, layer3Feature, proposals):
        outputs = {}
        # Pooling
        pooled_feature, obj_ind = self.pooler(layer3Feature, proposals)
        # Go through resnet head5
        resnet5_feature = self.res5head(pooled_feature)
        obj_prob = self.maskhead(resnet5_feature)

        outputs['insProb'] = obj_prob
        outputs['insInd'] = obj_ind
        outputs['insFeature'] = pooled_feature
        return outputs
