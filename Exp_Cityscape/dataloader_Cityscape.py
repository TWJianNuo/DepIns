from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
import json
from PIL import Image, ImageFile
from kitti_utils import read_calib_file, labels, translateTrainIdSemantics

import torch.utils.data as data
from torchvision import transforms

from utils import *

import time

ImageFile.LOAD_TRUNCATED_IMAGES = True # This can prevent some weried error
class CityscapeDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 max_depth,
                 is_train=False,
                 semanticspred_path=None
                 ):
        super(CityscapeDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train

        self.to_tensor = transforms.ToTensor()
        self.interp = Image.ANTIALIAS
        self.img_ext = '.png'

        self.orgh = 1024
        self.orgw = 2048

        if semanticspred_path is not None:
            self.semanticspred_path = semanticspred_path
        else:
            self.semanticspred_path = None

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        self.dirmapping = {'l': 'leftImg8bit', 'r': 'rightImg8bit'}

        self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)

        self.rescaleK = np.eye(4)
        self.rescaleK[0, 0] = self.width / self.orgh
        self.rescaleK[1, 1] = self.height / self.orgw

        self.max_depth = max_depth

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        # Do random Crop

        inputs["color"] = self.resize(inputs["color"])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                inputs[k] = self.to_tensor(f)
                inputs[k + "_aug"] = self.to_tensor(color_aug(f))

            elif "depthgt" in k:
                inputs[k] = torch.from_numpy(inputs[k]).unsqueeze(0)
            elif "K" in k:
                intrinsic = np.copy(inputs["K"])
                inputs["K"] = torch.from_numpy(intrinsic.astype(np.float32))
                inputs["K_scaled"] = torch.from_numpy((self.rescaleK @ intrinsic).astype(np.float32))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()

        cat = line[0]
        city = line[1]
        frame_index = line[2]
        side = line[3]

        # RGB
        inputs['color'] = self.get_color(cat, city, frame_index, side, do_flip)

        # intrinsic parameter
        K, bs = self.get_intrinsic(cat, city, frame_index, side)
        inputs['K'] = K

        # Depth Map
        inputs['depthgt'] = self.get_depthgt(bs, inputs['K'][0, 0], cat, city, frame_index, side, do_flip)

        if self.semanticspred_path is not None:
            inputs['semanticspred'] = self.get_semanticspred(cat, city, frame_index, side, do_flip)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        inputs['tag'] = self.initTag(index, do_flip)
        return inputs

    def initTag(self, index, do_flip):
        # Store Info of this Entry
        comps = self.filenames[index].split(' ')
        tag = str(comps[0] + ' ' + comps[1].zfill(6))
        if do_flip:
            tag = tag + ' flipYes'
        else:
            tag = tag + ' flipNoo'
        return tag

    def get_color(self, cat, city, frame_index, side, do_flip):
        color = pil.open(os.path.join(self.data_path, self.dirmapping[side], cat, city, "{}_{}_{}.png".format(city, frame_index, self.dirmapping[side])))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depthgt(self, bs, fx, cat, city, frame_index, side, do_flip):
        disparitygt = pil.open(os.path.join(self.data_path, 'disparity', cat, city, "{}_{}_{}.png".format(city, frame_index, 'disparity')))
        if do_flip:
            disparitygt = disparitygt.transpose(Image.FLIP_LEFT_RIGHT)
        disparitygt = np.array(disparitygt).astype(np.float32)
        disparitygt[disparitygt > 0] = (disparitygt[disparitygt > 0] - 1) / 256.0

        depthgt = np.copy(disparitygt)
        depthgt[depthgt > 0] = bs * fx / depthgt[depthgt > 0]
        depthgt = np.clip(depthgt, a_max=self.max_depth, a_min=-np.inf)
        return depthgt

    def get_intrinsic(self, cat, city, frame_index, side):
        calibfilepath = os.path.join(self.data_path, 'camera', cat, city, "{}_{}_{}.json".format(city, frame_index, 'camera'))
        with open(calibfilepath) as f:
            calibinfo = json.load(f)
        K = np.eye(4)
        K[0, 0] = calibinfo['intrinsic']['fx']
        K[0, 2] = calibinfo['intrinsic']['u0']
        K[1, 1] = calibinfo['intrinsic']['fy']
        K[1, 2] = calibinfo['intrinsic']['v0']
        bs = calibinfo['extrinsic']['baseline']
        return K, bs

    def get_normgt(self, folder, frame_index, side, do_flip):
        normgt = pil.open(os.path.join(self.gt_norm_path, folder, self.dirmapping[side], "{}.png".format(str(frame_index).zfill(10))))
        if do_flip:
            normgt = normgt.transpose(Image.FLIP_LEFT_RIGHT)
        return normgt

    def get_semanticspred(self, cat, city, frame_index, side, do_flip):
        return None