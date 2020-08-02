from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image
from kitti_utils import read_calib_file

import torch.utils.data as data
from torchvision import transforms

from utils import *

import time

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MonoDatasetRndCrop(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 is_train=False,
                 kitti_gt_path=None
                 ):
        super(MonoDatasetRndCrop, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train
        self.kitti_gt_path = kitti_gt_path

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.interp = Image.ANTIALIAS
        self.img_ext = '.png'
        self.side_map = {'l':2, 'r':3}

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.crph = 365
        self.crpw = 1220

        self.K = np.array([[0.58 * self.crpw, 0, 0.5 * self.crpw, 0],
                           [0, 1.92 * self.crph, 0.5 * self.crph, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)
    def preprocess(self, inputs, color_aug, rndseed):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.rndCropper(f, rndseed)
                inputs[(n, im, 0)] = self.resize(inputs[(n, im, i)])
            elif "depthgt" in k:
                inputs[k] = self.rndCropper(inputs['depthgt'], rndseed)

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                if i == 0:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

            elif "depthgt" in k:
                inputs[k] = np.array(f).astype(np.float32) / 256.0
                inputs[k] = torch.from_numpy(inputs[k]).unsqueeze(0)


    def __len__(self):
        return len(self.filenames)

    def rndCropper(self, img, rndseed):
        np.random.seed(rndseed)

        w, h = img.size
        left = np.random.randint(w - self.crpw + 1)
        top = np.random.randint(h - self.crph + 1)
        imgcropped = img.crop((left, top, left + self.crpw, top + self.crph))

        return imgcropped

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()

        folder = line[0]
        frame_index = int(line[1])
        side = line[2]

        rndseed = int(time.time())

        # Stereo Side
        other_side = {"r": "l", "l": "r"}[side]
        inputs[("color", 's', -1)] = self.get_color(folder, frame_index, other_side, do_flip)

        # Main Side
        inputs[("color", 0, -1)] = self.get_color(folder, frame_index, side, do_flip)

        # Depth Map
        inputs['depthgt'] = self.get_depthgt(folder, frame_index, side, do_flip)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug, rndseed)

        inputs['tag'] = self.initTag(index, do_flip)

        inputs['K'] = self.K

        return inputs

    def initTag(self, index, do_flip):
        # Store Info of this Entry
        comps = self.filenames[index].split(' ')
        tag = str(comps[0] + ' ' + comps[1].zfill(10) + ' ' + comps[2])
        if do_flip:
            tag = tag + ' fly'
        else:
            tag = tag + ' fln'
        return tag

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, "image_0{}".format(self.side_map[side]), 'data', f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depthgt(self, folder, frame_index, side, do_flip):
        rgb_path = os.path.join(self.kitti_gt_path, folder, "image_0{}".format(self.side_map[side]), "{:010d}.png".format(frame_index))
        depthgt = pil.open(rgb_path)
        if do_flip:
            depthgt = depthgt.transpose(Image.FLIP_LEFT_RIGHT)
        return depthgt
