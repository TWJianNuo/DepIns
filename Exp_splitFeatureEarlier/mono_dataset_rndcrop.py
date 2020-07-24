# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from kitti_utils import read_calib_file

import torch.utils.data as data
from torchvision import transforms

from utils import *

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


np.random.seed(0)
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


    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None


        # Stereo Side
        other_side = {"r": "l", "l": "r"}[side]
        inputs[("color", 's', -1)] = self.get_color(folder, frame_index, other_side, do_flip)

        # Current Side
        inputs[("color", 0, -1)] = self.get_color(folder, frame_index, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid

        K = self.get_camK
        inv_K = np.linalg.pinv(K)

        inputs[("K", 0)] = torch.from_numpy(K)
        inputs[("inv_K", 0)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        # Read The Entry tag
        comps = self.filenames[index].split(' ')
        inputs['entry_tag'] = str(comps[0] + ' ' + comps[1].zfill(10) + ' ' + comps[2])
        if do_flip:
            inputs['entry_tag'] = inputs['entry_tag'] + ' fly'
        else:
            inputs['entry_tag'] = inputs['entry_tag'] + ' fln'

        depth_gt = self.get_depth_fromfile(folder, frame_index, side, do_flip)
        inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        stereo_T = np.eye(4, dtype=np.float32)
        baseline_sign = -1 if do_flip else 1
        side_sign = -1 if side == "l" else 1
        stereo_T[0, 3] = side_sign * baseline_sign * 0.1

        inputs["stereo_T"] = torch.from_numpy(stereo_T)

        inputs.update(self.get_camK(folder, frame_index, side, do_flip))
        return inputs

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(self.data_path, folder, "image_0{}".format(self.side_map[side]), 'data', f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_camK(self, folder, side):
        cam = self.side_map[side]
        cam2cam = read_calib_file(os.path.join(self.data_path, folder.split('/')[0], 'calib_cam_to_cam.txt'))
        K = np.eye(4)
        P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
        K[0:3,:] = P_rect

        outputs = {}

        sidemap = {'l' : 2, 'r' : 3}

        calib_dir = os.path.join(self.data_path, folder.split("/")[0])
        velo_filename = os.path.join(self.data_path, folder, "velodyne_points", "data", "{:010d}.bin".format(int(frame_index)))
        cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # get image shape
        im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

        # get formation mat
        flipMat = self.get_flipMat(do_flip)
        rescaleMat = self.get_rescaleMat(im_shape)

        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_rect = cam2cam['P_rect_0' + str(sidemap[side])].reshape(3, 4)
        P_rect = np.append(P_rect, [[0, 0, 0, 1]], axis = 0)

        realIn = flipMat @ P_rect
        realEx = R_cam2rect @ velo2cam
        camK = realIn @ realEx
        invcamK = np.linalg.inv(camK)


        outputs['camK'] = camK.astype(np.float32)
        outputs['invcamK'] = invcamK.astype(np.float32)
        outputs['realIn'] = realIn.astype(np.float32)
        outputs['realEx'] = realEx.astype(np.float32)
        outputs['velo'] = velo
        return K
    def get_depth_fromfile(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
