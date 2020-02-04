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


import torch
import torch.utils.data as data
from torchvision import transforms

from utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from kitti_utils import read_calib_file, load_velodyne_points
from kitti_utils import labels
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class SFGAN_Base_Dataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
    """
    def __init__(self,
                 data_path,
                 filenames,
                 syn_filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 opts,
                 is_train=False,
                 load_seman = False,
                 ):
        super(SFGAN_Base_Dataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.syn_filenames = syn_filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
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

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_seman = load_seman
        self.opts = opts

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

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
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
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

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        inputs.update(self.get_syn_data(index))

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        # Read The Entry tag

        inputs['entry_tag'] = self.acquire_load_info(index=index, do_flip = do_flip, do_aug=do_color_aug)


        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # Load Depth data
        inputs["depth_gt"] = self.get_depth(folder, frame_index, side, do_flip)

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        inputs.update(self.get_camK(folder, frame_index, side, do_flip))

        if self.load_seman:
            inputs['real_semanLabel'] = self.get_seman_real(folder, frame_index, side, do_flip)

        return inputs

    def acquire_load_info(self, index, do_flip, do_aug):
        folder, ind, dir = self.filenames[index].split(' ')
        return str('Folder: ' + folder + '\nFrame_Index: ' + ind.zfill(10) + '\nDirection: ' + dir + '\nIndex: ' +str(index).zfill(10) + '\nDo_flip: ' + str(do_flip) + '\nDo_aug: ' + str(do_aug))

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, '.png')
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        gt_depth_path = os.path.join(self.opts.gtDepthRoot, folder, "image_0{}".format(self.side_map[side]), str(frame_index).zfill(10) + '.png')
        gt_depth = cv2.imread(gt_depth_path, -1)

        if gt_depth is None:
            gt_depth = np.zeros(self.full_res_shape[::-1], dtype=np.float32)
        else:
            gt_depth = gt_depth.astype(np.float32) / 256
            gt_depth = cv2.resize(gt_depth, self.full_res_shape, interpolation = cv2.INTER_NEAREST)

        if do_flip:
            gt_depth = np.copy(np.fliplr(gt_depth))

        gt_depth = np.expand_dims(gt_depth, axis=0)

        return gt_depth

    def get_flipMat(self, do_flip):
        if do_flip:
            flipMat = np.array([[-1, 0, self.width, 1],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
        else:
            flipMat = np.eye(4)
        return flipMat

    def get_rescaleMat(self, height, width, org_height, org_width):
        fx = width / org_width
        fy = height / org_height
        rescaleMat = np.array([[fx, 0, 0, 0], [0, fy, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return rescaleMat

    def get_velo(self, velo_filename):
        maxNum = 100000
        velo = np.ones([maxNum, 4]).astype(np.float32) * (-10.0)

        velo_pts = load_velodyne_points(velo_filename)
        velo_pts = velo_pts[velo_pts[:, 0] > 0, :]

        np.random.shuffle(velo_pts)
        copyNum = np.min([velo_pts.shape[0], velo.shape[0]])
        velo[0 : copyNum, :] = velo_pts[0 : copyNum, :]

        return velo

    def get_camK(self, folder, frame_index, side, do_flip):
        outputs = {}

        calib_dir = os.path.join(self.data_path, folder.split("/")[0])
        velo_filename = os.path.join(self.data_path, folder, "velodyne_points", "data",
                                     "{:010d}.bin".format(int(frame_index)))
        cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # get formation mat
        flipMat = self.get_flipMat(do_flip)

        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_rect = cam2cam['P_rect_0' + str(self.side_map[side])].reshape(3, 4)
        P_rect = np.append(P_rect, [[0, 0, 0, 1]], axis = 0)

        # get image shape
        im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)
        for i in range(0, self.num_scales):
            height = self.opts.height / np.power(2, i)
            width = self.opts.width / np.power(2, i)
            rescaleMat = self.get_rescaleMat(height=height, width=width, org_height=im_shape[0], org_width=im_shape[1])
            realIn = flipMat @ rescaleMat @ P_rect
            realEx = R_cam2rect @ velo2cam
            camK = realIn @ realEx
            invcamK = np.linalg.inv(camK)

            outputs[('camK', i)] = camK.astype(np.float32)
            outputs[('invcamK', i)] = invcamK.astype(np.float32)
            outputs[('realIn', i)] = realIn.astype(np.float32)
            outputs[('realEx', i)] = realEx.astype(np.float32)

        if self.opts.loadVelo:
            velo = self.get_velo(velo_filename)
            outputs['velo'] = velo

        return outputs

    def get_seman_real(self, folder, frame_index, side, do_flip):
        seman_real_path = os.path.join(self.opts.data_path, folder, 'semantic_prediction', "image_0{}".format(self.side_map[side]), str(frame_index).zfill(10) + '.png')

        semantic_label = pil.open(seman_real_path)

        # Do resize
        semantic_label = pil.Image.resize(semantic_label, [self.opts.width, self.opts.height], resample = Image.NEAREST)

        # Do flip
        if do_flip:
            semantic_label = semantic_label.transpose(pil.FLIP_LEFT_RIGHT)
        semantic_label_copy = np.array(semantic_label.copy())

        # Do label transformation
        for k in np.unique(semantic_label):
            semantic_label_copy[semantic_label_copy == k] = labels[k].trainId

        # visualize_semantic(semantic_label_copy)
        semantic_label_copy = np.expand_dims(semantic_label_copy, axis=0)
        return semantic_label_copy

    def get_seman_syn(self, folder, frame_index, do_flip):
        seman_real_path = os.path.join(self.opts.synRoot, folder, 'scene_label', str(frame_index).zfill(4) + '.png')

        semantic_label = pil.open(seman_real_path)

        # Do resize
        semantic_label = pil.Image.resize(semantic_label, [self.opts.width, self.opts.height], resample = Image.NEAREST)

        # Do flip
        if do_flip:
            semantic_label = semantic_label.transpose(pil.FLIP_LEFT_RIGHT)
        semantic_label_copy = np.array(semantic_label.copy())

        # Do label transformation
        for k in np.unique(semantic_label):
            semantic_label_copy[semantic_label_copy == k] = labels[k].trainId

        # visualize_semantic(semantic_label_copy)
        semantic_label_copy = np.expand_dims(semantic_label_copy, axis=0)

        return semantic_label_copy

    def get_syn_data(self, index):
        do_flip = self.is_train and random.random() > 0.5
        index = index % len(self.syn_filenames)

        inputs = {}
        seq, frame_ind, _ = self.syn_filenames[index].split(' ')

        # Read RGB
        B_path = os.path.join(self.opts.synRoot, seq, 'rgb', frame_ind + '.png')
        B_rgb = np.array(Image.open(B_path).convert('RGB')).astype(np.float32) / 255

        # Read Depth
        B_path = os.path.join(self.opts.synRoot, seq, 'depthgt', frame_ind + '.png')
        B_depth = np.array(cv2.imread(B_path, -1)).astype(np.float32) / 100 # 1 intensity inidicates 1 cm, max is 655.35 meters

        # Read Semantic Label
        B_semanLabel = self.get_seman_syn(folder = seq, frame_index=frame_ind, do_flip = do_flip)
        if do_flip:
            B_rgb = np.copy(np.fliplr(B_rgb))
            B_depth = np.copy(np.fliplr(B_depth))
            B_semanLabel = np.copy(np.fliplr(B_semanLabel ))
        inputs[('syn_rgb', 0)] = np.moveaxis(cv2.resize(B_rgb, (self.opts.width, self.opts.height), interpolation = cv2.INTER_LINEAR), [0,1,2], [1,2,0])
        inputs[('syn_depth', 0)] = np.expand_dims(cv2.resize(B_depth, (self.opts.width, self.opts.height), interpolation = cv2.INTER_LINEAR), axis=0)
        inputs['syn_semanLabel'] = B_semanLabel
        for i in range(1, self.num_scales):
            inputs[('syn_depth', i)] = np.expand_dims(cv2.resize(B_depth, (int(self.opts.width / np.power(2,i)), int(self.opts.height / np.power(2,i))), interpolation = cv2.INTER_LINEAR), axis=0)

        return inputs