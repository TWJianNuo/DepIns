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

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

np.random.seed(0)
class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png',
                 load_seman=False,
                 load_pose=False,
                 load_hints = False,
                 hints_path = '',
                 load_detect = False,
                 detect_path = '',
                 loadPredDepth = False,
                 predDepthPath = '',
                 load_syn = False,
                 syn_filenames = None,
                 syn_root = '',
                 PreSIL_root = None,
                 kitti_gt_path = None,
                 theta_gt_path=None,
                 surfnorm_gt_path=None
                 ):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

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
            self.resize[i] = transforms.Resize((self.height // s, self.width // s), interpolation=self.interp)



        self.load_depth = self.check_depth()
        self.load_detect = load_detect
        self.load_seman = load_seman
        if self.load_detect:
            self.detect_path = detect_path
            self.maxLoad = 100 # Load 100 objects per frame at most
        self.load_pose = load_pose
        self.loadPredDepth = loadPredDepth
        self.predDepthPath = predDepthPath

        self.load_hints = load_hints
        self.hints_path = hints_path

        self.load_syn = load_syn
        self.syn_root = syn_root
        self.syn_filenames = syn_filenames

        if PreSIL_root is not 'None':
            self.PreSIL_root = PreSIL_root
        else:
            self.PreSIL_root = None
        self.prsil_w = 1024
        self.prsil_h = 448
        self.prsil_cw = 32 * 10
        self.prsil_ch = 32 * 8

        if kitti_gt_path is not 'None':
            self.kitti_gt_path = kitti_gt_path
        else:
            self.kitti_gt_path = None

        if theta_gt_path is not 'None':
            self.theta_gt_path = theta_gt_path
        else:
            self.theta_gt_path = None

        if surfnorm_gt_path is not 'None':
            self.surfnorm_gt_path = surfnorm_gt_path
        else:
            self.surfnorm_gt_path = None


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

        if self.load_syn:
            inputs.update(self.get_syn_data(index, do_flip))

        self.preprocess(inputs, color_aug)

        # Read Detection Label
        imorgSize = inputs[("color", 0, -1)].shape[1:3]
        if self.load_detect:
            inputs["detect_label"] = self.get_detection(folder, frame_index, side, do_flip, imorgSize)

            # detect_label = inputs["detect_label"][inputs["detect_label"][:,0] > 0]
            # if(len(detect_label) > 0):
            #     # Create figure and axes
            #     fig, ax = plt.subplots(1)
            #
            #     # Display the image
            #     im = tensor2rgb(inputs[("color", 0, 0)].unsqueeze(0), ind=0)
            #     ax.imshow(im)
            #
            #     for k in range(len(inputs["detect_label"])):
            #         # Read Label
            #         sx = inputs["detect_label"][k][0]
            #         sy = inputs["detect_label"][k][1]
            #         rw = inputs["detect_label"][k][2] - sx
            #         rh = inputs["detect_label"][k][3] - sy
            #
            #         # Create a Rectangle patch
            #         rect = patches.Rectangle((sx, sy), rw, rh, linewidth=1, edgecolor='r', facecolor='none')
            #
            #         # Add the patch to the Axes
            #         ax.add_patch(rect)
            #     fig.savefig(os.path.join('/home/shengjie/Documents/Depins/tmp', str(index).zfill(10) + '.png'))  # save the figure to file
            #     plt.close(fig)

        # Read The Entry tag
        comps = self.filenames[index].split(' ')
        inputs['entry_tag'] = str(comps[0] + ' ' + comps[1].zfill(10) + ' ' + comps[2])
        if do_flip:
            inputs['entry_tag'] = inputs['entry_tag'] + ' fly'
        else:
            inputs['entry_tag'] = inputs['entry_tag'] + ' fln'

        for i in self.frame_idxs:
            inputs[("color", i, -1)] = F.interpolate(inputs[("color", i, -1)].unsqueeze(0), [self.full_res_shape[1], self.full_res_shape[0]], mode = 'bilinear', align_corners=True).squeeze(0)
            inputs[("color_aug", i, -1)] = F.interpolate(inputs[("color_aug", i, -1)].unsqueeze(0), [self.full_res_shape[1], self.full_res_shape[0]], mode = 'bilinear', align_corners=True).squeeze(0)
            # del inputs[("color", i, -1)]
            # del inputs[("color_aug", i, -1)]

        if self.load_depth:
            if self.kitti_gt_path is None:
                depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            else:
                try:
                    depth_gt = self.get_depth_fromfile(folder, frame_index, side, do_flip)
                except:
                    gtpath = os.path.join(folder, str(frame_index).zfill(10), side)
                    print("problem entry: %s" % gtpath)
                    import sys
                    sys.exit(0)
            # depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            # depth_gt_me = self.get_depth_fromfile(folder, frame_index, side, do_flip)
            # tensor2disp(torch.from_numpy(depth_gt).float().unsqueeze(0).unsqueeze(0), vmax = 30, ind = 0).show()
            # tensor2disp(torch.from_numpy(depth_gt_me).float().unsqueeze(0).unsqueeze(0), vmax=30, ind=0).show()
            # tensor2disp(torch.abs(torch.from_numpy(depth_gt_me) - torch.from_numpy(depth_gt).float()).unsqueeze(0).unsqueeze(0), vmax=1, ind = 0).show()
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        inputs.update(self.get_camK(folder, frame_index, side, do_flip))

        if self.load_seman:
            semanLabel, semantic_catmask = self.get_seman(folder, frame_index, side, do_flip)
            inputs['semanLabel'] = semanLabel
            inputs['semantic_catmask'] = semantic_catmask

        if self.load_pose:
            inputs['poseM'] = self.get_pose(folder, frame_index)
            inputs['bundledPoseM'] = self.get_bundlePose(folder, frame_index)

        if self.loadPredDepth:
            inputs['predDepth'] = self.get_predDepth(folder, frame_index, side, do_flip)

        if self.load_hints:
            depth_hint, depth_hint_mask = self.get_hints(folder, frame_index, side, do_flip)
            inputs["depth_hint"] = depth_hint
            inputs["depth_hint_mask"] = depth_hint_mask
        inputs['indicesRec'] = index

        if self.PreSIL_root is not None:
            # pSIL_rgb, pSIL_depth, pSIL_insMask, preSilIn, preSilEx, presil_projLidar = self.get_PreSIL()
            pSIL_rgb, pSIL_depth, pSIL_insMask, preSilIn, preSilEx = self.get_PreSIL()
            inputs["pSIL_rgb"] = pSIL_rgb
            inputs["pSIL_depth"] = pSIL_depth
            inputs["pSIL_insMask"] = pSIL_insMask
            inputs["preSilIn"] = preSilIn
            inputs["preSilEx"] = preSilEx
            # inputs["presil_projLidar"] = presil_projLidar

        if self.theta_gt_path is not None:
            inputs.update(self.get_theta_fromfile(folder, frame_index, side, do_flip))

        if self.surfnorm_gt_path is not None:
            inputs.update(self.get_surfnorm_fromfile(folder, frame_index, side, do_flip))
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_predDepth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_detection(self, folder, frame_index, side, do_flip, imorgSize):
        raise NotImplementedError

    def get_camK(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_seman(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_pose(self, folder, frame_index):
        raise NotImplementedError

    def get_bundlePose(self, folder, frame_index):
        raise NotImplementedError

    def get_hints(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_syn_data(self, index, do_flip):
        raise NotImplementedError

    def get_PreSIL(self):
        raise NotImplementedError

    def get_depth_fromfile(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_theta_fromfile(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def get_surfnorm_fromfile(self, folder, frame_index, side, do_flip):
        raise NotImplementedError