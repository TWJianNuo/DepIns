from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image, ImageFile
from kitti_utils import read_calib_file, labels

import torch.utils.data as data
from torchvision import transforms

from utils import *

import time

ImageFile.LOAD_TRUNCATED_IMAGES = True # This can prevent some weried error
class KittiDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 gt_path,
                 filenames,
                 height,
                 width,
                 crph=365,
                 crpw=1220,
                 is_train=False,
                 gt_norm_path='None',
                 semanticspred_path='None',
                 predang_path='None'
                 ):
        super(KittiDataset, self).__init__()

        self.data_path = data_path
        self.gt_path = gt_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.is_train = is_train

        self.to_tensor = transforms.ToTensor()
        self.interp = Image.ANTIALIAS
        self.img_ext = '.png'

        if gt_norm_path is 'None':
            self.gt_norm_path = None
        else:
            self.gt_norm_path = gt_norm_path

        if semanticspred_path is 'None':
            self.semanticspred_path = None
        else:
            self.semanticspred_path = semanticspred_path
            self.regularsemanticstype = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]

        if predang_path is 'None':
            self.predang_path = None
        else:
            self.predang_path = predang_path

        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.dirmapping = {'l': 'image_02', 'r': 'image_03'}

        self.crph = crph
        self.crpw = crpw

        self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)

        self.rescaleK = np.eye(4)
        self.rescaleK[0, 0] = self.width / self.crpw
        self.rescaleK[1, 1] = self.height / self.crph

    def preprocess(self, inputs, color_aug, rndseed):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        # Do random Crop
        cropped_color = self.rndcrop_color(inputs["color"], rndseed)
        cropped_depth, intrinsic = self.rndcrop_depth(inputs["depthgt"], rndseed, inputs['K'])
        if 'normgt' in inputs:
            cropped_norm = self.rndcrop_color(inputs["normgt"], rndseed)
        if 'semanticspred' in inputs:
            resized_semanticspred = inputs["semanticspred"].resize(inputs["color"].size, pil.NEAREST)
            cropped_semanticspred = self.rndcrop_color(resized_semanticspred, rndseed)
        if 'angh' in inputs and 'angv' in inputs:
            cropped_angh = self.rndcrop_color(inputs['angh'], rndseed)
            cropped_angv = self.rndcrop_color(inputs['angv'], rndseed)


        inputs["color"] = self.resize(cropped_color)
        inputs["depthgt"] = cropped_depth

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                inputs[k] = self.to_tensor(f)
                inputs[k + "_aug"] = self.to_tensor(color_aug(f))

            elif "depthgt" in k:
                inputs[k] = np.array(f).astype(np.float32) / 256.0
                inputs[k] = torch.from_numpy(inputs[k]).unsqueeze(0)
                inputs["K"] = torch.from_numpy(intrinsic.astype(np.float32))
                inputs["K_scaled"] = torch.from_numpy((self.rescaleK @ intrinsic).astype(np.float32))

            elif "normgt" in k:
                cropped_normnp = np.array(cropped_norm)
                norm_mask = np.sum(cropped_normnp, axis=2) == 0
                norm_mask = np.stack([norm_mask, norm_mask, norm_mask], axis=2)

                cropped_normnp = (cropped_normnp.astype(np.float32) / 255.0 - 0.5) * 2
                cropped_normnp_norm = np.sqrt(np.sum(cropped_normnp ** 2, axis=2))
                cropped_normnp = cropped_normnp / np.stack([cropped_normnp_norm, cropped_normnp_norm, cropped_normnp_norm], axis=2)
                cropped_normnp[norm_mask] = 0

                inputs[k] = torch.from_numpy(cropped_normnp).permute([2, 0, 1])

            elif "semanticspred" in k:
                cropped_semanticspred_copy = np.array(cropped_semanticspred.copy())
                semanticsregularmask = np.zeros_like(cropped_semanticspred, dtype=np.float32)
                for l in np.unique(np.array(cropped_semanticspred_copy)):
                    cropped_semanticspred_copy[cropped_semanticspred_copy == l] = labels[l].trainId
                inputs[k] = torch.from_numpy(cropped_semanticspred_copy.astype(np.float32)).unsqueeze(0)

                for l in self.regularsemanticstype:
                    semanticsregularmask[cropped_semanticspred_copy == l] = 1
                inputs['semanticsregularmask'] = torch.from_numpy(semanticsregularmask).unsqueeze(0)

            elif 'angh' in k:
                cropped_anghnp = np.array(cropped_angh).astype(np.float32)
                cropped_anghnp = (cropped_anghnp / 255.0 / 255.0 - 0.5) * 2 * np.pi
                inputs[k] = torch.from_numpy(cropped_anghnp).unsqueeze(0)

            elif 'angv' in k:
                cropped_angvnp = np.array(cropped_angv).astype(np.float32)
                cropped_angvnp = (cropped_angvnp / 255.0 / 255.0 - 0.5) * 2 * np.pi
                inputs[k] = torch.from_numpy(cropped_angvnp).unsqueeze(0)



    def __len__(self):
        return len(self.filenames)

    def rndcrop_color(self, img, rndseed):
        np.random.seed(rndseed)

        w, h = img.size

        if self.is_train:
            left = np.random.randint(w - self.crpw + 1)
            top = np.random.randint(h - self.crph + 1)
        else:
            left = int((w - self.crpw) / 2)
            top = int((h - self.crph) / 2)
        imgcropped = img.crop((left, top, left + self.crpw, top + self.crph))

        return imgcropped

    def rndcrop_depth(self, img, rndseed, intrinsic):
        np.random.seed(rndseed)

        w, h = img.size

        if self.is_train:
            left = np.random.randint(w - self.crpw + 1)
            top = np.random.randint(h - self.crph + 1)
        else:
            left = int((w - self.crpw) / 2)
            top = int((h - self.crph) / 2)
        imgcropped = img.crop((left, top, left + self.crpw, top + self.crph))

        intrinsic_cropped = np.copy(intrinsic)
        intrinsic_cropped[0, 2] = intrinsic_cropped[0, 2] - left
        intrinsic_cropped[1, 2] = intrinsic_cropped[1, 2] - top

        # Check
        # xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
        # xx = xx[np.array(img) > 0]
        # yy = yy[np.array(img) > 0]
        #
        # xx_old = xx[200]
        # yy_old = yy[200]
        # d_old = float(np.array(img)[yy_old, xx_old]) / 256.0
        # pts_old = np.array([[xx_old * d_old, yy_old * d_old, d_old, 1]]).T
        # pts_old = np.linalg.inv(intrinsic) @ pts_old
        #
        # xx_new = xx_old - left
        # yy_new = yy_old - top
        # d_new = float(np.array(imgcropped)[yy_new, xx_new]) / 256.0
        # pts_new = np.array([[xx_new * d_new, yy_new * d_new, d_new, 1]]).T
        # pts_new = np.linalg.inv(intrinsic_cropped) @ pts_new
        #
        # assert np.abs(pts_new - pts_old).max() < 1e-3

        return imgcropped, intrinsic_cropped

    def __getitem__(self, index):
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()

        folder = line[0]
        frame_index = int(line[1])
        side = line[2]

        rndseed = int(time.time())

        # RGB
        inputs['color'] = self.get_color(folder, frame_index, side, do_flip)

        # Depth Map
        inputs['depthgt'] = self.get_depthgt(folder, frame_index, side, do_flip)

        if self.gt_norm_path is not None:
            inputs['normgt'] = self.get_normgt(folder, frame_index, side, do_flip)

        if self.semanticspred_path is not None:
            inputs['semanticspred'] = self.get_semanticspred(folder, frame_index, side, do_flip)

        if self.predang_path is not None:
            inputs.update(self.get_angpred(folder, frame_index, side, do_flip))

        # intrinsic parameter
        inputs['K'] = self.get_intrinsic(folder, side)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug, rndseed)

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

    def get_color(self, folder, frame_index, side, do_flip):
        color = pil.open(os.path.join(self.data_path, folder, self.dirmapping[side], "data/{}.png".format(str(frame_index).zfill(10))))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depthgt(self, folder, frame_index, side, do_flip):
        depthgt = pil.open(os.path.join(self.gt_path, folder, self.dirmapping[side], "{}.png".format(str(frame_index).zfill(10))))
        if do_flip:
            depthgt = depthgt.transpose(Image.FLIP_LEFT_RIGHT)
        return depthgt

    def get_intrinsic(self, folder, side):
        cam2cam = read_calib_file(os.path.join(self.data_path, folder.split('/')[0], 'calib_cam_to_cam.txt'))
        K = np.eye(4)
        K[0:3, :] = cam2cam['P_rect_0{}'.format(self.dirmapping[side][-1])].reshape(3, 4)
        return K

    def get_normgt(self, folder, frame_index, side, do_flip):
        normgt = pil.open(os.path.join(self.gt_norm_path, folder, self.dirmapping[side], "{}.png".format(str(frame_index).zfill(10))))
        if do_flip:
            normgt = normgt.transpose(Image.FLIP_LEFT_RIGHT)
        return normgt

    def get_semanticspred(self, folder, frame_index, side, do_flip):
        semanticspred = pil.open(os.path.join(self.semanticspred_path, folder, 'semantic_prediction', self.dirmapping[side], "{}.png".format(str(frame_index).zfill(10))))
        if do_flip:
            semanticspred = semanticspred.transpose(Image.FLIP_LEFT_RIGHT)
        return semanticspred

    def get_angpred(self, folder, frame_index, side, do_flip):
        inputs = dict()
        if do_flip:
            angh = pil.open(os.path.join(self.predang_path, "angh_flipped", folder, self.dirmapping[side], str(frame_index).zfill(10) + '.png'))
        else:
            angh = pil.open(os.path.join(self.predang_path, "angh", folder, self.dirmapping[side], str(frame_index).zfill(10) + '.png'))

        if do_flip:
            angv = pil.open(os.path.join(self.predang_path, "angv_flipped", folder, self.dirmapping[side], str(frame_index).zfill(10) + '.png'))
        else:
            angv = pil.open(os.path.join(self.predang_path, "angv", folder, self.dirmapping[side], str(frame_index).zfill(10) + '.png'))

        inputs['angh'] = angh
        inputs['angv'] = angv
        return inputs