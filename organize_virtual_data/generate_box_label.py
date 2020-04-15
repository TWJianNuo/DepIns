# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
from PIL import Image, ImageDraw
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from utils import *

import random
import time
def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--PreSIL_root', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    return parser.parse_args()


def cvt_png2depth_PreSIL(tsv_depth):
    maxM = 1000
    sMax = 255 ** 3 - 1

    tsv_depth = tsv_depth.astype(np.float)
    depthIm = (tsv_depth[:,:,0] * 255 * 255 + tsv_depth[:,:,1] * 255 + tsv_depth[:,:,2]) / sMax * maxM
    return depthIm

def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    mask = pil.open('/home/shengjie/Documents/Project_SemanticDepth/util/' + 'PreSIL_mask.png')
    mask = np.array(mask) > 0
    xv, yv = np.meshgrid(range(1024), range(448))

    depmax = 25
    pixelcountmin = 70
    fill_rate = 0.3
    # pixelcountmin = 0

    st = time.time()
    for idx in range(3059, 51075):
        index = idx
        seq = int(index / 5000)

        label_root = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'boxlabels')
        os.makedirs(label_root, exist_ok=True)
        box_vls_root = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'boxvls')
        os.makedirs(box_vls_root, exist_ok=True)
        rgb_path = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'rgb', "{:06d}.png".format(index))
        depth_path = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'depth', "{:06d}.png".format(index))
        ins_path = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'ins', "{:06d}.png".format(index))

        f = open(os.path.join(label_root, "{:06d}.txt".format(index)), 'w')
        rgb = pil.open(rgb_path)
        depth_img = np.array(pil.open(depth_path))
        depth_img = cvt_png2depth_PreSIL(depth_img)
        ins_img = pil.open(ins_path)
        ins_img = np.array(ins_img).astype(np.float)
        ins_img = ins_img[:,:,0] * 255 * 255 + ins_img[:,:,1] * 255 + ins_img[:,:,2]

        # tensor2disp(torch.from_numpy(ins_img > 0).unsqueeze(0).unsqueeze(0), vmax=1, ind=0).show()

        self_ind_count = list()
        self_ind = list()
        self_indc = np.unique(ins_img[mask])
        for si in self_indc:
            if si != 0:
                self_ind.append(si)
                self_ind_count.append(np.sum(ins_img == si))
        if len(self_ind_count) == 0:
            self_ind = 0
        else:
            self_ind = self_ind[np.argmax(self_ind_count)]

        unq_inds = np.unique(ins_img)
        # tensor2disp(torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0), ind=0, percentile=93).show()
        # for k in unq_inds:
        #     tensor2disp(torch.from_numpy(ins_img == k).unsqueeze(0).unsqueeze(0), ind = 0, vmax=1).show()
        #     print(k)
        #     input("Press Enter to continue...")

        # dices_mask = ins_img == 9338760
        # tensor2disp(torch.from_numpy(ins_img == 9338760).unsqueeze(0).unsqueeze(0), ind=0, vmax=1).show()
        # dices_mask = ins_img == 3861090
        # print("%f, %f" % (np.mean(depth_img[dices_mask]), np.sum(dices_mask)))
        for ind in unq_inds:
            if ind != 0 and ind != self_ind:
                dices_mask = ins_img == ind
                # print("%f, %f" % (np.mean(depth_img[dices_mask]), np.sum(dices_mask)))
                if depth_img[dices_mask].min() < depmax and np.sum(dices_mask) > pixelcountmin:

                    xmin = xv[dices_mask].min()
                    xmax = xv[dices_mask].max()
                    ymin = yv[dices_mask].min()
                    ymax = yv[dices_mask].max()

                    cur_fill_rate = np.sum(dices_mask) / ((xmax - xmin) * (ymax - ymin))
                    if cur_fill_rate > fill_rate:
                        drawhanlde = ImageDraw.Draw(rgb)
                        drawhanlde.rectangle([xmin, ymin, xmax, ymax], outline="red")

                        f.write('%d %d %d %d %d\n' % (int(ind), int(xmin), int(xmax), int(ymin), int(ymax)))

        rgb.save(os.path.join(box_vls_root, "{:06d}.png".format(index)))
        f.close()

        dr = time.time() - st
        print("%d Finished, %f hours left" % (idx, dr / (idx + 1) * (51075 - idx) / 60 / 60))


        # rgb.show()

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)