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

    # For the ordinal encoding, first channel is the mask, then we encode 8 bits
    h = 448
    w = 1024
    mask = pil.open('/home/shengjie/Documents/Project_SemanticDepth/util/' + 'PreSIL_mask.png')
    mask = np.array(mask) > 0
    xv, yv = np.meshgrid(range(w), range(h))

    depmax = 25
    pixelcountmin = 70
    fill_rate = 0.3

    st = time.time()

    feel_loc = np.array(
        [
            [0,1], [0,-1], [1,0], [-1,0], [1,1], [1,-1], [-1,1], [-1,-1]
         ]
    )
    # feel_loc = np.array(
    #     [
    #         [0,2], [0,-2], [2,0], [-2,0], [2,2], [2,-2], [-2,2], [-2,-2]
    #      ]
    # )
    feel_loc = np.concatenate([feel_loc,feel_loc*2,feel_loc*4],axis=0)

    w = 4
    weight = np.zeros([feel_loc.shape[0],1,int(w*2+1),int(w*2+1)])
    for i in range(feel_loc.shape[0]):
        weight[i, 0, feel_loc[i,1] + w, feel_loc[i,0] + w] = 1
        weight[i, 0, w, w] = -1
    compareConv = torch.nn.Conv2d(1, feel_loc.shape[0], int(w*2+1), stride=1, padding=w, bias=False)
    compareConv.weight = torch.nn.Parameter(torch.from_numpy(weight), requires_grad = False)
    compareConv = compareConv.cuda()

    for idx in range(0, 51075):
        index = idx
        seq = int(index / 5000)

        rgb_path = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'rgb', "{:06d}.png".format(index))
        depth_path = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'depth', "{:06d}.png".format(index))
        ins_path = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'ins', "{:06d}.png".format(index))
        label_path = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'boxlabels', "{:06d}.txt".format(index))

        order_root = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'orderlabels')
        os.makedirs(order_root, exist_ok=True)

        ordervls_root = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'ordervls')
        os.makedirs(ordervls_root, exist_ok=True)

        # rgb = pil.open(rgb_path)
        depth_img = np.array(pil.open(depth_path))
        depth_img = cvt_png2depth_PreSIL(depth_img)


        depth_img_gpu = torch.from_numpy(depth_img).unsqueeze(0).unsqueeze(0).cuda()
        # plt.figure()
        # plt.imshow(0.1 / depth_img / 100, cmap = 'inferno')
        # plt.show()
        # tensor2disp(0.1 / depth_img_gpu, percentile=95, ind=0).show()
        cmpre = compareConv(depth_img_gpu) > 0
        # cmpre_np = cmpre[0].permute([1,2,0]).contiguous().cpu().numpy().astype(np.uint8)
        # cmpre_rgb = np.packbits(cmpre_np, axis=2)
        # pil.fromarray(cmpre_rgb).save(os.path.join(order_root, "{:06d}.png".format(index)))
        #
        # ins_img = pil.open(ins_path)
        # ins_img = np.array(ins_img).astype(np.float)
        # ins_img = ins_img[:,:,0] * 255 * 255 + ins_img[:,:,1] * 255 + ins_img[:,:,2]
        # with open(label_path) as f:
        #     lines = f.readlines()
        # if len(lines) > 0:
        #     centerptsx = list()
        #     centerptsy = list()
        #
        #     farptsx = list()
        #     farptsy = list()
        #
        #     nearptsx = list()
        #     nearptsy = list()
        #     for l in lines:
        #         objind = int(l.split(' ')[0])
        #         mask_selector = ins_img == objind
        #         xx = xv[mask_selector]
        #         yy = yv[mask_selector]
        #         rndind = random.randint(1,xx.shape[0]-1)
        #
        #         # depth_img[yy- 4: yy+4, xx-4:xx+4]
        #         xx = xx[rndind]
        #         yy = yy[rndind]
        #         centerptsx.append(xx)
        #         centerptsy.append(yy)
        #
        #         for k in range(feel_loc.shape[0]):
        #             if cmpre_np[yy, xx, k] == True:
        #                 farptsx.append(xx + feel_loc[k, 0])
        #                 farptsy.append(yy + feel_loc[k, 1])
        #             else:
        #                 nearptsx.append(xx + feel_loc[k, 0])
        #                 nearptsy.append(yy + feel_loc[k, 1])
        #
        #     dr = ImageDraw.Draw(rgb)
        #     r = 1
        #     for k in range(len(centerptsx)):
        #         dr.ellipse((centerptsx[k] - r, centerptsy[k] - r, centerptsx[k] + r, centerptsy[k] + r), 'blue')
        #
        #     for k in range(len(farptsx)):
        #         dr.ellipse((farptsx[k] - r, farptsy[k] - r, farptsx[k] + r, farptsy[k] + r), 'red')
        #
        #     for k in range(len(nearptsx)):
        #         dr.ellipse((nearptsx[k] - r, nearptsy[k] - r, nearptsx[k] + r, nearptsy[k] + r), 'green')
        #
        # rgb.save(os.path.join(ordervls_root, "{:06d}.png".format(index)))
        dr = time.time() - st
        print("%d Finished, %f hours left" % (idx, dr / (idx + 1) * (51075 - idx) / 60 / 60))



if __name__ == '__main__':
    args = parse_args()
    test_simple(args)