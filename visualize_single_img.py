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
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from utils import *

import random

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--PreSIL_root', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    device = torch.device("cuda")
    model_path = args.model_path
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(50, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()


    mask = pil.open('/home/shengjie/Documents/Project_SemanticDepth/util/' + 'PreSIL_mask.png')
    mask = np.array(mask) > 0
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx in range(0, 51074, 5000):
            index = idx
            seq = int(index / 5000)
            rgb_path = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'rgb', "{:06d}.png".format(index))
            depth_path = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'depth', "{:06d}.png".format(index))
            ins_path = os.path.join(args.PreSIL_root, "{:06d}".format(seq), 'ins', "{:06d}.png".format(index))

            rgb = pil.open(rgb_path)
            depth_img = pil.open(depth_path)
            ins_img = pil.open(ins_path)
            ins_img = np.array(ins_img).astype(np.float)
            ins_img = ins_img[:,:,0] * 255 * 255 + ins_img[:,:,1] * 255 + ins_img[:,:,2]

            unq_inds = np.unique(ins_img)
            banned_ind = np.unique(ins_img[mask])
            pixel_nums = list()
            val_inds = list()
            for ind in unq_inds:
                if ind != 0 and (ind not in banned_ind):
                    pixel_nums.append(np.sum(ins_img == ind))
                    val_inds.append(ind)
            largest_ind = val_inds[np.argmax(np.array(pixel_nums))]
            # if np.array(pixel_nums).max() < 10000:
            #     continue

            xv, yv = np.meshgrid(range(1024), range(448))
            xmin = xv[ins_img == largest_ind].min()
            xmax = xv[ins_img == largest_ind].max()
            ymin = yv[ins_img == largest_ind].min()
            ymax = yv[ins_img == largest_ind].max()

            # tensor2disp(torch.from_numpy(ins_img == largest_ind).unsqueeze(0).unsqueeze(0), vmax=1, ind=0).show()

            # fig, ax = plt.subplots(1)
            # ax.imshow(np.array(rgb))
            # rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            # plt.show()

            xmax = xmax + 32
            xmin = xmin - 32
            ymax = ymax + 32
            ymin = ymin - 32

            imgw = 32 * 10
            imgh = 32 * 8

            padw = imgw - (xmax - xmin)
            padh = imgh - (ymax - ymin)

            for i in range(30):
                rnd_bias_w = int(round((random.random() - 0.5) * padw))
                rnd_bias_h = int(round((random.random() - 0.5) * padh))

                cx = int((xmin + xmax) / 2) + rnd_bias_w
                cy = int((ymin + ymax) / 2) + rnd_bias_h
                w = 1024
                h = 448

                lx = int(cx - imgw / 2)
                rx = int(cx + imgw / 2)
                if lx < 0:
                    lx = 0
                    rx = lx + imgw

                if rx >= w:
                    rx = w
                    lx = rx - imgw

                uy = int(cy - imgh / 2)
                by = int(cy + imgh / 2)
                if uy <= 0:
                    uy = 0
                    by = uy + imgh

                if by >= h:
                    by = h
                    uy = by - imgh

                # fig, ax = plt.subplots(1)
                # ax.imshow(np.array(rgb))
                # rect = patches.Rectangle((lx, uy), rx - lx, by - uy, linewidth=1, edgecolor='r', facecolor='none')
                # ax.add_patch(rect)
                # plt.show()

                input_image = transforms.ToTensor()(rgb).unsqueeze(0)
                input_image_cropped = input_image[:,:,uy : by , lx : rx].cuda()
                features = encoder(input_image_cropped)
                outputs = depth_decoder(features)

                disp = outputs[("disp", 0)]
                fig1 = tensor2disp(disp, ind=0, vmax=0.1)
                fig2 = tensor2rgb(input_image_cropped, ind = 0)
                fig = pil.fromarray(np.concatenate([np.array(fig1), np.array(fig2)], axis=0))
                fig.save(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/visualization/width_test', str(index) + '_' + str(i) + '.png'))


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)