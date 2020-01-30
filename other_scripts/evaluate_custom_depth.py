# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import os

import argparse
import numpy as np
import PIL.Image as pil

from utils import readlines
from kitti_utils import generate_depth_map
import cv2
import time

import matplotlib.pyplot as plt
import torch
from utils import *

def get_entry_from_path(imgpath):
    comps = imgpath.split('/')
    if comps[-3] == 'image_02':
        direct = 'l'
    else:
        direct = 'r'
    entry = comps[-5] + '/' + comps[-4] + ' ' + comps[-1].split('.')[0] + ' ' + direct + '\n'
    return entry

def collect_all_entries(folder):
    import glob
    # folder = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_raw'
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    entries = list()
    for seqs_clusts in subfolders:
        seqs = [f.path for f in os.scandir(seqs_clusts) if f.is_dir()]
        for seq in seqs:
            imgFolder_02 = os.path.join(seq, 'image_02', 'data')
            imgFolder_03 = os.path.join(seq, 'image_03', 'data')
            for imgpath in glob.glob(imgFolder_02 + '/*.png'):
                entries.append(get_entry_from_path(imgpath))
            for imgpath in glob.glob(imgFolder_03 + '/*.png'):
                entries.append(get_entry_from_path(imgpath))
    return entries

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def export_gt_depths_kitti():
    gtDepthRoot = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/projected_groundtruth'
    predDepthRoot = '/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/monodepth2_prediction'

    # split_file = '/media/shengjie/other/Depins/Depins/splits/eigen/test_files.txt'
    split_file = '/media/shengjie/other/Depins/Depins/splits/sfnorm/trainA.txt'
    with open(split_file) as f:
        lines = f.readlines()
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    print("Evaluating")

    mapping = {'l': 'image_02', 'r': 'image_03'}

    ts = time.time()

    imgCount = 0

    errors = list()

    # gt_depths = np.load('/media/shengjie/other/Depins/Depins/splits/eigen/gt_depths.npz', fix_imports=True, encoding='latin1', allow_pickle = True)["data"]
    for idx, line in enumerate(lines):

        folder, frame_id, direction = line.split()

        if not os.path.isfile(os.path.join(gtDepthRoot, folder, mapping[direction], frame_id + '.png')):
            continue

        gtDepth = cv2.imread(os.path.join(gtDepthRoot, folder, mapping[direction], frame_id + '.png'), -1).astype(np.float32) / 256
        # gtDepth = gt_depths[idx]
        gt_height, gt_width = gtDepth.shape

        predDepth = cv2.imread(os.path.join(predDepthRoot, folder, mapping[direction], frame_id + '.png'), -1).astype(np.float32) / 256
        predDepth = cv2.resize(predDepth, (gt_width, gt_height), cv2.INTER_LINEAR)

        # predDepth_vis = torch.from_numpy(predDepth / 80).unsqueeze(0).unsqueeze(0)
        # tensor2disp(1 / predDepth_vis, ind=0).show()

        # rgb = pil.Image.load()

        mask = np.logical_and(gtDepth > MIN_DEPTH, gtDepth < MAX_DEPTH)

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        predDepth = predDepth[mask]
        gtDepth = gtDepth[mask]

        errors.append(compute_errors(gtDepth, predDepth))

        imgCount = imgCount + 1

        print("%d finished, %f hours left" % (imgCount, (time.time() - ts) / imgCount * (len(lines) - imgCount) / 60 / 60))

    # errrStats = np.array(errors)
    # plt.figure()
    # plt.stem(errrStats[:, 4])

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

if __name__ == "__main__":
    export_gt_depths_kitti()
