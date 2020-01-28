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

def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)

    parser.add_argument('--save_dir',
                        type=str,
                        help='path to the root of save folder',
                        required=True)

    opt = parser.parse_args()

    lines = collect_all_entries(opt.data_path)

    print("Exporting ground truth depths")

    mapping = {'l': 'image_02', 'r': 'image_03'}

    ts = time.time()

    imgCount = 0

    for line in lines:

        folder, frame_id, direction = line.split()
        frame_id = int(frame_id)

        calib_dir = os.path.join(opt.data_path, folder.split("/")[0])

        velo_filename = os.path.join(opt.data_path, folder,
                                     "velodyne_points/data", "{:010d}.bin".format(frame_id))
        if not os.path.isfile(velo_filename):
            continue

        gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        gt_depth = cv2.resize(gt_depth, (1242, 375), cv2.INTER_NEAREST)

        gt_depth = np.uint16(gt_depth * 256)

        output_folder = os.path.join(opt.save_dir, folder, mapping[direction])

        os.makedirs(output_folder, exist_ok=True)

        save_path = os.path.join(output_folder, str(frame_id).zfill(10) + '.png')

        cv2.imwrite(save_path, gt_depth)

        te = time.time()

        imgCount = imgCount + 1

        print("%d finished, %f hours left" % (imgCount, (te - ts) / imgCount * (len(lines) - imgCount) / 60 / 60))
if __name__ == "__main__":
    export_gt_depths_kitti()
