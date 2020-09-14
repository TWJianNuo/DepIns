from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

import PIL.Image as pil
from utils import *
from kitti_utils import *

def collect_all_entries(folder):
    import glob
    dates = [f.path for f in os.scandir(folder) if f.is_dir()]
    entries = list()
    for date in dates:
        seqs = [f.path for f in os.scandir(date) if f.is_dir()]
        for seq in seqs:
            foldl = os.path.join(seq, 'image_02')
            for imgpath in glob.glob(foldl + '/*.png'):
                entries.append(imgpath)
            foldr = os.path.join(seq, 'image_03')
            for imgpath in glob.glob(foldr + '/*.png'):
                entries.append(imgpath)
    return entries


dsttxtpath = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt/addresslisting.txt'
semidensegtroot = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt/'
entries = collect_all_entries(semidensegtroot)
wf = open(dsttxtpath, "w")
for entry in entries:
    wf.write(entry + '\n')