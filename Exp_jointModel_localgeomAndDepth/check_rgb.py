import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import readlines

import PIL.Image as pil
import numpy as np

from shutil import copyfile

import time

bckpath = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/kitti_raw'

train_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "eigen_full", "train_files.txt")
test_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "eigen", "test_files.txt")
ck_filenames = readlines(train_fpath) + readlines(test_fpath)

dirmapping = {'l':'image_02', 'r':'image_03'}

count = 0
st = time.time()
for entry in ck_filenames:
    comps = entry.split(' ')
    filepath = os.path.join('/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data', comps[0], dirmapping[comps[2]], 'data', comps[1].zfill(10) + '.png')
    replacepath = os.path.join(bckpath, comps[0], dirmapping[comps[2]], 'data', comps[1].zfill(10) + '.png')
    copyfile(src=replacepath, dst=filepath)
    count = count + 1

    avetime = (time.time() - st) / count
    resthrs = avetime * (len(ck_filenames) - count) / 60 / 60
    print("Rest time is %f hours, finished %d" % (resthrs, count))


