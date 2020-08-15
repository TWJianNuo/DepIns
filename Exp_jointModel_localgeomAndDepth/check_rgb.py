import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import readlines

import PIL.Image as pil
import numpy as np

from shutil import copyfile

bckpath = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'

train_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "eigen_full", "train_files.txt")
test_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "eigen", "test_files.txt")
ck_filenames = readlines(train_fpath) + readlines(test_fpath)

width = 1024
height = 320

dirmapping = {'l':'image_02', 'r':'image_03'}
for entry in ck_filenames:
    comps = entry.split(' ')
    filepath = os.path.join('/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data', comps[0], dirmapping[comps[2]], 'data', comps[1].zfill(10) + '.png')
    if not os.path.isfile(filepath):
        print("File %s missing" % filepath)
    else:
        try:
            rgb = pil.open(filepath)
            np.sum(np.array(rgb))
        except:
            print("Problematic: %s" % entry)
            replacepath = os.path.join(bckpath, comps[0], dirmapping[comps[2]], 'data', comps[1].zfill(10) + '.png')
            copyfile(src=replacepath, dst=filepath)
