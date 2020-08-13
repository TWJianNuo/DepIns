import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import readlines

import PIL.Image as pil
import numpy as np
import torch

train_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "eigen_full", "train_files.txt")
test_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "eigen", "test_files.txt")
ck_filenames = readlines(train_fpath) + readlines(test_fpath)

width = 1024
height = 320

dirmapping = {'l':'image_02', 'r':'image_03'}
entryck = {'htheta', 'htheta_flipped', 'vtheta', 'vtheta_flipped'}
for entry in ck_filenames:
    comps = entry.split(' ')
    for cke in entryck:
        filepath = os.path.join('/home/shengjie/Documents/Data/Kitti/kitti_theta_pred', comps[0], cke, dirmapping[comps[2]], comps[1].zfill(10) + '.png')
        if not os.path.isfile(filepath):
            print("File %s missing" % filepath)
        # else:
        #     try:
        #         theta = pil.open(filepath)
        #         theta = theta.resize([width, height], pil.BILINEAR)
        #         theta = np.array(theta).astype(np.float32) / 10 / 256
        #         theta = torch.from_numpy(theta).unsqueeze(0)
        #     except:
        #         os.remove(filepath)
        #         print("Delete: Entry: %s, cke: %s" % (entry, cke))
