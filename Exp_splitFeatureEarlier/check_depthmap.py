import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from utils import readlines
from options import MonodepthOptions
import PIL.Image as pil
import numpy as np

options = MonodepthOptions()
opts = options.parse()
train_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "semidense_eigen_full", "train_files.txt")
train_filenames = readlines(train_fpath)

dirmapping = {'l':'image_02', 'r':'image_03'}
imgssize = list()
for entry in train_filenames:
    comps = entry.split(' ')
    filepath = os.path.join(opts.kitti_gt_path, comps[0], dirmapping[comps[2]], comps[1].zfill(10) + '.png')
    if not os.path.isfile(filepath):
        print("File %s missing" % filepath)
    else:
        imgssize.append(np.array(pil.open(filepath).size))

imgssizenp = np.stack(imgssize, axis=0)
print(np.unique(imgssizenp, axis=0))