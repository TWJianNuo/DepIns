import numpy as np
import cv2
import cvbase as cvb
import os
import pycocotools.mask as maskUtils
from utils import *
import PIL.Image as pil
from PIL import Image, ImageFile
import time
ImageFile.LOAD_TRUNCATED_IMAGES = True

instanceroot = '/home/shengjie/Documents/Data/organized_kins/from_semankitti'
splitroot = '/home/shengjie/Documents/Project_SemanticDepth/splits/semidense_eigen_full'
dstsplitroot = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full_withins'
splitentry = ['test', 'train', 'val']
dirmapping = {'l':'image_02', 'r':'image_03'}

os.makedirs(dstsplitroot, exist_ok=True)

for split in splitentry:
    entries = readlines(os.path.join(splitroot, "{}_files.txt".format(split)))
    entrywithins = list()
    for entry in entries:
        seq, index, d = entry.split(' ')
        if os.path.exists(os.path.join(instanceroot, os.path.join(instanceroot, seq, dirmapping[d], "{}.png".format(index.zfill(10))))):
            entrywithins.append("{} {} {}".format(seq, index.zfill(10), d))

    with open(os.path.join(dstsplitroot, "{}_files.txt".format(split)), "w") as text_file:
        for entry in entrywithins:
            text_file.write(entry + '\n')
