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

def savelabel(producedinstancelabel, figname, outputroot, cam):
    date = figname[0:10]
    seq = "{}_sync".format(figname[0:21])
    ind = figname.split('_')[-1].zfill(10)
    os.makedirs(os.path.join(outputroot, date, seq, cam), exist_ok=True)
    producedinstancelabel.save(os.path.join(outputroot, date, seq, cam, "{}.png".format(ind)))

instanceroot = '/home/shengjie/Documents/Data/organized_kins'
dstroot = '/home/shengjie/Documents/Data/organized_kins/from_all'
splitroot = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full'
splitentry = ['test', 'train', 'val']
lookuporder = ['from_semantics', 'from_kins', 'from_tracking', 'from_semankitti']
dirmapping = {'l':'image_02', 'r':'image_03'}

totnum = 0
for split in splitentry:
    entries = readlines(os.path.join(splitroot, "{}_files.txt".format(split)))
    totnum = totnum + len(entries)

st = time.time()

count = 0
for split in splitentry:
    entries = readlines(os.path.join(splitroot, "{}_files.txt".format(split)))
    for entry in entries:
        seq, index, direction = entry.split(' ')
        labelpath = os.path.join(seq, dirmapping[direction], "{}.png".format(index.zfill(10)))
        for source in lookuporder:
            curlabelpath = os.path.join(instanceroot, source, labelpath)
            if os.path.exists(curlabelpath):
                savelabel(pil.open(curlabelpath), "{}_{}".format(seq.split('/')[1], index), dstroot, dirmapping[direction])
                break
        count = count + 1
        dr = time.time() - st
        leftime = (totnum - count) * (dr / count) / 60 / 60
        print("Finished: %d, remains hours: %f" % (count, leftime))


