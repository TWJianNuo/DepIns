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

def savelabel(producedinstancelabel, rgb, figname, vlsroot_rgb, outputroot, cam):
    date = figname[0:10]
    seq = "{}_sync".format(figname[0:21])
    ind = figname.split('_')[-1]
    os.makedirs(os.path.join(outputroot, date, seq, cam), exist_ok=True)
    os.makedirs(os.path.join(vlsroot_rgb, date, seq, cam), exist_ok=True)
    assert producedinstancelabel.max() < 255
    pil.fromarray(producedinstancelabel.astype(np.uint8)).save(os.path.join(outputroot, date, seq, cam, "{}.png".format(ind)))

    coloredimg = np.zeros_like(rgb)
    ratio = 0.3
    for k in np.unique(producedinstancelabel):
        if k > 0:
            selector = producedinstancelabel == k
            coloredimg[selector, :] = np.repeat((np.random.random([1, 3]) * 255).astype(np.uint8), np.sum(selector), axis=0)
    combined = rgb.astype(np.float) * (1-ratio) + coloredimg.astype(np.float)*ratio
    combined = combined.astype(np.uint8)
    pil.fromarray(combined).save(os.path.join(vlsroot_rgb, date, seq, cam, "{}.png".format(ind)))

kittisemanticsroot = '/home/shengjie/Documents/Data/KittiInstance/kitti_semantics'
mappingfilepath = '/home/shengjie/Documents/Data/KittiInstance/kitti_semantics/training_mapping.txt'
dstroot = '/home/shengjie/Documents/Data/organized_kins/from_semantics'
vlsroot = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/producedInstanceLabel_semantics'
kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
mappings = readlines(mappingfilepath)

count = 0
for entry in mappings:
    if entry != '':
        semanticsfilepath = os.path.join(kittisemanticsroot, 'training', 'instance', "{}_10.png".format(str(count).zfill(6)))
        instancelabel = np.array(pil.open(semanticsfilepath)).astype(np.int) % 256

        date, seq, index = entry.split(' ')
        img_path = os.path.join(kittiroot, date, seq, 'image_02', 'data', "{}.png".format(index.zfill(10)))
        rgb = pil.open(img_path)

        savelabel(instancelabel, np.array(rgb), "{}_{}".format(seq, index), vlsroot, dstroot, 'image_02')
    count = count + 1