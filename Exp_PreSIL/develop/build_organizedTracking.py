import numpy as np
import cv2
import cvbase as cvb
import os
import pycocotools.mask as maskUtils
from utils import *
import PIL.Image as pil
from PIL import Image, ImageFile
import time
import glob
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

def acquire_mapping(kittitrackingroot, kittiroot):
    seqs = glob.glob(os.path.join(kittitrackingroot, '*/'))

    kittiseqs = list()
    days = glob.glob(os.path.join(kittiroot, '*/'))
    for day in days:
        kittiseqs = kittiseqs + glob.glob(os.path.join(day, '*/'))

    mapping = list()
    for seq in seqs:
        sampleindex = 2
        imgsampetracking = np.array(pil.open(os.path.join(seq, "{}.png".format(str(sampleindex).zfill(6)))))

        diffarr = list()
        seqnamearr = list()
        for kittiseq in kittiseqs:
            imgsamperaw = np.array(pil.open(os.path.join(kittiseq, 'image_02', 'data', "{}.png".format(str(sampleindex).zfill(10)))))
            if imgsampetracking.shape != imgsamperaw.shape:
                continue
            diff = np.abs(imgsampetracking - imgsamperaw).mean()
            diffarr.append(diff)
            seqnamearr.append(kittiseq.split('/')[-2])
        diffarr = np.array(diffarr)
        minind = np.argmin(diffarr)

        if diffarr.min() < 0.1:
            mapentry = (seq.split('/')[-2], seqnamearr[minind])
            mapping.append(mapentry)
    return mapping

def gettotnum(kittitrackingroot, mappings):
    totnum = 0
    for mapentry in mappings:
        totnum = totnum + len(glob.glob(os.path.join(kittitrackingroot, 'instances', mapentry[0], '*.png')))
    return totnum

kittitrackingroot = '/home/shengjie/Documents/Data/KittiInstance/kittiTracking'
dstroot = '/home/shengjie/Documents/Data/organized_kins/from_tracking'
vlsroot = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/producedInstanceLabel_tracking'
kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
mappings = acquire_mapping(os.path.join(kittitrackingroot, 'training', 'image_02'), kittiroot)

totnum = gettotnum(kittitrackingroot, mappings)
count = 0
st = time.time()
for entry in mappings:
    trackingseq, rawseq = entry
    labelpaths = glob.glob(os.path.join(kittitrackingroot, 'instances', trackingseq, "*.png"))
    for lpath in labelpaths:
        instancelabel = np.array(pil.open(lpath)).astype(np.int)
        rgbpath = os.path.join(kittiroot, rawseq[0:10], rawseq, 'image_02', 'data', "{}.png".format(lpath.split('/')[-1].split('.')[-2].zfill(10)))
        if os.path.exists(rgbpath):
            recomputedinstancelabel = np.zeros_like(instancelabel, dtype=np.uint8)
            localabelcount = 1
            for k in np.unique(instancelabel):
                if k > 0 and k < 10000:
                    recomputedinstancelabel[instancelabel == k] = localabelcount
                    localabelcount = localabelcount + 1
            if recomputedinstancelabel.max() > 0:
                figname = "{}_{}".format(rawseq, lpath.split('/')[-1].split('.')[-2].zfill(10))
                savelabel(recomputedinstancelabel, np.array(pil.open(rgbpath)), figname, vlsroot, dstroot, 'image_02')
        count = count + 1
        dr = time.time() - st
        print("Finished %d images, remain time %f hours" % (count, dr / count * (totnum - count) / 60 / 60))
