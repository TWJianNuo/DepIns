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

def make_json_dict(imgs, anns):
    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict


def savelabel(producedinstancelabel, rgb, figname, vlsroot_rgb, outputroot, cam):
    date = figname[0:10]
    seq = "{}_sync".format(figname[0:21])
    ind = figname.split('_')[-1]
    os.makedirs(os.path.join(outputroot, date, seq, cam), exist_ok=True)
    os.makedirs(os.path.join(vlsroot_rgb, date, seq, cam), exist_ok=True)
    pil.fromarray(producedinstancelabel).save(os.path.join(outputroot, date, seq, cam, "{}.png".format(ind)))

    coloredimg = np.zeros_like(rgb)
    ratio = 0.3
    for k in np.unique(producedinstancelabel):
        if k > 0:
            selector = producedinstancelabel == k
            coloredimg[selector, :] = np.repeat((np.random.random([1, 3]) * 255).astype(np.uint8), np.sum(selector), axis=0)
    combined = rgb.astype(np.float) * (1-ratio) + coloredimg.astype(np.float)*ratio
    combined = combined.astype(np.uint8)
    pil.fromarray(combined).save(os.path.join(vlsroot_rgb, date, seq, cam, "{}.png".format(ind)))

if __name__ == '__main__':
    src_img_path = "/home/shengjie/Documents/Data/KittiInstance/kins/training/image_2"
    src_gt7_path = "/home/shengjie/Documents/Data/KittiInstance/kins/update_train_2020.json"
    kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
    mappingfilepath = '/home/shengjie/Documents/Data/KittiInstance/kins/train_mapping.txt'
    perbfilepath = '/home/shengjie/Documents/Data/KittiInstance/kins/train_rand.txt'
    dstroot = '/home/shengjie/Documents/Data/organized_kins/from_kins'
    vlsroot = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/producedInstanceLabel_kins'
    mappings = readlines(mappingfilepath)
    randind = readlines(perbfilepath)
    randind = randind[0].split(',')

    anns = cvb.load(src_gt7_path)
    imgs_info = anns['images']
    anns_info = anns["annotations"]

    imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)
    count = 0
    totnum = len(anns_dict.keys())

    st = time.time()

    for img_id in anns_dict.keys():
        indnum = int(imgs_dict[img_id].split('.')[0])
        mappedind = int(randind[indnum]) - 1

        entry = mappings[mappedind]
        date, seq, index = entry.split(' ')
        img_path = os.path.join(kittiroot, date, seq, 'image_02', 'data', "{}.png".format(index.zfill(10)))

        if os.path.exists(img_path):
            img = pil.open(img_path)

            height, width, _ = np.array(img).shape

            inslabel = np.zeros([height, width], dtype=np.uint8)

            locallabelcount = 1
            anns = anns_dict[img_id]
            for ann in anns:
                imodal_rle = maskUtils.frPyObjects(ann['i_segm'], height, width)
                inmodal_ann_mask = maskUtils.decode(imodal_rle)
                inmodal_ann_mask = inmodal_ann_mask[:, :, 0] == 1
                inslabel[inmodal_ann_mask] = locallabelcount
                locallabelcount = locallabelcount + 1

            savelabel(inslabel, np.array(img), "{}_{}".format(seq, index), vlsroot, dstroot, 'image_02')

        count = count + 1
        dr = time.time() - st
        leftime = (totnum - count) * (dr / count) / 60 / 60
        print("Finished: %d, remains hours: %f" % (count, leftime))