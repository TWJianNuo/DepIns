from __future__ import absolute_import, division, print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--syn_root', type=str)
opt = parser.parse_args()

import glob
import time
import numpy as np
from kitti_utils import *
from PIL.Image import Image
import PIL as pil

def convert_synSemantic(opt):
    syn_root = opt.syn_root

    seqs = ['0001', '0002', '0006', '0018', '0020']

    cat_list = list()
    cat_map_all = dict()
    for seq in seqs:
        cat_map_sub = dict()
        txt_path = os.path.join(syn_root, seq, 'scenegt_rgb_encoding.txt')
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines[1::]:
                # Distill all categories
                cat = line.split(' ')[0].split(':')[0]
                if cat not in cat_list:
                    cat_list.append(cat)

                # Distill sequence specific categories
                name, vals1, vals2, vals3 = line.split(' ')
                catVal = int(vals1) * 255 * 255 + int(vals2) * 255 + int(vals3)
                cat_map_sub[catVal] = name.split(':')[0]
        cat_map_all[seq] = cat_map_sub
    mapping = {
        'Terrain':'terrain',
        'Sky':'sky',
        'Tree':'vegetation',
        'Vegetation':'terrain',
        'Building':'building',
        'Road':'road',
        'GuardRail':'guard rail',
        'TrafficSign':'traffic sign',
        'TrafficLight':'traffic light',
        'Pole':'pole',
        'Misc':'unlabeled',
        'Truck':'truck',
        'Car':'car',
        'Van':'caravan'
    }

    tot_count = 0

    for seq in seqs:
        img_paths = glob.glob(os.path.join(syn_root, seq, 'scenegt', '*.png'))
        cat_map_sub = cat_map_all[seq]
        for img_path in img_paths:
            seman_fig = pil.Image.open(img_path)

            seman_arr = np.array(seman_fig).astype(np.int)
            seman_arr_composed = np.zeros_like(seman_arr, dtype=np.uint8)

            seman_arr_flat = seman_arr[:, :, 0] * 255 * 255 + seman_arr[:, :, 1] * 255 + seman_arr[:, :, 2]
            seman_arr_new = np.zeros_like(seman_arr_flat, dtype=np.int) - 1
            for ind_cat in np.unique(seman_arr_flat):
                seman_arr_new[seman_arr_flat == ind_cat] = name2label[mapping[cat_map_sub[ind_cat]]].id
                seman_arr_composed[seman_arr_flat == ind_cat, :] = name2label[mapping[cat_map_sub[ind_cat]]].color

            assert np.sum(seman_arr_new < 0) == 0, print("Fail to test complete test")

            sv_folder_label = os.path.join(syn_root, seq, 'scene_label')
            sv_folder_compose = os.path.join(syn_root, seq, 'scene_label_composed')
            os.makedirs(sv_folder_label, exist_ok=True)
            os.makedirs(sv_folder_compose, exist_ok=True)

            entry_name = img_path.split('/')[-1]
            label_path = os.path.join(sv_folder_label, entry_name)
            pil.Image.fromarray(seman_arr_new.astype(np.uint8)).save(label_path)

            composed_path = os.path.join(sv_folder_compose, entry_name)
            pil.Image.fromarray(seman_arr_composed.astype(np.uint8)).save(composed_path)

            tot_count = tot_count + 1
            print("Finished %d images" % tot_count)
if __name__ == "__main__":
    convert_synSemantic(opt)
