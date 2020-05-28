# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import os
import argparse
import numpy as np
import PIL.Image as pil
from utils import readlines
from kitti_utils import generate_depth_map
import cv2
import time
import torch
from torch.utils.data import DataLoader
import datasets
import networks
from utils import *

def get_entry_from_path(imgpath):
    comps = imgpath.split('/')
    if comps[-3] == 'image_02':
        direct = 'l'
    else:
        direct = 'r'
    entry = comps[-5] + '/' + comps[-4] + ' ' + comps[-1].split('.')[0] + ' ' + direct + '\n'
    return entry

def collect_all_entries(folder):
    import glob
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    entries = list()
    for seqs_clusts in subfolders:
        seqs = [f.path for f in os.scandir(seqs_clusts) if f.is_dir()]
        for seq in seqs:
            imgFolder_02 = os.path.join(seq, 'image_02', 'data')
            imgFolder_03 = os.path.join(seq, 'image_03', 'data')
            for imgpath in glob.glob(imgFolder_02 + '/*.png'):
                entries.append(get_entry_from_path(imgpath))
            for imgpath in glob.glob(imgFolder_03 + '/*.png'):
                entries.append(get_entry_from_path(imgpath))
    return entries



def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_pred_theta')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--save_dir',
                        type=str,
                        help='path to the root of save folder',
                        required=True)
    parser.add_argument('--load_weights_folder',
                        type=str,
                        help='path to the root of save folder',
                        required=True)
    parser.add_argument('--num_layers',
                        type=int,
                        default=18)
    parser.add_argument('--num_workers',
                        type=int,
                        default=16)
    parser.add_argument('--banvls',
                        action='store_true')


    opt = parser.parse_args()
    os.makedirs(opt.save_dir, exist_ok=True)
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path)
    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, num_output_channels=3)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()


    lines = collect_all_entries(opt.data_path)
    lines_valid = list()
    for line in lines:
        folder, frame_id, direction = line.split()
        frame_id = int(frame_id)
        velo_filename = os.path.join(opt.data_path, folder, "velodyne_points/data", "{:010d}.bin".format(frame_id))
        if os.path.isfile(velo_filename):
            lines_valid.append(line)

    mapping = {'l': 'image_02', 'r': 'image_03'}
    mapping_cam = {'l': 2, 'r': 3}

    ts = time.time()
    imgCount = 0

    dataset = datasets.KITTIRAWDataset(opt.data_path, lines_valid, encoder_dict['height'], encoder_dict['width'], [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    with torch.no_grad():
        for data in dataloader:
            outputs = dict()
            input_color = data[("color", 0, 0)].cuda()
            outputs.update(depth_decoder(encoder(input_color)))

            for i in range(outputs[('disp', 0)].shape[0]):
                folder, frame_id, direction, _ = data['entry_tag'][i].split()
                frame_id = int(frame_id)

                output_folder_h = os.path.join(opt.save_dir, folder, 'htheta', mapping[direction])
                output_folder_v = os.path.join(opt.save_dir, folder, 'vtheta', mapping[direction])
                os.makedirs(output_folder_h, exist_ok=True)
                os.makedirs(output_folder_v, exist_ok=True)
                save_path_h = os.path.join(output_folder_h, str(frame_id).zfill(10) + '.png')
                save_path_v = os.path.join(output_folder_v, str(frame_id).zfill(10) + '.png')

                thetah = outputs[('disp', 0)][i:i+1,0:1,:,:] * 2 * np.pi
                thetav = outputs[('disp', 0)][i:i + 1, 1:2, :, :] * 2 * np.pi

                thetahnp = thetah.squeeze(0).squeeze(0).cpu().numpy()
                thetavnp = thetav.squeeze(0).squeeze(0).cpu().numpy()

                thetahnp_towrite = (thetahnp * 10 * 256).astype(np.uint16)
                thetavnp_towrite = (thetavnp * 10 * 256).astype(np.uint16)
                cv2.imwrite(save_path_h, thetahnp_towrite)
                cv2.imwrite(save_path_v, thetavnp_towrite)

                # reopen_h = np.array(pil.open(save_path_h)).astype(np.float32) / 10 / 256
                # reopen_v = np.array(pil.open(save_path_v)).astype(np.float32) / 10 / 256
                # print(np.abs(reopen_h - thetahnp).max())
                # print(np.abs(reopen_v - thetavnp).max())

                if not opt.banvls:
                    output_folder_hvls = os.path.join(opt.save_dir, folder, 'htheta_vls', mapping[direction])
                    output_folder_vvls = os.path.join(opt.save_dir, folder, 'vtheta_vls', mapping[direction])
                    os.makedirs(output_folder_hvls, exist_ok=True)
                    os.makedirs(output_folder_vvls, exist_ok=True)
                    figh = tensor2disp(thetah - 1, vmax=4, ind=0)
                    figv = tensor2disp(thetav - 1, vmax=4, ind=0)
                    save_path_hvls = os.path.join(output_folder_hvls, str(frame_id).zfill(10) + '.png')
                    save_path_vvls = os.path.join(output_folder_vvls, str(frame_id).zfill(10) + '.png')
                    figh.save(save_path_hvls)
                    figv.save(save_path_vvls)

                te = time.time()
                imgCount = imgCount + 1
                print("%d finished, %f hours left" % (imgCount, (te - ts) / imgCount * (len(lines) - imgCount) / 60 / 60))
if __name__ == "__main__":
    export_gt_depths_kitti()
