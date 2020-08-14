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

import argparse
import cv2
import time
from torch.utils.data import DataLoader
import datasets
import networks
from utils import *
from layers import disp_to_depth

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
    mapping = {'l': 'image_02', 'r': 'image_03'}

    ts = time.time()
    imgCount = 0

    min_depth = 0.1
    max_depth = 100
    STEREO_SCALE_FACTOR = 5.4

    dataset = datasets.KITTIRAWDataset(opt.data_path, lines, encoder_dict['height'], encoder_dict['width'], [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    with torch.no_grad():
        for data in dataloader:

            outputs = dict()
            input_color = data[("color", 0, 0)].cuda()
            outputs.update(depth_decoder(encoder(input_color)))

            scaledDisp, depth = disp_to_depth(outputs[('disp', 0)], min_depth, max_depth)
            depth = depth * STEREO_SCALE_FACTOR

            for i in range(outputs[('disp', 0)].shape[0]):
                folder, frame_id, direction, _, _ = data['entry_tag'][i].split()
                direction = direction[0]
                frame_id = int(frame_id)

                output_folder = os.path.join(opt.save_dir, folder, mapping[direction])
                os.makedirs(output_folder, exist_ok=True)
                save_path = os.path.join(output_folder, str(frame_id).zfill(10) + '.png')

                depthnp = depth[i,0,:,:].cpu().numpy()
                depthnp = depthnp * 256
                depthnp = depthnp.astype(np.uint16)

                cv2.imwrite(save_path, depthnp)

                # read_depth = pil.open(save_path)
                # read_depth = np.array(read_depth).astype(np.float32) / 256.0
                # np.abs(depth[i,0,:,:].cpu().numpy() - read_depth).max()

                te = time.time()
                imgCount = imgCount + 1
                print("%d finished, %f hours left" % (imgCount, (te - ts) / imgCount * (len(lines) - imgCount) / 60 / 60))
if __name__ == "__main__":
    export_gt_depths_kitti()
