from __future__ import absolute_import, division, print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from options import MonodepthOptions
import warnings

import torch.optim as optim
from torch.utils.data import DataLoader
from layers import *

import datasets
import networks

import time
import json

options = MonodepthOptions()
opts = options.parse()


from glob import glob
import cv2
if __name__ == "__main__":

    encoder = networks.ResnetEncoder(18, pretrained = True).cuda()
    decoder = networks.DepthDecoder(encoder.num_ch_enc, opts.scales, num_output_channels=4).cuda()

    encoderpath = os.path.join(opts.load_weights_folder, 'encoder.pth')
    model_dict = encoder.state_dict()
    pretrained_dict = torch.load(encoderpath)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    encoder.load_state_dict(model_dict)

    decoderpath = os.path.join(opts.load_weights_folder, 'depth.pth')
    model_dict = decoder.state_dict()
    pretrained_dict = torch.load(decoderpath)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    decoder.load_state_dict(model_dict)

    decoder = decoder.eval()
    encoder = encoder.eval()


    dayfolds = glob(os.path.join(opts.data_path, '*'))
    imglist = list()
    for dayfold in dayfolds:
        seqfolds = glob(os.path.join(opts.data_path, dayfold, '*/'))
        for seqfold in seqfolds:
            imglist = imglist + glob(os.path.join(seqfold, 'image_02/data', '*.png')) + glob(os.path.join(seqfold, 'image_03/data', '*.png'))


    st = time.time()
    counts = 0
    for imgpath in imglist:
        rgb = pil.open(imgpath)
        rgb = rgb.resize([opts.width, opts.height], pil.BILINEAR)
        rgb = torch.from_numpy(np.array(rgb).astype(np.float32) / 255).permute([2,0,1]).unsqueeze(0).cuda()
        with torch.no_grad():
            outs = decoder(encoder(rgb))
            theta1 = outs['disp', 0][:, 0:1, :, :]
            theta1 = theta1.cpu().numpy()[0,0,:,:]
        theta1 = (theta1 * 256).astype(np.uint16)

        comps = imgpath.split('/')
        target_dir = os.path.join(opts.output_dir, comps[-5], comps[-4], comps[-3])
        os.makedirs(target_dir, exist_ok=True)
        cv2.imwrite(os.path.join(target_dir, comps[-1]), theta1)

        counts = counts + 1
        dr = time.time() - st
        print("Time left %f hours" % ((dr / counts) * (len(imglist) - counts) / 60 / 60))
        # recon = pil.open(os.path.join(target_dir, comps[-1]))
        # recon = np.array(recon).astype(np.float32) / 256
        # recon = torch.from_numpy(recon).unsqueeze(0).unsqueeze(0)
        # tensor2disp(recon, vmax=1, ind=0).show()