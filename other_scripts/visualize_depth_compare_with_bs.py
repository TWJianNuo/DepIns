from __future__ import absolute_import, division, print_function
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from layers import *
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from bnmorph.bnmorph import BNMorph
splits_dir = os.path.join(os.path.dirname(__file__), "splits")
vis_dir = os.path.join(os.path.dirname(__file__), "visualization")
STEREO_SCALE_FACTOR = 5.4

import matlab
import matlab.engine
eng = matlab.engine.start_matlab()
def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))

    # Load Encoder and Decoder
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


    encoder_path = os.path.join('/home/shengjie/Documents/Project_SemanticDepth/tmp/patchmatch_bs/weights_13', "encoder.pth")
    decoder_path = os.path.join('/home/shengjie/Documents/Project_SemanticDepth/tmp/patchmatch_bs/weights_13', "depth.pth")
    encoder_dict = torch.load(encoder_path)

    encoder_bs = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder_bs = networks.DepthDecoder(encoder.num_ch_enc, num_output_channels=3)

    model_dict = encoder.state_dict()
    encoder_bs.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder_bs.load_state_dict(torch.load(decoder_path))

    encoder_bs.cuda()
    encoder_bs.eval()
    depth_decoder_bs.cuda()
    depth_decoder_bs.eval()


    filenames = readlines('/home/shengjie/Documents/Project_SemanticDepth/splits/eigen/test_files.txt')


    opt.frame_ids.append("s")
    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                       encoder_dict['height'], encoder_dict['width'],
                                       [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    count = 0
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                if not (key == 'entry_tag' or key == 'syn_tag'):
                    inputs[key] = ipt.to(torch.device("cuda"))
            input_color = inputs[("color", 0, 0)].cuda()
            outputs = depth_decoder(encoder(input_color))
            outputs_bs = depth_decoder_bs(encoder_bs(input_color))
            for i in range(input_color.shape[0]):
                figbs = tensor2disp(outputs_bs[('disp', 0)][:,2:3,:,:], vmax = 0.1, ind = i)
                fig2 = tensor2disp(outputs[('disp', 0)][:, 2:3, :, :], vmax=0.1, ind=i)
                figrgb = tensor2rgb(inputs[("color", 0, 0)], ind = i)
                combined = np.concatenate([np.array(figrgb), np.array(figbs), np.array(fig2)])
                pil.fromarray(combined).save(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/vls_patchmatch_test_visualization', str(count) + '.png'))
                count = count + 1

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())