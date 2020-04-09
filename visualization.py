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
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()


    encoder_path = os.path.join('/home/shengjie/Documents/Project_EdgeDepth/EdgeDepth_release/tmp/2fenf_bs_occ_lp15_res50_ft/models/weights_3', "encoder.pth")
    decoder_path = os.path.join('/home/shengjie/Documents/Project_EdgeDepth/EdgeDepth_release/tmp/2fenf_bs_occ_lp15_res50_ft/models/weights_3', "depth.pth")
    # encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    # decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path)

    encoder_bs = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder_bs = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder_bs.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder_bs.load_state_dict(torch.load(decoder_path))

    encoder_bs.cuda()
    encoder_bs.eval()
    depth_decoder_bs.cuda()
    depth_decoder_bs.eval()

    # Init Data loader
    # filenames = readlines(os.path.join(splits_dir, opt.split, "val_files.txt"))
    filenames = readlines('/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full/train_files.txt')
    syn_train_filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", opt.syn_split, "{}_files.txt").format("train"))

    opt.frame_ids.append("s")
    train_dataset = datasets.KITTIRAWDataset(
        opt.data_path, filenames, opt.height, opt.width,
        opt.frame_ids, 4, is_train=False, load_seman=True, load_hints=opt.load_hints,
        hints_path=opt.hints_path,
        load_syn=opt.load_syn, syn_filenames=syn_train_filenames, syn_root=opt.syn_path
    )
    dataloader = DataLoader(
        train_dataset, opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)


    from eppl_render.eppl_render import EpplRender
    epplrender = EpplRender(height=opt.height, width=opt.width, batch_size=opt.batch_size, sampleNum=opt.eppsm).cuda()

    viewIndex = 0
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                if not (key == 'entry_tag' or key == 'syn_tag'):
                    inputs[key] = ipt.to(torch.device("cuda"))

            input_color = inputs[("color", 0, 0)].cuda()
            features = encoder(input_color)
            outputs = dict()
            outputs.update(depth_decoder(features))
            scaledDisp, depth = disp_to_depth(outputs[('disp', 0)], opt.min_depth, opt.max_depth)
            camIndex = [1]
            rendered_real, _, _ = epplrender.forward(depthmap=depth * STEREO_SCALE_FACTOR,
                                                          semanticmap=inputs['semanLabel'],
                                                          intrinsic=inputs['realIn'],
                                                          extrinsic=inputs['realEx'],
                                                          camIndex=camIndex)

            rendered_syn, _, _ = epplrender.forward(depthmap=inputs[('syn_depth', 0)],
                                                         semanticmap=inputs['syn_semanLabel'],
                                                         intrinsic=inputs['realIn'],
                                                         extrinsic=inputs['realEx'],
                                                         camIndex=camIndex)
            # print(torch.mean(rendered_real))
            disparityMap = outputs[('disp', 0)]
            fig_seman = tensor2semantic(inputs['semanLabel'], ind=viewIndex, isGt=True)
            fig_rgb = tensor2rgb(inputs[('color', 0, 0)], ind=viewIndex)
            fig_disp = tensor2disp(disparityMap, ind=viewIndex, vmax=0.1)
            fig_disp_morph = tensor2disp(rendered_real, ind=viewIndex, vmax=0.1)

            fig_seman_syn = tensor2semantic(inputs['syn_semanLabel'], ind=viewIndex, isGt=True)
            fig_disp_morph_syn = tensor2disp(rendered_syn, ind=viewIndex, vmax=0.1)


            features = encoder_bs(input_color)
            outputs = dict()
            outputs.update(depth_decoder_bs(features))
            scaledDisp, depth = disp_to_depth(outputs[('disp', 0)], opt.min_depth, opt.max_depth)
            rendered_real_bs, _, _ = epplrender.forward(depthmap=depth * STEREO_SCALE_FACTOR,
                                                          semanticmap=inputs['semanLabel'],
                                                          intrinsic=inputs['realIn'],
                                                          extrinsic=inputs['realEx'],
                                                          camIndex=camIndex)

            # print(torch.mean(rendered_real_bs))
            disparityMap_bs = outputs[('disp', 0)]
            fig_seman_bs = tensor2semantic(inputs['semanLabel'], ind=viewIndex, isGt=True)
            fig_rgb_bs = tensor2rgb(inputs[('color', 0, 0)], ind=viewIndex)
            fig_disp_bs = tensor2disp(disparityMap_bs, ind=viewIndex, vmax=0.1)
            fig_disp_morph_bs = tensor2disp(rendered_real_bs, ind=viewIndex, vmax=0.1)





            visfd_name = os.path.join(opt.load_weights_folder).split('/')[os.path.join(opt.load_weights_folder).split('/').index("tmp") + 1]
            fdpath = os.path.join(vis_dir, visfd_name)
            os.makedirs(fdpath, exist_ok=True)

            fig_combined = pil.fromarray(np.concatenate(
                [np.array(fig_rgb), np.array(fig_seman), np.array(fig_disp), np.array(fig_disp_morph)], axis=0))
            fig_combined_syn = pil.fromarray(np.concatenate(
                [np.array(fig_seman_syn), np.array(fig_seman_syn), np.array(fig_seman_syn), np.array(fig_disp_morph_syn)], axis=0))
            fig_combined_bs = pil.fromarray(np.concatenate(
                [np.array(fig_rgb_bs), np.array(fig_seman_bs), np.array(fig_disp_bs), np.array(fig_disp_morph_bs)], axis=0))


            fig_sv = pil.fromarray(np.concatenate([np.array(fig_combined), np.array(fig_combined_bs), np.array(fig_combined_syn)], axis=1))
            fig_sv.save(
                os.path.join(fdpath, str(idx) + ".png"))

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())