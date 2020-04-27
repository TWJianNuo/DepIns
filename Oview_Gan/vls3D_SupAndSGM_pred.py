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

    filenames = readlines('/home/shengjie/Documents/Project_SemanticDepth/splits/kitti_seman_mapped2depth//train_files.txt')

    mapping = readlines(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/splits', 'training_mapping.txt'))
    mapping_ind = list()
    for idx, m in enumerate(mapping):
        if len(m) > 1:
            mapping_ind.append(idx)


    opt.frame_ids.append("s")
    train_dataset = datasets.KITTIRAWDataset(
        opt.data_path, filenames, opt.height, opt.width,
        opt.frame_ids, 4, is_train=False, load_seman=True, load_hints=opt.load_hints,
        hints_path=opt.hints_path,
        load_syn=opt.load_syn, PreSIL_root = opt.PreSIL_path)
    dataloader = DataLoader(
        train_dataset, opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    weights = torch.tensor([[1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]])
    weights = weights.view(1, 1, 3, 3)
    shrinkbar = 8
    shrinkConv = nn.Conv2d(1, 1, 3, bias=False, padding=1)
    shrinkConv.weight = nn.Parameter(weights, requires_grad=False)
    shrinkConv = shrinkConv.cuda()

    viewIndex = 0
    prsil_cw = 32 * 10
    prsil_ch = 32 * 8

    bp3d = BackProj3D(height=prsil_ch, width=prsil_cw, batch_size=opt.batch_size).cuda()
    bp3d_kitti = BackProj3D(height=opt.height, width=opt.width, batch_size=opt.batch_size).cuda()
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            # inputs = train_dataset.__getitem__(32)
            # idx = 32
            for key, ipt in inputs.items():
                if not (key == 'entry_tag' or key == 'syn_tag'):
                    inputs[key] = ipt.to(torch.device("cuda"))

            instance_semantic_gt = pil.open(os.path.join('/home/shengjie/Documents/Data/Kitti/kitti_semantics/training/instance', str(mapping_ind[idx]).zfill(6) + '_10.png'))
            instance_semantic_gt = instance_semantic_gt.resize([opt.width, opt.height], pil.NEAREST)
            instance_semantic_gt = np.array(instance_semantic_gt).astype(np.uint16)
            semantic_gt = instance_semantic_gt // 256

            # semantic_selector = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
            semantic_selector = [26]
            addmask = np.zeros_like(semantic_gt)
            for vt in semantic_selector:
                addmask = addmask + (semantic_gt == vt)
            addmask = addmask > 0

            input_color = inputs[("color", 0, 0)].cuda()

            invcamK = torch.inverse(inputs['realIn'] @ inputs['realEx'])

            # depth = inputs['depth_hint']

            depth_path = os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Bts_Pred/result_bts_semantic/raw', filenames[idx].split(' ')[0].split('/')[1] + '_' + filenames[idx].split(' ')[1].zfill(10) + '.png')
            # depth_path = os.path.join('/home/shengjie/Documents/Data/Kitti/kitti_gtDepthMap/', filenames[idx].split(' ')[0], 'image_02', filenames[idx].split(' ')[1].zfill(10) + '.png')
            depth = pil.open(depth_path)
            depth = depth.resize([opt.width, opt.height], pil.NEAREST)
            depth = np.array(depth).astype(np.uint16).astype(np.float32)
            depth = depth / 256
            depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float().cuda()
            pts3d_real = bp3d_kitti(predDepth=depth, invcamK=invcamK)


            depth_path_gt = os.path.join('/home/shengjie/Documents/Data/Kitti/kitti_gtDepthMap/', filenames[idx].split(' ')[0], 'image_02', filenames[idx].split(' ')[1].zfill(10) + '.png')
            depth_gt = pil.open(depth_path_gt)
            depth_gt = depth_gt.resize([opt.width, opt.height], pil.NEAREST)
            depth_gt = np.array(depth_gt).astype(np.uint16).astype(np.float32)
            depth_gt = depth_gt / 256
            depth_gt = torch.from_numpy(depth_gt).unsqueeze(0).unsqueeze(0).float().cuda()
            pts3d_real_gt = bp3d_kitti(predDepth=depth_gt, invcamK=invcamK)
            for ind in np.unique(instance_semantic_gt[addmask]):
                selector = instance_semantic_gt == ind
                selector_torch = torch.from_numpy(selector).unsqueeze(0).unsqueeze(0).cuda().float()
                selector = (shrinkConv(selector_torch) > shrinkbar).float()
                selector = selector * (depth > 0).float()
                # fig1 = tensor2disp(selector, ind=0, vmax=1)
                # fig2 = tensor2disp(depth, vmax = 100, ind = 0)
                # img2 = np.concatenate([np.array(fig2), np.array(fig1)], axis=0)
                # fig = pil.fromarray(img2)
                # fig.show()
                if torch.sum(selector) < 500:
                    continue

                selector = selector.cpu().numpy()[viewIndex, 0, :, :].flatten() == 1

                drawX_real = pts3d_real[viewIndex, 0, :, :].detach().cpu().numpy().flatten()[selector]
                drawY_real = pts3d_real[viewIndex, 1, :, :].detach().cpu().numpy().flatten()[selector]
                drawZ_real = pts3d_real[viewIndex, 2, :, :].detach().cpu().numpy().flatten()[selector]

                drawX_real = matlab.double(drawX_real.tolist())
                drawY_real = matlab.double(drawY_real.tolist())
                drawZ_real = matlab.double(drawZ_real.tolist())


                selector_gt = instance_semantic_gt == ind
                selector_torch_gt = torch.from_numpy(selector_gt).unsqueeze(0).unsqueeze(0).cuda().float()
                selector_gt = (shrinkConv(selector_torch_gt) > shrinkbar).float()
                selector_gt = selector_gt * (depth > 0).float()
                selector_gt = selector_gt.cpu().numpy()[viewIndex, 0, :, :].flatten() == 1

                drawX_real_gt = pts3d_real_gt[viewIndex, 0, :, :].detach().cpu().numpy().flatten()[selector_gt]
                drawY_real_gt = pts3d_real_gt[viewIndex, 1, :, :].detach().cpu().numpy().flatten()[selector_gt]
                drawZ_real_gt = pts3d_real_gt[viewIndex, 2, :, :].detach().cpu().numpy().flatten()[selector_gt]

                drawX_real_gt = matlab.double(drawX_real_gt.tolist())
                drawY_real_gt = matlab.double(drawY_real_gt.tolist())
                drawZ_real_gt = matlab.double(drawZ_real_gt.tolist())

                eng.eval('figure()', nargout=0)
                # h = eng.scatter3(drawX_syn, drawY_syn, drawZ_syn, 5, 'filled', 'r', nargout = 0)
                eng.eval('hold on', nargout=0)
                h = eng.scatter3(drawX_real, drawY_real, drawZ_real, 5, 'filled', 'g', nargout=0)
                eng.eval('hold on', nargout=0)
                h = eng.scatter3(drawX_real_gt, drawY_real_gt, drawZ_real_gt, 5, 'filled', 'r', nargout=0)
                eng.eval('xlim([10,20])', nargout = 0)
                eng.eval('axis equal', nargout = 0)
                eng.eval('grid off', nargout=0)
                eng.eval('close all', nargout=0)

                fig1 = tensor2disp(selector_torch, ind = 0, vmax = 1)
                fig2 = tensor2rgb(input_color, ind=viewIndex)
                fig3 = tensor2disp(depth, vmax=80, ind=viewIndex)
                img2 = np.concatenate([np.array(fig2), np.array(fig1), np.array(fig3)], axis=0)
                fig = pil.fromarray(img2)
                # fig.save(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/visualization/pts3d_compare_kitti', str(idx) + '_' + str(ind) + '_2d.png'))

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())