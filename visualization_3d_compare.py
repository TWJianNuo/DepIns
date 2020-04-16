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


    encoder_path = os.path.join('/home/shengjie/Documents/Project_SemanticDepth/tmp/wSyn0_3/models/weights_54240', "encoder.pth")
    decoder_path = os.path.join('/home/shengjie/Documents/Project_SemanticDepth/tmp/wSyn0_3/models/weights_54240', "depth.pth")
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


    # filenames = readlines('/home/shengjie/Documents/Project_SemanticDepth/splits/eigen_full/train_files.txt')
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
        load_syn=opt.load_syn, PreSIL_root = opt.PreSIL_path
    )
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
            for key, ipt in inputs.items():
                if not (key == 'entry_tag' or key == 'syn_tag'):
                    inputs[key] = ipt.to(torch.device("cuda"))
            #
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
            outputs = depth_decoder(encoder(input_color))
            scaledDisp, depth = disp_to_depth(outputs[('disp', 0)], opt.min_depth, opt.max_depth)

            outputs_bs = depth_decoder_bs(encoder_bs(input_color))
            scaledDisp_bs, depth_bs = disp_to_depth(outputs_bs[('disp', 0)], opt.min_depth, opt.max_depth)

            invcamK = torch.inverse(inputs['realIn'] @ inputs['realEx'])

            pts3d_real = bp3d_kitti(predDepth=depth, invcamK=invcamK)
            pts3d_real_bs = bp3d_kitti(predDepth=depth_bs, invcamK=invcamK)


            for ind in np.unique(instance_semantic_gt[addmask]):
                selector = instance_semantic_gt == ind
                selector_torch = torch.from_numpy(selector).unsqueeze(0).unsqueeze(0).cuda().float()
                selector = (shrinkConv(selector_torch) > shrinkbar).float()

                selector = selector.cpu().numpy()[viewIndex, 0, :, :].flatten() == 1

                drawX_real = pts3d_real[viewIndex, 0, :, :].detach().cpu().numpy().flatten()[selector]
                drawY_real = pts3d_real[viewIndex, 1, :, :].detach().cpu().numpy().flatten()[selector]
                drawZ_real = pts3d_real[viewIndex, 2, :, :].detach().cpu().numpy().flatten()[selector]

                drawX_real_bs = pts3d_real_bs[viewIndex, 0, :, :].detach().cpu().numpy().flatten()[selector]
                drawY_real_bs = pts3d_real_bs[viewIndex, 1, :, :].detach().cpu().numpy().flatten()[selector]
                drawZ_real_bs = pts3d_real_bs[viewIndex, 2, :, :].detach().cpu().numpy().flatten()[selector]

                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(drawX_real, drawY_real, drawZ_real, s=0.7, c='r')
                ax.scatter(drawX_real_bs, drawY_real_bs, drawZ_real_bs, s=0.7, c='g')
                set_axes_equal(ax)
                plt.savefig(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/visualization/pts3d_compare_kitti', str(idx) + '_' + str(ind) + '_3d.png'))
                plt.close()

                fig1 = tensor2disp(selector_torch, ind = 0, vmax = 1)
                fig2 = tensor2rgb(input_color, ind=viewIndex)
                fig3 = tensor2disp(outputs[('disp', 0)], vmax=0.1, ind=viewIndex)
                fig4 = tensor2disp(outputs_bs[('disp', 0)], vmax=0.1, ind=viewIndex)
                img1 = np.concatenate([np.array(fig4), np.array(fig3)], axis=0)
                img2 = np.concatenate([np.array(fig2), np.array(fig1)], axis=0)
                fig = pil.fromarray(np.concatenate([img1, img2], axis=1))
                fig.save(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/visualization/pts3d_compare_kitti', str(idx) + '_' + str(ind) + '_2d.png'))
            # syn_pred = depth_decoder(encoder(inputs['pSIL_rgb']))
            # syn_pred_bs = depth_decoder_bs(encoder_bs(inputs['pSIL_rgb']))
            # pSIL_insMask_shrinked = (shrinkConv(inputs['pSIL_insMask']) > shrinkbar).float()
            #
            #
            # syn_disp = syn_pred[("disp", 0)]
            # syn_disp_bs = syn_pred_bs[("disp", 0)]
            #
            #
            # disp_gt = 0.1 / inputs['pSIL_depth']
            # disp_scalef = torch.sum(syn_disp * pSIL_insMask_shrinked, dim=[1,2,3]) / torch.sum(disp_gt * pSIL_insMask_shrinked, dim=[1,2,3])
            # disp_scalef_ex = disp_scalef.view(opt.batch_size, 1, 1, 1).expand([-1, -1, prsil_ch, prsil_cw])
            # disp_gt_scaled = disp_gt * disp_scalef_ex
            #
            # syn_disp_bs_scalef = torch.sum(syn_disp * pSIL_insMask_shrinked, dim=[1,2,3]) / torch.sum(syn_disp_bs * pSIL_insMask_shrinked, dim=[1,2,3])
            # syn_disp_bs_scalef_ex = syn_disp_bs_scalef.view(opt.batch_size, 1, 1, 1).expand([-1, -1, prsil_ch, prsil_cw])
            # syn_disp_bs_scaled = syn_disp_bs * syn_disp_bs_scalef_ex
            #
            # _, syn_depth = disp_to_depth(syn_disp, opt.min_depth, opt.max_depth)
            # _, syn_depth_bs = disp_to_depth(syn_disp_bs_scaled, opt.min_depth, opt.max_depth)
            # _, syn_depth_gt = disp_to_depth(disp_gt_scaled, opt.min_depth, opt.max_depth)
            # preSILIn = np.eye(4)
            # preSILIn[0, 0] = 512
            # preSILIn[1, 1] = 512
            # preSILIn[0, 3] = 512
            # preSILIn[1, 3] = 288
            # preSILIn = torch.from_numpy(preSILIn).unsqueeze(0).expand([opt.batch_size, -1, -1]).cuda().float()
            # invcamK = torch.inverse(preSILIn @ inputs['realEx'])
            #
            # pts3d_syn = bp3d(predDepth=syn_depth, invcamK=invcamK)
            # pts3d_syn_bs = bp3d(predDepth=syn_depth_bs, invcamK=invcamK)
            # pts3d_syn_gt = bp3d(predDepth=syn_depth_gt, invcamK=invcamK)
            #
            #
            # selector = pSIL_insMask_shrinked.cpu().numpy()[viewIndex, 0].flatten() == 1
            #
            # drawX_syn = pts3d_syn[viewIndex, 0, :, :].detach().cpu().numpy().flatten()[selector]
            # drawY_syn = pts3d_syn[viewIndex, 1, :, :].detach().cpu().numpy().flatten()[selector]
            # drawZ_syn = pts3d_syn[viewIndex, 2, :, :].detach().cpu().numpy().flatten()[selector]
            #
            # drawX_syn_bs = pts3d_syn_bs[viewIndex, 0, :, :].detach().cpu().numpy().flatten()[selector]
            # drawY_syn_bs = pts3d_syn_bs[viewIndex, 1, :, :].detach().cpu().numpy().flatten()[selector]
            # drawZ_syn_bs = pts3d_syn_bs[viewIndex, 2, :, :].detach().cpu().numpy().flatten()[selector]
            #
            # drawX_syn_gt = pts3d_syn_gt[viewIndex, 0, :, :].detach().cpu().numpy().flatten()[selector]
            # drawY_syn_gt = pts3d_syn_gt[viewIndex, 1, :, :].detach().cpu().numpy().flatten()[selector]
            # drawZ_syn_gt = pts3d_syn_gt[viewIndex, 2, :, :].detach().cpu().numpy().flatten()[selector]
            # from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.scatter(drawX_syn, drawY_syn, drawZ_syn, s=0.7, c='r')
            # ax.scatter(drawX_syn_bs, drawY_syn_bs, drawZ_syn_bs, s=0.7, c='g')
            # ax.scatter(drawX_syn_gt, drawY_syn_gt, drawZ_syn_gt, s=0.7, c='b')
            # set_axes_equal(ax)
            # plt.savefig(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/visualization/pts3d_compare_presil', str(idx) + '_1.png'))
            # plt.close()
            # print("finished %d" % idx)
            #
            # fig1 = tensor2disp(syn_pred['disp', 0], vmax=0.1, ind=viewIndex)
            # fig4 = tensor2disp(syn_pred_bs['disp', 0], vmax=0.1, ind=viewIndex)
            # fig2 = tensor2rgb(inputs['pSIL_rgb'], ind=viewIndex)
            # fig3 = tensor2disp(inputs['pSIL_insMask'], vmax=1, ind=viewIndex)
            #
            # img1 = np.concatenate([np.array(fig4), np.array(fig1)], axis=0)
            # img2 = np.concatenate([np.array(fig2), np.array(fig3)], axis=0)
            # pil.fromarray(np.concatenate([img1, img2], axis=1)).save(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/visualization/pts3d_compare_presil', str(idx) + '_2.png'))

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())