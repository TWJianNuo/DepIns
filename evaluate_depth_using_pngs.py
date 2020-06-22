from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import glob
from utils import *
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle = True)["data"]

    # gt_depths = list()
    # missingcount = 0
    # dirmapping = {'l':'image_02', 'r':'image_03'}
    # for i in range(len(filenames)):
    #     seq, frame, dir = filenames[i].split(' ')
    #     seq = seq.split('/')[1]
    #     semidense_gtpath1 = os.path.join('/home/shengjie/Documents/Data/Kitti/kitti_dense_depth/data_depth_annotated', 'train', seq, 'proj_depth', 'groundtruth', dirmapping[dir], frame + '.png')
    #     semidense_gtpath2 = os.path.join('/home/shengjie/Documents/Data/Kitti/kitti_dense_depth/data_depth_annotated', 'val', seq, 'proj_depth', 'groundtruth', dirmapping[dir], frame + '.png')
    #     if os.path.isfile(semidense_gtpath1):
    #         semidense_gtpath = semidense_gtpath1
    #     elif os.path.isfile(semidense_gtpath2):
    #         semidense_gtpath = semidense_gtpath2
    #     else:
    #         semidense_gtpath = None
    #
    #     if semidense_gtpath is not None:
    #         semidepth_depth = np.array(pil.open(semidense_gtpath)).astype(np.float32) / 256
    #         gt_depths.append(semidepth_depth)
    #     else:
    #         gt_depths.append(np.zeros([375, 1242]))
    #         missingcount = missingcount + 1
    # print("Missing: %d" % missingcount)


    errors = []
    ratios = []

    for i in range(len(filenames)):

        gt_depth = gt_depths[i]

        if np.sum(gt_depth) == 0:
            continue

        gt_height, gt_width = gt_depth.shape[:2]

        seq, frame, dir = filenames[i].split(' ')
        seq = seq.split('/')[1]
        pred_depth = pil.open(os.path.join(opt.load_weights_folder, seq + '_' + frame + '.png'))
        pred_depth = np.array(pred_depth).astype(np.float32) / 256
        # tensor2disp(1/ torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0), ind = 0, percentile=95).show()

        # from utils import tensor2disp
        # tensor2disp(1 / torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0), ind = 0, percentile=95).show()
        if opt.eval_split == "eigen":
            if not opt.dokb_crop:
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
            else:
                height = gt_depth.shape[0]
                width = gt_depth.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                gt_depth = gt_depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                if pred_depth.shape[0] != gt_depth.shape[0] or pred_depth.shape[1] != gt_depth.shape[1]:
                    pred_depth = pred_depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    args = options.parse()
    if args.load_weights_folders is not None:
        folders_to_eval = glob.glob(os.path.join(args.load_weights_folders, '*/'))
        to_order = list()
        for i in range(len(folders_to_eval)):
            to_order.append(int(folders_to_eval[i].split('/')[-2].split('_')[1]))
        to_order = np.array(to_order)
        to_order_index = np.argsort(to_order)
        for i in to_order_index:
            print(folders_to_eval[i])
            args.load_weights_folder = folders_to_eval[i]
            evaluate(args)
    else:
        evaluate(args)

