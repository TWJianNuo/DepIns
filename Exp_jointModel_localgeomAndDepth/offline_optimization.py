from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import *
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import glob
import torch.optim as optim

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "..", "splits")

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

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"
    save_vls_theta = False
    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, theta_gt_path=opt.theta_gt_path)
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, num_output_channels=3)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        predw = encoder_dict['width']
        predh = encoder_dict['height']
        intrinsicKitti = np.array([
            [0.58 * predw, 0, 0.5 * predw],
            [0, 1.92 * predh, 0.5 * predh],
            [0, 0, 1]], dtype=np.float32)
        predsize_localdesp = LocalThetaDesp(height=predh, width=predw, batch_size=1, intrinsic=intrinsicKitti).cuda()
        thetalossmap = torch.zeros([1, 1, predh, predw]).expand([opt.batch_size, -1, -1, -1]).cuda()
        thetalossmap[:,:,110::,:] = 1

        lr = 1e-2
        opt_time = 5000
        scale = 0.1


        preddepths = list()
        preddepths_bs = list()
        count = 0

        diff_bf = 0
        diff_af = 0

        save_vls_theta = True
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()

            output = depth_decoder(encoder(input_color))

            htheta = data['htheta'].cuda()
            vtheta = data['vtheta'].cuda()
            _, pred_depth = disp_to_depth(output[("disp", 0)][:,2:3,:,:], opt.min_depth, opt.max_depth)
            preddepth = pred_depth * STEREO_SCALE_FACTOR
            preddepth_cp = preddepth.clone().detach()

            preddepth = nn.Parameter(preddepth.detach(), requires_grad = True)
            optimizer = optim.Adam([preddepth], lr)
            for k in range(opt_time):
                _, derivx, num_grad = predsize_localdesp.depth_localgeom_consistency(preddepth, htheta, vtheta, isoutput_grads=True)
                loss = torch.abs(derivx - num_grad) + scale * ((preddepth_cp - preddepth) ** 2)
                loss = torch.sum(loss * thetalossmap) / torch.sum(thetalossmap)

                # tensor2disp(torch.abs(num_grad), vmax=0.1, ind=0).show()
                # tensor2disp(torch.abs(derivx), vmax=0.1, ind=0).show()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # tensor2disp(1 / preddepth, percentile=95, ind = 0).show()
            # tensor2disp(1 / preddepth_cp, percentile=95, ind=0).show()


            for i in range(htheta.shape[0]):
                depth_gt_eval = gt_depths[count]
                ch, cw = depth_gt_eval.shape
                depth_gt_eval = torch.from_numpy(depth_gt_eval).float().cuda()
                preddepthi = preddepth[i:i+1,:,:,:]
                preddepthi = F.interpolate(preddepthi, [ch, cw], mode='bilinear', align_corners=True)
                preddepth_cpi = preddepth_cp[i:i+1,:,:,:]
                preddepth_cpi = F.interpolate(preddepth_cpi, [ch, cw], mode='bilinear', align_corners=True)
                diff_af = diff_af + torch.sum(torch.abs(preddepthi - depth_gt_eval) * (depth_gt_eval > 0).float()) / torch.sum((depth_gt_eval > 0).float())
                diff_bf = diff_bf + torch.sum(torch.abs(preddepth_cpi - depth_gt_eval) * (depth_gt_eval > 0).float()) / torch.sum((depth_gt_eval > 0).float())
            print("Before:%f, After:%f, count:%d" % (diff_bf.detach().cpu().numpy() / (count + 1), diff_af.detach().cpu().numpy() / (count + 1), count))


            for i in range(htheta.shape[0]):
                depth_gt_eval = gt_depths[count]
                ch, cw = depth_gt_eval.shape
                preddepthi = preddepth[i:i+1,:,:,:]
                preddepthi = F.interpolate(preddepthi, [ch, cw], mode='bilinear', align_corners=True)
                preddepth_bsi = preddepth_cp[i:i+1,:,:,:]
                preddepth_bsi = F.interpolate(preddepth_bsi, [ch, cw], mode='bilinear', align_corners=True)
                preddepths.append(preddepthi[0,0,:,:].detach().cpu().numpy())
                preddepths_bs.append(preddepth_bsi[0,0,:,:].detach().cpu().numpy())
                count = count + 1

            if save_vls_theta:
                _, derivx, num_grad_old = predsize_localdesp.depth_localgeom_consistency(preddepth_cp, htheta, vtheta, isoutput_grads=True)
                _, derivx, num_grad_new = predsize_localdesp.depth_localgeom_consistency(preddepth, htheta, vtheta, isoutput_grads=True)
                fig_derivx = tensor2disp(torch.abs(derivx), vmax=0.1, ind=0)
                fig_numgrad_old = tensor2disp(torch.abs(num_grad_old), vmax=0.1, ind=0)
                fig_numgrad_new = tensor2disp(torch.abs(num_grad_new), vmax=0.1, ind=0)
                figcombined_left = pil.fromarray(np.concatenate([np.array(fig_derivx), np.array(fig_numgrad_old), np.array(fig_numgrad_new)], axis=0))


                figref_theta = tensor2disp(htheta-1, vmax=4, ind=0)
                new_theta_h, new_theta_v = predsize_localdesp.get_theta(preddepth[0:1,:,:,:])
                old_theta_h, old_theta_v = predsize_localdesp.get_theta(preddepth_cp[0:1, :, :, :])
                fig_new_theta = tensor2disp(new_theta_h - 1, vmax=4, ind=0)
                fig_old_theta = tensor2disp(old_theta_h - 1, vmax=4, ind=0)
                figcombined_center = pil.fromarray(np.concatenate([np.array(figref_theta), np.array(fig_old_theta), np.array(fig_new_theta)], axis=0))

                figrgb = tensor2rgb(input_color, ind=0)
                fig_dispold = tensor2disp(1/preddepth_cp,vmax=0.2,ind=0)
                fig_dispnew = tensor2disp(1 / preddepth, vmax=0.2, ind=0)
                figcombined_right = pil.fromarray(
                    np.concatenate([np.array(figrgb), np.array(fig_dispold), np.array(fig_dispnew)], axis=0))

                figcombined = pil.fromarray(np.concatenate([np.array(figcombined_left), np.array(figcombined_center), np.array(figcombined_right)], axis=1))
                figcombined.save(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/vls_offline_naivel2',str(count) + '.png'))



    print("-> Evaluating")

    errors = []
    errors_bs = []


    for i in range(len(preddepths)):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_depth = preddepths[i]
        # from utils import tensor2disp
        # tensor2disp(1 / torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0), ind = 0, percentile=95).show()
        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]


        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

        pred_depth_bs = preddepths_bs[i]
        pred_depth_bs = pred_depth_bs[mask]

        pred_depth_bs[pred_depth_bs < MIN_DEPTH] = MIN_DEPTH
        pred_depth_bs[pred_depth_bs > MAX_DEPTH] = MAX_DEPTH

        errors_bs.append(compute_errors(gt_depth, pred_depth_bs))

    mean_errors = np.array(errors).mean(0)
    mean_errors_bs = np.array(errors_bs).mean(0)

    print("Before optimization:\n")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.4f}  " * 7).format(*mean_errors_bs.tolist()) + "\\\\")

    print("After optimization:\n")
    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.4f}  " * 7).format(*mean_errors.tolist()) + "\\\\")


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

