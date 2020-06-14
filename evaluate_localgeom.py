from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
from layers import *
import datasets
import networks
import glob
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



def evaluate_sets(ind, target_depth, pred_depth_theta, refdepthi, refdepthi_bs, refdepthi_bs_bs, MIN_DEPTH, MAX_DEPTH):
    ch, gt_height, gt_width = target_depth.shape[1:]
    mask = (target_depth > MIN_DEPTH) * (target_depth < MAX_DEPTH)
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros([1, ch, gt_height, gt_width])
    crop_mask[:,:,crop[0]:crop[1], crop[2]:crop[3]] = 1
    crop_mask = torch.from_numpy(crop_mask).float().cuda()

    totmask = ind * mask.float() * crop_mask
    totmask = totmask == 1
    tosendgt = target_depth[totmask].cpu().numpy()

    toevallist = list()
    relist = list()
    toevallist.append(pred_depth_theta)
    toevallist.append(refdepthi)
    toevallist.append(refdepthi_bs)
    toevallist.append(refdepthi_bs_bs)

    for i in range(len(toevallist)):
        cureval = toevallist[i]
        cureval = torch.clamp(cureval, min=MIN_DEPTH,max=MAX_DEPTH)
        tosend = cureval[totmask].cpu().numpy()
        relist.append(compute_errors(tosendgt,tosend))
    return relist[0], relist[1], relist[2], relist[3]

def reporcess_filename(filenames):
    filenames_processed = list()
    for filename in filenames:
        comps = filename.split()
        filenames_processed.append(comps[0] + '/' + comps[1] + ' ' + comps[2] + ' ' + 'l')
    return filenames_processed

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        filenames = reporcess_filename(filenames)
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
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

        dirmapping = {'l':'image_02', 'r':'image_03'}
        localgeomDict = dict()

        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        metric_theta_hs = list()
        metric_reldepth_hs = list()
        metric_redepthref_hs = list()
        metric_redepthref_bs_hs = list()
        metric_theta_vs = list()
        metric_reldepth_vs = list()
        metric_redepthref_vs = list()
        metric_redepthref_bs_vs = list()

        count = 0
        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))
                _, preddepth = disp_to_depth(output[("disp", 0)][:,2:3,:,:], opt.min_depth, opt.max_depth)
                preddepth = preddepth * STEREO_SCALE_FACTOR
                htheta = output[("disp", 0)][:,0:1,:,:] * 2 * np.pi
                vtheta = output[("disp", 0)][:, 1:2, :, :] * 2 * np.pi

                # for i in range(htheta.shape[0]):
                #     tensor2rgb(input_color, ind=i).show()

                for i in range(htheta.shape[0]):

                    depth_gt_eval = gt_depths[i]
                    dat, frame, dir = filenames[i].split(' ')
                    depth_gt_input = pil.open(os.path.join(opt.kitti_gt_path, dat, dirmapping[dir], frame + '.png'))
                    depth_gt_input = np.array(depth_gt_input).astype(np.float32) / 256
                    depth_gt_input = torch.from_numpy(depth_gt_input).unsqueeze(0).unsqueeze(0).float().cuda()
                    _, _, ch, cw = depth_gt_input.shape
                    acckey = str(ch) + '_' + str(cw)
                    if acckey not in localgeomDict:
                        kittiw = cw
                        kittih = ch
                        intrinsicKitti = np.array([
                            [0.58 * kittiw, 0, 0.5 * kittiw],
                            [0, 1.92 * kittih, 0.5 * kittih],
                            [0, 0, 1]], dtype=np.float32)
                        localthetadesp = LocalThetaDesp(height=kittih, width=kittiw, batch_size=1, intrinsic=intrinsicKitti).cuda()
                        localgeomDict[acckey] = localthetadesp
                    hthetai = htheta[i:i+1,:,:,:]
                    hthetai = F.interpolate(hthetai, [ch, cw], mode='bilinear', align_corners=True)
                    vthetai = vtheta[i:i+1,:,:,:]
                    vthetai = F.interpolate(vthetai, [ch, cw], mode='bilinear', align_corners=True)
                    preddepthi = preddepth[i:i+1,:,:,:]
                    preddepthi = F.interpolate(preddepthi, [ch, cw], mode='bilinear', align_corners=True)


                    rgbi = input_color[i:i+1,:,:,:]
                    rgbi = F.interpolate(rgbi, [ch, cw], mode='bilinear', align_corners=True)
                    localgeomDict[acckey].optimize_depth_using_theta(depthmap=preddepthi, htheta=hthetai, vtheta=vthetai, rgb = rgbi, depthmaplidar = depth_gt_input)

                    indh, target_depth_h, pred_depth_theta_h, refdepthi_h, refdepthi_h_bs, refdepthi_h_bs_bs, indv, target_depth_v, pred_depth_theta_v, refdepthi_v, refdepthi_v_bs, refdepthi_v_bs_bs =localgeomDict[acckey].recover_depth(depthmap=depth_gt_input, htheta = hthetai, vtheta = vthetai, preddepthi = preddepthi)
                    metric_theta_h, metric_reldepth_h, metric_redepthref_h, metric_redepthref_bs_h = evaluate_sets(ind = indh, target_depth = target_depth_h, pred_depth_theta = pred_depth_theta_h, refdepthi = refdepthi_h, refdepthi_bs = refdepthi_h_bs, refdepthi_bs_bs=refdepthi_h_bs_bs, MIN_DEPTH=opt.min_depth, MAX_DEPTH=opt.max_depth)
                    metric_theta_v, metric_reldepth_v, metric_redepthref_v, metric_redepthref_bs_v = evaluate_sets(ind = indv, target_depth = target_depth_v, pred_depth_theta = pred_depth_theta_v, refdepthi = refdepthi_v, refdepthi_bs = refdepthi_v_bs, refdepthi_bs_bs=refdepthi_v_bs_bs, MIN_DEPTH=opt.min_depth, MAX_DEPTH=opt.max_depth)

                    metric_theta_hs.append(metric_theta_h)
                    metric_reldepth_hs.append(metric_reldepth_h)
                    metric_redepthref_hs.append(metric_redepthref_h)
                    metric_redepthref_bs_hs.append(metric_redepthref_bs_h)
                    metric_theta_vs.append(metric_theta_v)
                    metric_reldepth_vs.append(metric_reldepth_v)
                    metric_redepthref_vs.append(metric_redepthref_v)
                    metric_redepthref_bs_vs.append(metric_redepthref_bs_v)
                    # tensor2disp(htheta -1, vmax=4, ind=i).show()
                    # print(metric_theta_h[0])
                    # print(metric_reldepth_h[0])
                    # print(metric_redepthref_h[0])
                    # print(metric_theta_v[0])
                    # print(metric_reldepth_v[0])
                    # print(metric_redepthref_v[0])
                    count = count + 1

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*(np.array(metric_theta_hs).mean(0)).tolist()) + "\\\\")
        print("=======================================")

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*(np.array(metric_reldepth_hs).mean(0)).tolist()) + "\\\\")
        print("=======================================")

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*(np.array(metric_redepthref_hs).mean(0)).tolist()) + "\\\\")
        print("=======================================")

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*(np.array(metric_redepthref_bs_hs).mean(0)).tolist()) + "\\\\")
        print("=======================================")

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*(np.array(metric_theta_vs).mean(0)).tolist()) + "\\\\")
        print("=======================================")

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*(np.array(metric_reldepth_vs).mean(0)).tolist()) + "\\\\")
        print("=======================================")

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*(np.array(metric_redepthref_vs).mean(0)).tolist()) + "\\\\")
        print("=======================================")

        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*(np.array(metric_redepthref_bs_vs).mean(0)).tolist()) + "\\\\")
        print("=======================================")




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

