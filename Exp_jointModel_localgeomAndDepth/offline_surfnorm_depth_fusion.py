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
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
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
        localGeomDesp = LocalThetaDesp(height=predh, width=predw, batch_size=1, intrinsic=intrinsicKitti).cuda()
        thetalossmap = torch.zeros([1, 1, predh, predw]).expand([opt.batch_size, -1, -1, -1]).cuda()
        thetalossmap[:,:,110::,:] = 1


        invcamK = np.eye(4)
        invcamK[0:3,0:3] = intrinsicKitti
        invcamK = np.linalg.inv(invcamK)
        invcamK = torch.from_numpy(invcamK).float().cuda().unsqueeze(0)
        surfnorm_depth_computer = ComputeSurfaceNormal(height = predh, width = predw, batch_size = opt.batch_size).cuda()

        window_sz = 11
        nsig = 1
        op_gaussblur_kernel = gkern(kernlen=window_sz, nsig=nsig)
        op_gaussblur = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=window_sz, padding=int((window_sz-1)/2))
        op_gaussblur_kernel = nn.Parameter(torch.from_numpy(op_gaussblur_kernel).unsqueeze(0).unsqueeze(0).float(), requires_grad=False)
        op_gaussblur.weight = op_gaussblur_kernel
        op_gaussblur = op_gaussblur.cuda()

        import matlab
        import matlab.engine
        eng = matlab.engine.start_matlab()

        count = 0
        localgeomDict = dict()
        for data in dataloader:
            # if count < 235:
            #     count = count + 1
            #     continue
            input_color = data[("color", 0, 0)].cuda()

            output = depth_decoder(encoder(input_color))

            htheta = data['htheta'].cuda()
            vtheta = data['vtheta'].cuda()
            _, depth = disp_to_depth(output[("disp", 0)][:,2:3,:,:], opt.min_depth, opt.max_depth)
            depth = depth * STEREO_SCALE_FACTOR

            # surfnorm_theta = localGeomDesp.surfnorm_from_localgeom(htheta=htheta, vtheta=vtheta)
            # dir3d = torch.clamp(((surfnorm_theta + 1) / 2), min=0, max=1)
            # dir3d = dir3d.permute([0,3,1,2]).contiguous()
            # fig_surfnorm_theta = tensor2rgb(dir3d, ind=0)
            #
            # surfnorm_depth = surfnorm_depth_computer.forward(depth, invcamK)
            # dir3d = torch.clamp(((surfnorm_depth + 1) / 2), min=0, max=1).contiguous()
            # fig_surfnorm_depth = tensor2rgb(dir3d, ind=0)
            #
            # closs, derivx, num_grad = localGeomDesp.depth_localgeom_consistency(depth, htheta, vtheta, isoutput_grads=True)
            # htheta_d, vtheta_d = localGeomDesp.get_theta(depth)

            gt_depth = gt_depths[count]
            gtheight, gtwidth = gt_depth.shape
            acckey = str(gtheight) + '_' + str(gtwidth)
            if acckey not in localgeomDict:
                gtscale_intrinsic = np.array([
                    [0.58 * gtwidth, 0, 0.5 * gtwidth],
                    [0, 1.92 * gtheight, 0.5 * gtheight],
                    [0, 0, 1]], dtype=np.float32)
                gtsize_localGeomDesp = LocalThetaDesp(height=gtheight, width=gtwidth, batch_size=1, intrinsic=gtscale_intrinsic).cuda()
                localgeomDict[acckey] = gtsize_localGeomDesp
            depth_gtsize = F.interpolate(depth, [gtheight, gtwidth], mode='bilinear', align_corners=True)
            htheta_gtsize = F.interpolate(htheta, [gtheight, gtwidth], mode='bilinear', align_corners=True)
            vtheta_gtsize = F.interpolate(vtheta, [gtheight, gtwidth], mode='bilinear', align_corners=True)
            input_color_gtsize = F.interpolate(input_color, [gtheight, gtwidth], mode='bilinear', align_corners=True)
            localgeomDict[acckey].debias(depth_gtsize, htheta_gtsize, vtheta_gtsize, gt_depth, input_color_gtsize, str(count).zfill(5), eng)

            # tensor2disp(htheta_d - 1, vmax=4, ind=0).show()
            # tensor2disp(htheta - 1, vmax=4, ind=0).show()
            #
            # htheta_bluured = op_gaussblur(htheta)
            # htheta_d_bluured = op_gaussblur(htheta_d)
            # htheta_d_bluured2, _ = localGeomDesp.get_theta(op_gaussblur(depth))
            #
            # tensor2disp(htheta_bluured - 1, vmax=4, ind=0).show()
            # tensor2disp(htheta_d_bluured - 1, vmax=4, ind=0).show()
            # tensor2disp(htheta_d_bluured2 - 1, vmax=4, ind=0).show()

            count = count + 1
            print("%d finished" % count)







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

