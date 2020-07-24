from __future__ import absolute_import, division, print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from torch.utils.data import DataLoader
from layers import *
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

import glob
import torch.optim as optim
import cv2
import numpy as np
import torch


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

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

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
                                           [0], 4, is_train=False, theta_gt_path=opt.theta_gt_path, load_seman=True)
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
        thetalossmap = torch.zeros([1, 1, predh, predw]).expand([opt.batch_size, -1, -1, -1]).cuda()
        thetalossmap[:,:,110::,:] = 1


        import matlab
        import matlab.engine
        eng = matlab.engine.start_matlab()

        localgeomDict = dict()


        bfopt_errs = list()
        afopt_errs = list()
        for count in range(0,len(filenames)):
            # count = 46
            count = 178
            comps = filenames[count].split(' ')
            semidense_path = os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt', comps[0], 'image_02', comps[1].zfill(10)+'.png')
            if (not os.path.isfile(semidense_path)):
                continue
            semidense_depth = pil.open(semidense_path)
            semidense_depth = np.array(semidense_depth).astype(np.float32) / 256

            data = dataset.__getitem__(count)
            input_color = data[("color", 0, 0)].unsqueeze(0).cuda()

            output = depth_decoder(encoder(input_color))

            htheta = data['htheta'].unsqueeze(0).cuda()
            vtheta = data['vtheta'].unsqueeze(0).cuda()
            _, depth = disp_to_depth(output[("disp", 0)][:,2:3,:,:], opt.min_depth, opt.max_depth)
            depth = depth * STEREO_SCALE_FACTOR

            semantics = torch.from_numpy(data['semanLabel']).unsqueeze(0).float()

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
            semantics_gtsize = F.interpolate(semantics, [gtheight, gtwidth], mode='nearest')

            optimizedDepth_torch = localgeomDict[acckey].vls_jointopt_angleConstrain_superpixels(depth_gtsize, htheta_gtsize, vtheta_gtsize, gt_depth, input_color_gtsize, semantics=semantics_gtsize, svname = str(count).zfill(5), eng=eng)
            # optimizedDepth_torch = localgeomDict[acckey].vls_jointopt_angleConstrain_superpixels(depth_gtsize, htheta_gtsize, vtheta_gtsize, semidense_depth, input_color_gtsize, str(count).zfill(5), eng=eng)

            if opt.eval_split == "eigen":
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
                crop = np.array([0.40810811 * gtheight, 0.99189189 * gtheight,
                                 0.03594771 * gtwidth, 0.96405229 * gtwidth]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)
            else:
                mask = gt_depth > 0

            gt_depth = gt_depth[mask]
            depth_bfopt = depth_gtsize[0,0,:,:].detach().cpu().numpy()[mask]
            depth_afopt = optimizedDepth_torch[0,0,:,:].detach().cpu().numpy()[mask]

            depth_bfopt[depth_bfopt < MIN_DEPTH] = MIN_DEPTH
            depth_bfopt[depth_bfopt > MAX_DEPTH] = MAX_DEPTH
            depth_afopt[depth_afopt < MIN_DEPTH] = MIN_DEPTH
            depth_afopt[depth_afopt > MAX_DEPTH] = MAX_DEPTH

            bfopt_errs.append(compute_errors(gt_depth, depth_bfopt))
            afopt_errs.append(compute_errors(gt_depth, depth_afopt))

            print("%d finished." % count)

        print(np.mean(bfopt_errs, 0))
        print(np.mean(afopt_errs, 0))




if __name__ == "__main__":
    options = MonodepthOptions()
    args = options.parse()
    evaluate(args)

