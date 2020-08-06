from __future__ import absolute_import, division, print_function

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
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
from layers import *
from collections import OrderedDict
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "../splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


class DepthDecoder_earlysplit(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True, split_level=-1):
        super(DepthDecoder_earlysplit, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.split_level = split_level
        self.output_cat = ['theta', 'depth']
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            if i > self.split_level:
                # upconv_0
                num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

                # upconv_1
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            else:
                for cat in self.output_cat:
                    # upconv_0
                    num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
                    num_ch_out = self.num_ch_dec[i]
                    self.convs[("upconv", i, 0, cat)] = ConvBlock(num_ch_in, num_ch_out)

                    # upconv_1
                    num_ch_in = self.num_ch_dec[i]
                    if self.use_skips and i > 0:
                        num_ch_in += self.num_ch_enc[i - 1]
                    num_ch_out = self.num_ch_dec[i]
                    self.convs[("upconv", i, 1, cat)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s, 'theta')] = Conv3x3(self.num_ch_dec[s], 2)
            self.convs[("dispconv", s, 'depth')] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        x_theta = None
        x_depth = None
        for i in range(4, -1, -1):
            if i > self.split_level:
                x = self.convs[("upconv", i, 0)](x)
                x = [upsample(x)]
                if self.use_skips and i > 0:
                    x += [input_features[i - 1]]
                x = torch.cat(x, 1)
                x = self.convs[("upconv", i, 1)](x)
            else:
                if x_theta is None:
                    x_theta = x
                x_theta = self.convs[("upconv", i, 0, 'theta')](x_theta)
                x_theta = [upsample(x_theta)]
                if self.use_skips and i > 0:
                    x_theta += [input_features[i - 1]]
                x_theta = torch.cat(x_theta, 1)
                x_theta = self.convs[("upconv", i, 1, 'theta')](x_theta)

                if x_depth is None:
                    x_depth = x
                x_depth = self.convs[("upconv", i, 0, 'depth')](x_depth)
                x_depth = [upsample(x_depth)]
                if self.use_skips and i > 0:
                    x_depth += [input_features[i - 1]]
                x_depth = torch.cat(x_depth, 1)
                x_depth = self.convs[("upconv", i, 1, 'depth')](x_depth)

            if i in self.scales:
                if i > self.split_level:
                    theta_pred = self.sigmoid(self.convs[("dispconv", i, 'theta')](x))
                    depth_pred = self.sigmoid(self.convs[("dispconv", i, 'depth')](x))
                else:
                    theta_pred = self.sigmoid(self.convs[("dispconv", i, 'theta')](x_theta))
                    depth_pred = self.sigmoid(self.convs[("dispconv", i, 'depth')](x_depth))
                self.outputs[("disp", i)] = torch.cat([theta_pred, depth_pred], dim=1)
        return self.outputs

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
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False)
        depth_decoder = DepthDecoder_earlysplit(encoder.num_ch_enc, split_level = opt.split_level)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))
        count = 0
        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depth_decoder(encoder(input_color))
                if output[("disp", 0)].shape[1] > 1:
                    output[("disp", 0)] = output[("disp", 0)][:,2:3,:,:]

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)
                count = count + 1

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle = True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]

        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

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
