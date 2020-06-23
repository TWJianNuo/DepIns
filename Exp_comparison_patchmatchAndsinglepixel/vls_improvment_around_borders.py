from __future__ import absolute_import, division, print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import PIL.Image as pil
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import networks
from layers import *
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


class DistanceMap():
    def __init__(self, distance_arr, img_width, img_heigth):
        self.distance_arr = np.array(distance_arr)
        self.kernel_size = int(self.distance_arr.max() * 2 + 1)
        self.conv_test = torch.nn.Conv2d(1, len(self.distance_arr), self.kernel_size, stride=1, padding=int((self.kernel_size - 1) / 2), dilation=1, groups=1, bias=False, padding_mode='zeros')
        identify_weights = torch.zeros([len(self.distance_arr), 1, self.kernel_size, self.kernel_size])
        center = (self.kernel_size - 1) / 2
        for i in range(len(self.distance_arr)):
            for m in range(self.kernel_size):
                for n in range(self.kernel_size):
                    if i == 0:
                        low = 0
                    else:
                        low = self.distance_arr[i - 1]
                    high = self.distance_arr[i]
                    if np.abs(m - center) + np.abs(n - center) < high:
                        identify_weights[i, 0, m, n] = 1
        self.conv_test.weight = nn.Parameter(identify_weights,requires_grad=False)
        self.conv_test = self.conv_test.cuda()
        self.rec_err = torch.zeros(len(self.distance_arr) + 1).cuda()
        self.tot_num = torch.zeros(len(self.distance_arr) + 1).cuda()

    def compute_masks(self, mask_in):
        mask_out = self.conv_test(mask_in.float().cuda())
        mask_out[:, 1::, :, :] = (((mask_out[:, 1::, :, :] > 0).float() - (mask_out[:, :-1:, :, :] > 0).float()) > 0).float()
        mask_out = torch.cat([mask_out, 1 - torch.sum(mask_out, dim=1, keepdim=True)], dim=1)
        return mask_out

STEREO_SCALE_FACTOR = 5.4
full_res_shape = (1242, 375)
distarr = np.arange(1, 80, 2).astype(np.int)
distancemap = DistanceMap(distance_arr = distarr, img_heigth = full_res_shape[1], img_width=full_res_shape[0])
get_grad_tool = grad_computation_tools(height=full_res_shape[1], width=full_res_shape[0], batch_size=1).cuda()

encoder_path = '/home/shengjie/Documents/Project_SemanticDepth/tmp/trainOnKitti2/models/weights_11/encoder.pth'
decoder_path = '/home/shengjie/Documents/Project_SemanticDepth/tmp/trainOnKitti2/models/weights_11/depth.pth'
encoder_dict = torch.load(encoder_path)

encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, num_output_channels=3)

model_dict = encoder.state_dict()
encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
depth_decoder.load_state_dict(torch.load(decoder_path))

encoder.cuda()
encoder.eval()
depth_decoder.cuda()
depth_decoder.eval()



raw_data_root = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
depth_gt_root = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'
depth_pred_root = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/vls_offline_patchpixelCompare/depthmap'
patch_img_paths = glob(os.path.join(depth_pred_root, 'patch', '*.png'))
numbins = len(distarr) + 1
num_for_eachbin = np.zeros(numbins)
absrel_patch = np.zeros(numbins)
absrel_pixel = np.zeros(numbins)
count=0
for patch_img_path in patch_img_paths:
    rgbpath = os.path.join(raw_data_root, patch_img_path.split('/')[-1][0:10], patch_img_path.split('/')[-1][0:26], patch_img_path.split('/')[-1][38:46], 'data', patch_img_path.split('/')[-1].split('_')[6] + '.png')
    rgb = pil.open(rgbpath).resize([1024, 320], pil.BILINEAR)
    rgb = (torch.from_numpy(np.array(rgb).astype(np.float32)) / 255).permute([2,0,1]).unsqueeze(0).cuda()
    outputs = depth_decoder(encoder(rgb))
    # _, depth = disp_to_depth(outputs['disp', 0][:,2:3,:,:], min_depth=0.1, max_depth=100)
    # depth = depth * STEREO_SCALE_FACTOR
    # depth = F.interpolate(depth, [full_res_shape[1], full_res_shape[0]], mode='bilinear', align_corners=True)
    # depth_grad = get_grad_tool.get_gradient(depth)
    # depth_grad_normalized = depth_grad / depth
    # edges = depth_grad_normalized > 0.1

    disp = outputs['disp', 0][:,2:3,:,:]
    disp = F.interpolate(disp, [full_res_shape[1], full_res_shape[0]], mode='bilinear', align_corners=True)
    disp_grad = get_grad_tool.get_gradient(disp)
    edges = disp_grad > 0.02

    # from utils import *
    # tensor2disp(disp,ind=0,percentile=95).show()
    # tensor2disp(disp_grad,ind=0,percentile=95).show()
    # tensor2disp(edges,ind=0,vmax=1).show()

    edgemasks = distancemap.compute_masks(edges)
    # tensor2disp(edgemasks[:,10:11,:,:],vmax=1,ind=0).show()

    pixel_img_path = os.path.join(depth_pred_root, 'pixel', patch_img_path.split('/')[-1])
    patch_depth = np.array(pil.open(patch_img_path)).astype(np.float32) / 256
    pixel_depth = np.array(pil.open(pixel_img_path)).astype(np.float32) / 256

    gt_depth_path = os.path.join(depth_gt_root, patch_img_path.split('/')[-1][0:10], patch_img_path.split('/')[-1][0:26], patch_img_path.split('/')[-1][38:46], patch_img_path.split('/')[-1].split('_')[6] + '.png')
    gt_depth_path = pil.open(gt_depth_path).resize(full_res_shape, pil.NEAREST)
    gt_depth_path = np.array(gt_depth_path).astype(np.float32) / 256

    selector = gt_depth_path > 1e-3

    for i in range(numbins):
        curselector = selector * (edgemasks[0,i,:,:].cpu().numpy() == 1)
        # tensor2disp(torch.from_numpy(curselector).unsqueeze(0).unsqueeze(0).float(), ind=0, vmax=1).show()
        curdist_scan_num = np.sum(curselector)
        if curdist_scan_num > 0:
            num_for_eachbin[i] = num_for_eachbin[i] + curdist_scan_num

            gt_depth_path_selected = gt_depth_path[curselector]
            patch_depth_selected = patch_depth[curselector]
            pixel_depth_selected = pixel_depth[curselector]

            patch_depth_absrel = np.abs(gt_depth_path_selected - patch_depth_selected) / gt_depth_path_selected
            pixel_depth_absrel = np.abs(gt_depth_path_selected - pixel_depth_selected) / gt_depth_path_selected

            absrel_patch[i] = absrel_patch[i] + np.sum(patch_depth_absrel)
            absrel_pixel[i] = absrel_pixel[i] + np.sum(pixel_depth_absrel)

    print("%d finished" % count)
    count = count + 1
absrel_patch_mean = absrel_patch / (num_for_eachbin + 1)
absrel_pixel_mean = absrel_pixel / (num_for_eachbin + 1)

plt.figure(figsize=(24, 18), dpi=80)
markerline1, stemlines, _ = plt.stem(distarr.tolist()+[81], absrel_patch_mean, use_line_collection = True)
plt.setp(markerline1, 'markerfacecolor', 'b')
markerline2, stemlines, _ = plt.stem(distarr.tolist()+[81], absrel_pixel_mean, use_line_collection = True)
plt.setp(markerline2, 'markerfacecolor', 'r')
plt.legend(['patch', 'pixel'])
plt.savefig(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/vls_offline_patchpixelCompare', 'compare_edgedistance'))







