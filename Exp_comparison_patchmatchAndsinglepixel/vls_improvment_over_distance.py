import os
import PIL.Image as pil
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
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

full_res_shape = (1242, 375)
depth_gt_root = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'
depth_pred_root = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/vls_offline_patchpixelCompare/depthmap'
patch_img_paths = glob(os.path.join(depth_pred_root, 'patch', '*.png'))
bins = np.linspace(0, 100, 100)
num_for_eachbin = np.zeros(len(bins))
absrel_patch = np.zeros(len(bins))
absrel_pixel = np.zeros(len(bins))
for patch_img_path in patch_img_paths:
    pixel_img_path = os.path.join(depth_pred_root, 'pixel', patch_img_path.split('/')[-1])
    patch_depth = np.array(pil.open(patch_img_path)).astype(np.float32) / 256
    pixel_depth = np.array(pil.open(pixel_img_path)).astype(np.float32) / 256

    gt_depth_path = os.path.join(depth_gt_root, patch_img_path.split('/')[-1][0:10], patch_img_path.split('/')[-1][0:26], patch_img_path.split('/')[-1][38:46], patch_img_path.split('/')[-1].split('_')[6] + '.png')
    gt_depth_path = pil.open(gt_depth_path).resize(full_res_shape, pil.NEAREST)
    gt_depth_path = np.array(gt_depth_path).astype(np.float32) / 256

    selector = gt_depth_path > 1e-3

    gt_depth_path_selected = gt_depth_path[selector]
    patch_depth_selected = patch_depth[selector]
    pixel_depth_selected = pixel_depth[selector]

    patch_depth_absrel = np.abs(gt_depth_path_selected - patch_depth_selected) / gt_depth_path_selected
    pixel_depth_absrel = np.abs(gt_depth_path_selected - pixel_depth_selected) / gt_depth_path_selected

    indices = np.digitize(gt_depth_path_selected, bins)

    for i in range(len(bins)):
        num_for_eachbin[i] = num_for_eachbin[i] + np.sum(indices == i)
        absrel_pixel[i] = absrel_pixel[i] + np.sum(pixel_depth_absrel[indices == i])
        absrel_patch[i] = absrel_patch[i] + np.sum(patch_depth_absrel[indices == i])

absrel_patch_mean = absrel_patch / (num_for_eachbin + 1)
absrel_pixel_mean = absrel_pixel / (num_for_eachbin + 1)

plt.figure(figsize=(24, 18), dpi=80)
markerline1, stemlines, _ = plt.stem(bins, absrel_patch_mean, use_line_collection = True)
plt.setp(markerline1, 'markerfacecolor', 'b')
markerline2, stemlines, _ = plt.stem(bins, absrel_pixel_mean, use_line_collection = True)
plt.setp(markerline2, 'markerfacecolor', 'r')
# plt.plot(bins, absrel_patch_mean)
# plt.plot(bins, absrel_pixel_mean)
plt.legend(['patch', 'pixel'])
plt.savefig(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/vls_offline_patchpixelCompare', 'compare_distance_wise'))







