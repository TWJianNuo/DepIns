import cv2
import numpy as np
from matplotlib import cm
from PIL import Image
import os
import math
import torch
from utils import *

# Constants
rows = 1080 #image height
cols = 1920 #image width

dataset_dir = "/home/shengjie/Documents/Data/PreSIL"
depth_dir = os.path.join(dataset_dir, 'depth')
stencil_dir = os.path.join(dataset_dir, 'stencil')

# # Predefined Kitti Intrinsic Matrix, (fx = 594.89, fy = 615.71)
# # Predefined PreSIL Intrinsic Matrix, (fx = 960, fy = 960)
cut = 4
w = 1024
h = 576
def ndcToDepth(ndc):
    nc_z = 0.15
    fc_z = 600
    fov_v = 59 #degrees
    nc_h = 2 * nc_z * math.tan(fov_v / 2.0)
    nc_w = 1920 / 1080.0 * nc_h

    depth = np.zeros((rows,cols))

    # Iterate through values
    # d_nc could be saved as it is identical for each computation
    # Then the rest of the calculations could be vectorized
    # TODO if need to use this frequently
    for j in range(0,rows):
        for i in range(0,cols):
            nc_x = abs(((2 * i) / (cols - 1.0)) - 1) * nc_w / 2.0
            nc_y = abs(((2 * j) / (rows - 1.0)) - 1) * nc_h / 2.0

            d_nc = math.sqrt(pow(nc_x,2) + pow(nc_y,2) + pow(nc_z,2))
            depth[j,i] = d_nc / (ndc[j,i] + (nc_z * d_nc / (2 * fc_z)))
            if ndc[j,i] == 0.0:
                depth[j,i] = fc_z
    return depth

def batch_ndcToDepth(ndc):
    nc_z = 0.15
    fc_z = 600
    fov_v = 59 #degrees
    nc_h = 2 * nc_z * math.tan(fov_v / 2.0)
    nc_w = 1920 / 1080.0 * nc_h

    xv, yv = np.meshgrid(range(rows), range(cols), sparse=False, indexing='ij')
    nc_xc = abs(((2 * yv) / (cols - 1.0)) - 1) * nc_w / 2.0
    nc_yc = abs(((2 * xv) / (rows - 1.0)) - 1) * nc_h / 2.0
    d_ncc = np.sqrt(nc_xc * nc_xc + nc_yc * nc_yc + nc_z * nc_z)
    depthc = d_ncc / (ndc + (nc_z * d_ncc / (2 * fc_z)))
    depthc[ndc == 0.0] = fc_z
    return depthc

def cvt_depth2png_PreSIL(depthIm):
    maxM = 650
    sMax = 255 ** 3 - 1

    depthIm = np.clip(depthIm, a_min=0.1, a_max=maxM)

    depthIm_f = depthIm * (sMax / maxM)
    r = np.floor(depthIm_f / 255 / 255)
    g = np.floor((depthIm_f - r * 255 * 255) / 255)
    b = np.floor((depthIm_f - r * 255 * 255 - g * 255))

    tsv_depth = np.stack([r, g, b], axis=2).astype(np.uint8)
    return tsv_depth, depthIm

def cvt_png2depth_PreSIL(tsv_depth):
    maxM = 650
    sMax = 255 ** 3 - 1

    tsv_depth = tsv_depth.astype(np.float)
    depthIm = (tsv_depth[:,:,0] * 255 * 255 + tsv_depth[:,:,1] * 255 + tsv_depth[:,:,2]) / sMax * maxM
    return depthIm

def resize_arr_depth(depthIm):
    depthIm_torch = torch.from_numpy(depthIm).unsqueeze(0).unsqueeze(0)
    depthIm_torch = F.interpolate(depthIm_torch, [h, w], mode = 'bilinear', align_corners = True)
    depthIm_resized = depthIm_torch[0,0].numpy()
    depthIm_resized = depthIm_resized[32 * cut::, :]
    return depthIm_resized

def resize_arr_rgb(rgb):
    rgb_r = pil.Image.resize(rgb, [w, h],  resample = pil.LANCZOS)
    rgb_r = np.array(rgb_r)[32 * cut::, :, :]
    return rgb_r

def resize_arr_ins(ins_label):
    ins_label_r = pil.Image.resize(ins_label, [w, h],  resample = pil.NEAREST)
    ins_label_r = np.array(ins_label_r)[32 * cut::, :, :]
    return ins_label_r



maxM = 650
sMax = 255 ** 3 - 1

import time
st = time.time()
target_dir = '/home/shengjie/Documents/Data/PreSIL_organized'
for img_idx in range(0, 51050):
    # img_idx = 27
    file_path = depth_dir + '/{:06d}.bin'.format(img_idx)
    fd = open(file_path, 'rb')
    f = np.fromfile(fd, dtype=np.float32,count=rows*cols)
    im = f.reshape((rows, cols))

    depthIm = batch_ndcToDepth(im)
    depthIm = resize_arr_depth(depthIm)
    tsv_depth, depthIm_clipped = cvt_depth2png_PreSIL(depthIm)


    # depthIm_recon = cvt_png2depth_PreSIL(tsv_depth)
    # tensor2disp(torch.from_numpy(im).unsqueeze(0).unsqueeze(0), percentile=95, ind=0).show()
    # tensor2disp(torch.from_numpy(1 / im).unsqueeze(0).unsqueeze(0), percentile=90, ind=0).show()
    # tensor2disp(torch.from_numpy(depthIm).unsqueeze(0).unsqueeze(0), percentile=95, ind=0).save("/home/shengjie/Downloads/depth_alike.png")
    rgb = pil.open(os.path.join(dataset_dir, 'image_2', "{:06d}.png".format(img_idx)))
    rgb_r = resize_arr_rgb(rgb)

    ins_label = pil.open(os.path.join(dataset_dir, 'other', 'instSegImage', "{:06d}.png".format(img_idx)))
    ins_label_r = resize_arr_ins(ins_label)

    # fig1 = tensor2disp(torch.from_numpy(depthIm_recon).unsqueeze(0).unsqueeze(0), percentile=95, ind=0)
    # fig2 = pil.fromarray(rgb_r)
    # fig3 = pil.fromarray(ins_label_r)
    # combined = np.concatenate([np.array(fig1), np.array(fig2), np.array(fig3)], axis=0)
    # pil.fromarray(combined).show()

    seqNum = int(img_idx / 5000)
    seq_path = os.path.join(target_dir, "{:06d}".format(seqNum))
    depth_path = os.path.join(seq_path, "depth")
    rgb_path = os.path.join(seq_path, "rgb")
    ins_path = os.path.join(seq_path, "ins")
    os.makedirs(seq_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(ins_path, exist_ok=True)

    # Save
    depth_img_path = os.path.join(depth_path, "{:06d}.png".format(img_idx))
    rgb_img_path = os.path.join(rgb_path, "{:06d}.png".format(img_idx))
    ins_img_path = os.path.join(ins_path, "{:06d}.png".format(img_idx))

    pil.fromarray(tsv_depth).save(depth_img_path)
    pil.fromarray(rgb_r).save(rgb_img_path)
    pil.fromarray(ins_label_r).save(ins_img_path)

    dr = time.time() - st
    print("Finished %d images, remain time %f hours" % (img_idx, dr / (img_idx + 1) * (51050 - img_idx) / 60 / 60))


