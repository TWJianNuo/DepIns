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


maxM = 1000
sMax = 255 ** 3 - 1
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
    depthIm = np.clip(depthIm, a_min=0.1, a_max=maxM)

    depthIm_f = depthIm * (sMax / maxM)
    r = np.floor(depthIm_f / 255 / 255)
    g = np.floor((depthIm_f - r * 255 * 255) / 255)
    b = np.floor((depthIm_f - r * 255 * 255 - g * 255))

    tsv_depth = np.stack([r, g, b], axis=2).astype(np.uint8)
    return tsv_depth, depthIm

def cvt_png2depth_PreSIL(tsv_depth):
    tsv_depth = tsv_depth.astype(np.float)
    depthIm = (tsv_depth[:,:,0] * 255 * 255 + tsv_depth[:,:,1] * 255 + tsv_depth[:,:,2]) / sMax * maxM
    return depthIm

def resize_arr_depth(din):
    depthIm_torch = torch.from_numpy(din).unsqueeze(0).unsqueeze(0)
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



import time
st = time.time()
target_dir = '/home/shengjie/Documents/Data/PreSIL_organized'

import matlab
import matlab.engine
eng = matlab.engine.start_matlab()
for img_idx in range(51075):
    img_idx = 29
    file_path = depth_dir + '/{:06d}.bin'.format(img_idx)
    fd = open(file_path, 'rb')
    f = np.fromfile(fd, dtype=np.float32, count=rows*cols)
    im = f.reshape((rows, cols))

    depthMap = 0.1 / (im + 1e-6)
    # depthIm = batch_ndcToDepth(im)
    # tensor2disp(torch.from_numpy(depthMap).unsqueeze(0).unsqueeze(0), percentile=93, ind=0).show()
    # tensor2disp(torch.from_numpy(depthIm).unsqueeze(0).unsqueeze(0), percentile=93, ind=0).show()

    Tr_velo_to_cam = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    intrinsic = np.array([
        [960, 0, 960],
        [0, 960, 540],
        [0, 0, 1]
    ])
    depthMap_clip = np.clip(depthMap, a_min=0, a_max=100)
    # tensor2disp(torch.from_numpy(depthMap_clip).unsqueeze(0).unsqueeze(0), percentile=93, ind=0).show()

    xx, yy = np.meshgrid(range(1920), range(1080), indexing='xy')
    ons = np.ones_like(xx)
    pts2dr = np.stack([xx, yy, ons], axis=2)

    pts2dr = torch.from_numpy(pts2dr).float().unsqueeze(3)
    intrinsic = torch.from_numpy(intrinsic).float()
    Tr_velo_to_cam = torch.from_numpy(Tr_velo_to_cam).float()

    pts3dr = torch.matmul(torch.inverse(intrinsic @ Tr_velo_to_cam), pts2dr)
    pts3d = pts3dr * torch.from_numpy(depthMap_clip).float().unsqueeze(2).unsqueeze(3).expand([-1,-1,3,-1])


    sampleR = 50
    dx = pts3d[:, :, 0, 0].numpy().flatten()[::sampleR]
    dy = pts3d[:, :, 1, 0].numpy().flatten()[::sampleR]
    dz = pts3d[:, :, 2, 0].numpy().flatten()[::sampleR]

    dx = matlab.double(dx.tolist())
    dy = matlab.double(dy.tolist())
    dz = matlab.double(dz.tolist())

    eng.eval('close all', nargout=0)
    eng.eval('figure()', nargout=0)
    eng.eval('hold on', nargout=0)
    eng.scatter3(dx, dy, dz, 5, 'filled', 'g', nargout=0)
    eng.eval('axis equal', nargout=0)
    eng.eval('grid off', nargout=0)
    eng.eval('xlabel(\'X\')', nargout=0)
    eng.eval('ylabel(\'Y\')', nargout=0)
    eng.eval('zlabel(\'Z\')', nargout=0)
    eng.eval('xlim([0 50])', nargout=0)
    eng.eval('ylim([-40 40])', nargout=0)
    eng.eval('zlim([-3 10])', nargout=0)


    depthIm = resize_arr_depth(depthMap)
    # tensor2disp(torch.from_numpy(depthIm).unsqueeze(0).unsqueeze(0), vmax=100, ind=0).show()
    # tensor2disp(torch.from_numpy(depthMap).unsqueeze(0).unsqueeze(0), vmax=100, ind=0).show()

    scaleM = np.array([
        [1024 / 1920, 0, 0],
        [0, 576 / 1080, -128],
        [0, 0, 1]
    ])

    Tr_velo_to_cam = np.array([
        [0, -1, 0],
        [0, 0, -1],
        [1, 0, 0]
    ])

    intrinsic = np.array([
        [960, 0, 960],
        [0, 960, 540],
        [0, 0, 1]
    ])
    xx, yy = np.meshgrid(range(1024), range(448), indexing='xy')
    ons = np.ones_like(xx)
    pts2dr = np.stack([xx, yy, ons], axis=2)
    pts2dr = torch.from_numpy(pts2dr).float().unsqueeze(3)
    intrinsic = torch.from_numpy(intrinsic).float()
    scaleM = torch.from_numpy(scaleM).float()
    Tr_velo_to_cam = torch.from_numpy(Tr_velo_to_cam).float()


    pts3dr = torch.matmul(torch.inverse(scaleM @ intrinsic @ Tr_velo_to_cam), pts2dr)
    pts3d = pts3dr * torch.from_numpy(depthIm).float().unsqueeze(2).unsqueeze(3).expand([-1,-1,3,-1])


    sampleR = 5
    dx = pts3d[:, :, 0, 0].numpy().flatten()[::sampleR]
    dy = pts3d[:, :, 1, 0].numpy().flatten()[::sampleR]
    dz = pts3d[:, :, 2, 0].numpy().flatten()[::sampleR]

    dx = matlab.double(dx.tolist())
    dy = matlab.double(dy.tolist())
    dz = matlab.double(dz.tolist())

    # eng.eval('close all', nargout=0)
    eng.eval('figure()', nargout=0)
    eng.eval('hold on', nargout=0)
    eng.scatter3(dx, dy, dz, 5, 'filled', 'g', nargout=0)
    eng.eval('axis equal', nargout=0)
    eng.eval('grid off', nargout=0)
    eng.eval('xlabel(\'X\')', nargout=0)
    eng.eval('ylabel(\'Y\')', nargout=0)
    eng.eval('zlabel(\'Z\')', nargout=0)
    eng.eval('xlim([0 50])', nargout=0)
    eng.eval('ylim([-40 40])', nargout=0)
    eng.eval('zlim([-3 10])', nargout=0)





    tsv_depth, depthIm_clipped = cvt_depth2png_PreSIL(depthIm)
    recon_di = cvt_png2depth_PreSIL(tsv_depth)

    rgb = pil.open(os.path.join(dataset_dir, 'image_2', "{:06d}.png".format(img_idx)))
    rgb_r = resize_arr_rgb(rgb)

    ins_label = pil.open(os.path.join(dataset_dir, 'other', 'instSegImage', "{:06d}.png".format(img_idx)))
    ins_label_r = resize_arr_ins(ins_label)

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


