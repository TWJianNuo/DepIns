import cv2
import numpy as np
from matplotlib import cm
from PIL import Image
import os
import math
import torch
from utils import *

# Constants
rows = 1080 # image height
cols = 1920 # image width

dataset_dir = "/home/shengjie/Documents/Data/PreSIL"
depth_dir = os.path.join(dataset_dir, 'depth')
lidar_dir = '/home/shengjie/Documents/Data/PreSIL_organized/{}/lidar'

# # Predefined Kitti Intrinsic Matrix, (fx = 594.89, fy = 615.71)
# # Predefined PreSIL Intrinsic Matrix, (fx = 960, fy = 960)
cut = 4
w = 1024
h = 576


maxM = 1000
sMax = 255 ** 3 - 1
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

def cvt_depth2png(depthIm):
    tsv_depth = np.clip(depthIm, a_min=0, a_max=255.99609375)
    tsv_depth = (tsv_depth * 256.0).astype(np.uint16)
    return tsv_depth

def cvt_png2depth(tsv_depth):
    depthIm = (tsv_depth).astype(np.float32) / 256.0
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

def check_validity(seqNum, img_idx, target_dir):
    valid = True
    try:
        pil.open(os.path.join(target_dir, str(seqNum).zfill(6), 'rgb', "{}.png".format(str(img_idx).zfill(6))))
        pil.open(os.path.join(target_dir, str(seqNum).zfill(6), 'depth', "{}.png".format(str(img_idx).zfill(6))))
    except:
        valid = False

    return valid

import time
st = time.time()
target_dir = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/PreSIL_organized'

# import matlab
# import matlab.engine
# eng = matlab.engine.start_matlab()

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

# create velo scan mask
r = 10
rsw = w
rsh = h - 32 * cut
lidar_mask = np.zeros([rsh, rsw], dtype=np.bool)
lidar_angleDensity = np.linspace(0, 2 * np.pi, int(8e2))
lidar_heightDensity = np.linspace(-7, 1, int(50))
lidar_dummyscan = list()
for lh in lidar_heightDensity:
    lidar_dummyscan.append(np.stack([np.cos(lidar_angleDensity) * r, np.sin(lidar_angleDensity) * r, np.ones(len(lidar_angleDensity)) * lh, np.ones(len(lidar_angleDensity))], axis=1))
lidar_dummyscan = np.concatenate(lidar_dummyscan, axis=0)
lidar_dummyscan = lidar_dummyscan[lidar_dummyscan[:,0] > 1, :]

mapP = np.eye(4)
mapP[0:3, 0:3] = scaleM @ intrinsic @ Tr_velo_to_cam
lidar_dummyscanp2d = (mapP @ lidar_dummyscan.T).T
lidar_dummyscanp2d[:, 0] = lidar_dummyscanp2d[:, 0] / lidar_dummyscanp2d[:, 2]
lidar_dummyscanp2d[:, 1] = lidar_dummyscanp2d[:, 1] / lidar_dummyscanp2d[:, 2]
lidar_dummyscanp2d = np.round(lidar_dummyscanp2d[:, 0:2]).astype(np.int)
valsel = (lidar_dummyscanp2d[:, 0] >= 0) & (lidar_dummyscanp2d[:, 1] >= 0) & (lidar_dummyscanp2d[:, 0] < rsw) & (lidar_dummyscanp2d[:, 1] < rsh)
lidar_dummyscanp2dval = lidar_dummyscanp2d[valsel, :]
lidar_mask[lidar_dummyscanp2dval[:, 1], lidar_dummyscanp2dval[:, 0]] = True

pil.fromarray(lidar_mask.astype(np.uint8) * 255).save(os.path.join(target_dir, 'lidar_mask.png'))

for img_idx in range(51075):
    seqNum = int(img_idx / 5000)

    if check_validity(seqNum, img_idx, target_dir):
        continue

    file_path = depth_dir + '/{:06d}.bin'.format(img_idx)
    fd = open(file_path, 'rb')
    f = np.fromfile(fd, dtype=np.float32, count=rows*cols)
    im = f.reshape((rows, cols))

    depthMap = 0.1 / (im + 1e-6)

    depthIm = resize_arr_depth(depthMap)

    tsv_depth = cvt_depth2png(depthIm)

    rgb = pil.open(os.path.join(dataset_dir, 'image_2', "{:06d}.png".format(img_idx)))
    rgb_r = resize_arr_rgb(rgb)

    # Save
    seq_path = os.path.join(target_dir, "{:06d}".format(seqNum))
    depth_path = os.path.join(seq_path, "depth")
    rgb_path = os.path.join(seq_path, "rgb")
    os.makedirs(seq_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(rgb_path, exist_ok=True)

    depth_img_path = os.path.join(depth_path, "{:06d}.png".format(img_idx))
    rgb_img_path = os.path.join(rgb_path, "{:06d}.png".format(img_idx))

    pil.fromarray(tsv_depth).save(depth_img_path)
    pil.fromarray(rgb_r).save(rgb_img_path)

    dr = time.time() - st
    print("Finished %d images, remain time %f hours" % (img_idx, dr / (img_idx + 1) * (51050 - img_idx) / 60 / 60))

    # Visualization of original Depth map
    '''
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

    xx, yy = np.meshgrid(range(1920), range(1080), indexing='xy')
    ons = np.ones_like(xx)
    pts2dr = np.stack([xx, yy, ons], axis=2)

    pts2dr = torch.from_numpy(pts2dr).float().unsqueeze(3)
    intrinsic = torch.from_numpy(intrinsic).float()
    Tr_velo_to_cam = torch.from_numpy(Tr_velo_to_cam).float()

    pts3dr = torch.matmul(torch.inverse(intrinsic @ Tr_velo_to_cam), pts2dr)
    pts3d = pts3dr * torch.from_numpy(depthMap_clip).float().unsqueeze(2).unsqueeze(3).expand([-1,-1,3,-1])

    sampleR = 2
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
    '''

    # Visualization of save-ready Depth map
    '''
    depthMap = cvt_png2depth(cvt_depth2png(depthMap))
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
    '''

    # Visualization of Lidar
    '''
    sampleR = 1
    dx = points[::sampleR, 0]
    dy = points[::sampleR, 1]
    dz = points[::sampleR, 2]

    dx = matlab.double(dx.tolist())
    dy = matlab.double(dy.tolist())
    dz = matlab.double(dz.tolist())

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

    cm = plt.get_cmap('magma')
    colors = cm(1 / points2d[:,2] * 5)
    plt.figure()
    plt.imshow(rgb_r)
    plt.scatter(points2d[:, 0], points2d[:, 1], s=0.1, c=colors[:, 0:3])
    '''