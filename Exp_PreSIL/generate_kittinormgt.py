from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

import PIL.Image as pil
from utils import *
from kitti_utils import *

def get_entry_from_path(imgpath):
    comps = imgpath.split('/')
    if comps[-2] == 'image_02':
        direct = 'l'
    else:
        direct = 'r'
    entry = comps[-4] + '/' + comps[-3] + ' ' + comps[-1].split('.')[0] + ' ' + direct + '\n'
    return entry

def collect_all_entries(folder):
    import glob
    dates = [f.path for f in os.scandir(folder) if f.is_dir()]
    entries = list()
    for date in dates:
        seqs = [f.path for f in os.scandir(date) if f.is_dir()]
        for seq in seqs:
            foldl = os.path.join(seq, 'image_02')
            for imgpath in glob.glob(foldl + '/*.png'):
                entries.append(get_entry_from_path(imgpath))
            foldr = os.path.join(seq, 'image_03')
            for imgpath in glob.glob(foldr + '/*.png'):
                entries.append(get_entry_from_path(imgpath))
    return entries

unoccluded_depthfolder = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'
unoccluded_surfacefolder = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/surface_normal_from_unoccluded_lidar'
rawdata_root = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
entries = collect_all_entries(unoccluded_depthfolder)
dirmapping = {'l': 'image_02', 'r': 'image_03'}
for entry in entries:
    entry = '2011_09_29/2011_09_29_drive_0004_sync 0000000134 l\n'
    seq, frame, dir = entry[:-1].split(' ')
    rgbpath = os.path.join(rawdata_root, seq, dirmapping[dir], "data", "{}.png".format(frame.zfill(10)))
    surfacepath = os.path.join(unoccluded_surfacefolder, seq, dirmapping[dir], "{}.png".format(frame.zfill(10)))
    depthpath = os.path.join(unoccluded_depthfolder, seq, dirmapping[dir], "{}.png".format(frame.zfill(10)))

    rgb = pil.open(rgbpath)
    depth = np.array(pil.open(depthpath)).astype(np.float32) / 256.0
    surface = np.array(pil.open(surfacepath)).astype(np.float32) / 127.5 - 1

    w, h = rgb.size

    cam2cam = read_calib_file(os.path.join(rawdata_root, seq.split('/')[0], 'calib_cam_to_cam.txt'))
    K = np.eye(4)
    K[0:3, :] = cam2cam['P_rect_0{}'.format(dirmapping[dir][-1])].reshape(3, 4)

    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    vallidarmask = depth > 0

    vlsxx = xx[vallidarmask]
    vlsyy = yy[vallidarmask]
    vlsd = depth[vallidarmask]
    vlsnorm = surface[vallidarmask, :]
    pts3d = np.linalg.inv(K) @ np.stack([vlsxx * vlsd, vlsyy * vlsd, vlsd, np.ones_like(vlsd)], axis=0)
    pts3d = pts3d.T

    obsy = 314
    obsx = 586
    obsw = 7
    obsnorm = surface[obsy, obsx, :]
    obsxx = xx[obsy - obsw : obsy + obsw, obsx - obsw : obsx + obsw]
    obsxx = obsxx[vallidarmask[obsy - obsw : obsy + obsw, obsx - obsw : obsx + obsw]]
    obsyy = yy[obsy - obsw : obsy + obsw, obsx - obsw : obsx + obsw]
    obsyy = obsyy[vallidarmask[obsy - obsw : obsy + obsw, obsx - obsw : obsx + obsw]]
    obsd = depth[obsy - obsw : obsy + obsw, obsx - obsw : obsx + obsw]
    obsd = obsd[vallidarmask[obsy - obsw : obsy + obsw, obsx - obsw : obsx + obsw]]
    obs3d = np.linalg.inv(K) @ np.stack([obsxx * obsd, obsyy * obsd, obsd, np.ones_like(obsd)], axis=0)
    obs3d = obs3d.T

    u, s, vh = np.linalg.svd(obs3d.T)
    normdir = u[0:3, 3]
    normdir = normdir / np.sqrt(np.sum(normdir ** 2))

    import matlab
    import matlab.engine
    eng = matlab.engine.start_matlab()

    pts3dxm = matlab.double(pts3d[:, 0].tolist())
    pts3dym = matlab.double(pts3d[:, 1].tolist())
    pts3dzm = matlab.double(pts3d[:, 2].tolist())
    normxm = matlab.double(vlsnorm[:, 0].tolist())
    normym = matlab.double(vlsnorm[:, 1].tolist())
    normzm = matlab.double(vlsnorm[:, 2].tolist())

    eng.eval('figure()', nargout=0)
    eng.scatter3(pts3dxm, pts3dym, pts3dzm, 5, 'g', 'filled', nargout=0)
    eng.eval('axis equal', nargout=0)
    eng.eval('hold on', nargout=0)
    eng.quiver3(pts3dxm, pts3dym, pts3dzm, normxm, normym, normzm, nargout=0)
    eng.xlabel('X', nargout=0)
    eng.ylabel('Y', nargout=0)
    eng.zlabel('Z', nargout=0)

    obs3dxm = matlab.double(obs3d[:, 0].tolist())
    obs3dym = matlab.double(obs3d[:, 1].tolist())
    obs3dzm = matlab.double(obs3d[:, 2].tolist())
    obsnormxm = matlab.double(obsnorm[0:1].tolist())
    obsnormym = matlab.double(obsnorm[1:2].tolist())
    obsnormzm = matlab.double(obsnorm[2:3].tolist())
    obsnormxmd = matlab.double(obs3d[0:1, 0].tolist())
    obsnormymd = matlab.double(obs3d[0:1, 1].tolist())
    obsnormzmd = matlab.double(obs3d[0:1, 2].tolist())

    eng.eval('figure()', nargout=0)
    eng.scatter3(obs3dxm, obs3dym, obs3dzm, 5, 'g', 'filled', nargout=0)
    eng.eval('axis equal', nargout=0)
    eng.eval('hold on', nargout=0)
    eng.quiver3(obsnormxmd, obsnormymd, obsnormzmd, obsnormxm, obsnormym, obsnormzm, 0.1, nargout=0)
    eng.xlabel('X', nargout=0)
    eng.ylabel('Y', nargout=0)
    eng.zlabel('Z', nargout=0)

    vls = surface
    vls = torch.from_numpy(surface).permute([2, 0, 1]).unsqueeze(0)
    selvls = torch.sum(vls ** 2, dim=1, keepdim=True) < 2
    vls[selvls.expand([-1, 3, -1, -1])] = -vls[selvls.expand([-1, 3, -1, -1])]
    tensor2rgb((vls + 1) / 2, ind=0).show()

    plt.figure()
    plt.imshow(pil.open(surfacepath))