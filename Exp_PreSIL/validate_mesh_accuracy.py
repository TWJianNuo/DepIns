from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import os

import PIL.Image as pil
from utils import *
from kitti_utils import *

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

@numba.jit(nopython=True, parallel=True)
def compute_triangplaneapprox_err(triangles, xloc, yloc, height, width, freeptrecorder, depth, gtdepthrecorder, approxdepthrecorder, fx, fy, bx, by, isplanefitting):
    numcount = 0
    for i in range(triangles.shape[0]):

        ptxs = xloc[triangles[i]] + 1e-3
        ptys = yloc[triangles[i]] + 1e-3

        minx = int(np.floor(np.min(ptxs)))
        maxx = int(np.ceil(np.max(ptxs)))

        miny = int(np.floor(np.min(ptys)))
        maxy = int(np.ceil(np.max(ptys)))

        area = 0.5 * (-ptys[1] * ptxs[2] + ptys[0] * (-ptxs[1] + ptxs[2]) + ptxs[0] * (ptys[1] - ptys[2]) + ptxs[1] * ptys[2])

        insidepts = list()
        trianggtdepth = list()

        vstackcount = 0

        for m in range(miny, maxy + 1):
            for n in range(minx, maxx + 1):
                isInRange = (m < height) and (n < width) and (m >= 0) and (n >= 0)
                if isInRange:
                    if (depth[m, n] > 0) and freeptrecorder[m, n]:
                        my = m
                        nx = n
                        s = 1 / (2 * area) * (ptys[0] * ptxs[2] - ptxs[0] * ptys[2] + (ptys[2] - ptys[0]) * nx + (ptxs[0] - ptxs[2]) * my)
                        t = 1 / (2 * area) * (ptxs[0] * ptys[1] - ptys[0] * ptxs[1] + (ptys[0] - ptys[1]) * nx + (ptxs[1] - ptxs[0]) * my)

                        if s > 0 and t > 0 and 1 - (s + t) > 0:
                            insidepts.append((n, m))
                            freeptrecorder[m, n] = False
                            trianggtdepth.append(depth[m, n])

                            if isplanefitting:
                                if vstackcount == 0:
                                    triangpts = np.expand_dims(np.array([(n - bx) / fx * depth[m, n], (m - by) / fy * depth[m, n], depth[m, n], 1]), 0)
                                    triangpts2d = np.expand_dims(np.array([n, m]), 0)
                                else:
                                    triangpts = np.vstack((triangpts, np.expand_dims(np.array([(n - bx) / fx * depth[m, n], (m - by) / fy * depth[m, n], depth[m, n], 1]), 0)))
                                    triangpts2d = np.vstack((triangpts2d, np.expand_dims(np.array([n, m]), 0)))
                                vstackcount = vstackcount + 1

        if len(trianggtdepth) > 0:
            meanapproxdepth = np.mean(np.array(trianggtdepth))
            if (not isplanefitting) or len(trianggtdepth) < 3:
                for kk in range(len(trianggtdepth)):
                    gtdepthrecorder[numcount] = trianggtdepth[kk]
                    approxdepthrecorder[numcount] = meanapproxdepth
                    numcount = numcount + 1
            else:
                evl, singval, evr = np.linalg.svd(triangpts.T)
                planeparam = evl[:, -1]
                planeapproxdepth_records = list()
                for kk in range(len(trianggtdepth)):
                    div = (planeparam[0] * (triangpts2d[kk, 0] - bx) / fx + planeparam[1] * (triangpts2d[kk, 1] - by) / fy + planeparam[2])
                    if np.abs(div) > 1e-7:
                        planeapproxdepth = -planeparam[3] / div
                    else:
                        planeapproxdepth = -1e6
                    planeapproxdepth_records.append(planeapproxdepth)

                meanapproxerr = 0
                planeapproxerr = 0
                for kk in range(len(trianggtdepth)):
                    meanapproxerr = meanapproxerr + np.abs(meanapproxdepth - trianggtdepth[kk])
                    planeapproxerr = planeapproxerr + np.abs(planeapproxdepth_records[kk] - trianggtdepth[kk])

                useplaneapprox = (planeapproxerr < meanapproxerr) and (np.array(planeapproxdepth_records).min() > 0)

                for kk in range(len(trianggtdepth)):
                    if useplaneapprox:
                        approxdepthrecorder[numcount] = planeapproxdepth_records[kk]
                    else:
                        approxdepthrecorder[numcount] = meanapproxdepth
                    gtdepthrecorder[numcount] = trianggtdepth[kk]
                    numcount = numcount + 1
    return

def meshextractor_perscale(mesh_extractor, rgb, scale):
    w, h = rgb.size

    sw = int(w / (2 ** scale))
    sh = int(h / (2 ** scale))

    rgb_scaled = rgb.resize((sw, sh), pil.LANCZOS)

    base_mesh = mesh_extractor(np.array(rgb_scaled))

    triangles = base_mesh.faces
    xloc = base_mesh.verts_2d[:, 0] * w
    yloc = base_mesh.verts_2d[:, 1] * h
    return triangles, xloc, yloc

semidense_depthfolder = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt'
rawdata_root = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
split_root = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen/test_files.txt'
dirmapping = {'l': 'image_02', 'r': 'image_03'}

with open(split_root) as f:
    entries = f.readlines()

valentries = list()
for entry in entries:
    seq, frame, dir = entry[:-1].split(' ')
    depthpath = os.path.join(semidense_depthfolder, seq, dirmapping[dir], "{}.png".format(frame.zfill(10)))
    if os.path.isfile(depthpath):
        valentries.append(entry)
np.random.seed(0)
np.random.shuffle(valentries)
valentries = valentries[0:10]

from tridepth.extractor import Mesh2DExtractor

mesh_extractor = Mesh2DExtractor(canny_params={"denoise": False}, at_params={"filter_itr": 4, "error_thresh": 0.01}, min_edge_size=12)

for scale in range(3):
    count = 0
    err_recorder = list()
    for entry in valentries:
        seq, frame, dir = entry[:-1].split(' ')
        rgbpath = os.path.join(rawdata_root, seq, dirmapping[dir], "data", "{}.png".format(frame.zfill(10)))
        depthpath = os.path.join(semidense_depthfolder, seq, dirmapping[dir], "{}.png".format(frame.zfill(10)))

        rgb = pil.open(rgbpath)
        depth = np.array(pil.open(depthpath)).astype(np.float32) / 256.0

        w, h = rgb.size

        cam2cam = read_calib_file(os.path.join(rawdata_root, seq.split('/')[0], 'calib_cam_to_cam.txt'))
        K = np.eye(4)
        K[0:3, :] = cam2cam['P_rect_0{}'.format(dirmapping[dir][-1])].reshape(3, 4)

        fx = K[0, 0]
        bx = K[0, 2]
        fy = K[1, 1]
        by = K[1, 2]

        triangles, xloc, yloc = meshextractor_perscale(mesh_extractor, rgb, scale)

        freeptrecorder = np.ones([h, w], dtype=np.bool)
        gtdepthrecorder = np.zeros(np.sum(depth > 0))
        approxdepthrecorder = np.zeros(np.sum(depth > 0))
        compute_triangplaneapprox_err(triangles, xloc, yloc, h, w, freeptrecorder, depth, gtdepthrecorder, approxdepthrecorder, fx, fy, bx, by, True)

        gtdepthrecorder = gtdepthrecorder[gtdepthrecorder > 0]
        approxdepthrecorder = approxdepthrecorder[approxdepthrecorder > 0]

        err_recorder.append(compute_errors(gtdepthrecorder, approxdepthrecorder))

        print("Scale: %d, Frame: %d" % (scale, count))

        if count == 0:
            import matplotlib.tri as tri
            triang = tri.Triangulation(xloc, yloc, triangles)
            fig1, ax1 = plt.subplots(figsize=(16, 32))
            ax1.imshow(rgb)
            ax1.set_aspect('equal')
            ax1.triplot(triang, 'b-', lw=1)
            ax1.set_xlim([0, w])
            ax1.set_ylim([h, 0])
            ax1.set_title("Triangulation at scale {}".format(scale))
            fig1.savefig(os.path.join('/home/shengjie/Desktop/scale_{}.png'.format(scale)))
            plt.close(fig1)
            print("figure scale {} saved".format(scale))

        count = count + 1

    mean_errors = np.mean(np.array(err_recorder), axis=0)
    print("\nScale: %d" % scale)
    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

