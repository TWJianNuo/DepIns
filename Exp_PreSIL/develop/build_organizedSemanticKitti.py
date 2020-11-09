import glob
from utils import readlines
import PIL.Image as pil
import os
from kitti_utils import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from matplotlib.path import Path
import time
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
labelroot = '/home/shengjie/Documents/Data/semanticKitti'
mappingroot = '/home/shengjie/Documents/Data/semanticKitti/sequencemapping.txt'
kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
filterlidarroot = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'
vlsroot = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/producedInstanceLabel'
vlsroot_rgb = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/producedInstanceLabel_rgb'
outputroot = '/home/shengjie/Documents/Data/organized_kins/from_semankitti'
mappings = readlines(mappingroot)

def retrieveProjectionM(calib_dir, cam):
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    return P_velo2im, im_shape

def retrieveveloind(velo, depthgt, P_velo2im):
    im_shape = depthgt.shape
    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1

    inboundselector = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0) & (velo[:, 0] >= 0) & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    projy = velo_pts_im[:, 1].astype(np.int)
    projx = velo_pts_im[:, 0].astype(np.int)

    gtdepth = depthgt[projy[inboundselector], projx[inboundselector]]
    refdepth = (velo_pts_im[inboundselector, 2] * 256).astype(np.int)

    dismatcheddepth = gtdepth == refdepth
    inboundselector[inboundselector] = dismatcheddepth

    return projy, projx, inboundselector

def savelabel(producedinstancelabel, rgb, figname, vlsroot_rgb, outputroot, cam):
    date = figname[0:10]
    seq = "{}_sync".format(figname[0:21])
    ind = figname.split('_')[-1]
    os.makedirs(os.path.join(outputroot, date, seq, cam), exist_ok=True)
    os.makedirs(os.path.join(vlsroot_rgb, date, seq, cam), exist_ok=True)
    pil.fromarray(producedinstancelabel).save(os.path.join(outputroot, date, seq, cam, "{}.png".format(ind)))

    coloredimg = np.zeros_like(rgb)
    ratio = 0.3
    for k in np.unique(producedinstancelabel):
        if k > 0:
            selector = producedinstancelabel == k
            coloredimg[selector, :] = np.repeat((np.random.random([1, 3]) * 255).astype(np.uint8), np.sum(selector), axis=0)
    combined = rgb.astype(np.float) * (1-ratio) + coloredimg.astype(np.float)*ratio
    combined = combined.astype(np.uint8)
    pil.fromarray(combined).save(os.path.join(vlsroot_rgb, date, seq, cam, "{}.png".format(ind)))

def labelin2d(projy, projx, z, inboundselector, instancelabel, im_shape, img=None, vlspath=None, figname=None, dovls=False):
    zbar = 0.05
    valinstancelabl = instancelabel[inboundselector]

    instancelabelrecorder = np.ones(im_shape, dtype=np.int) * -1
    instancelabelrecorder[projy[inboundselector], projx[inboundselector]] = instancelabel[inboundselector]
    producedinstancelabel = np.zeros(im_shape, dtype=np.uint8)

    framewiseinscount = 1

    if dovls:
        plt.figure(figsize=(16, 9))
        plt.imshow(img)

    for k in np.unique(valinstancelabl):
        if k > 0:
            color = np.random.random([3])
            selector = inboundselector & (instancelabel == k)
            if np.sum(selector) > 0:
                zmin = z[selector].min()
                selector = selector & (z > (zmin + zbar))
                if np.sum(selector) > 5:
                    try:
                        insx = projx[selector]
                        insy = projy[selector]
                        hull = ConvexHull(np.stack([insx, insy], axis=1))
                        hull_path = Path(np.stack([insx, insy], axis=1)[hull.vertices])

                        minx = insx.min()
                        maxx = insx.max()
                        miny = insy.min()
                        maxy = insy.max()
                        ckx, cky = np.meshgrid(range(minx, maxx+1), range(miny, maxy+1), indexing='xy')
                        ckx = ckx.flatten()
                        cky = cky.flatten()
                        ckpts = np.stack([ckx, cky], axis=1)
                        insidehull = hull_path.contains_points(ckpts)

                        if np.sum((instancelabelrecorder[cky[insidehull], ckx[insidehull]] != k) & (instancelabelrecorder[cky[insidehull], ckx[insidehull]] > 0)) == 0:

                            producedinstancelabel[cky[insidehull], ckx[insidehull]] = framewiseinscount
                            framewiseinscount = framewiseinscount + 1

                            if dovls:
                                plt.scatter(insx, insy, 1.0, color=color)
                                for simplex in hull.simplices:
                                    plt.plot(insx[simplex], insy[simplex], color=color)
                    except:
                        continue
    if dovls:
        plt.savefig(os.path.join(vlspath, "{}.png".format(figname)))
        plt.close()
    return producedinstancelabel

def gettotnum(mappings):
    totnum = 0
    for mapentry in mappings:
        semanseqind, kittiorg, startind, endind = mapentry.split(' ')
        totnum = totnum + int(endind) - int(startind) + 1
    return totnum * 2

def getfiletogenerate(outputroot, mappings, cams):
    filetogenerate = list()
    for mapentry in mappings:
        semanseqind, kittiorg, startind, endind = mapentry.split(' ')
        for cam in cams:
            labelcounts = 0
            for idx in range(int(startind), int(endind) + 1):
                filetogenerate.append((mapentry, cam, idx, labelcounts))
                labelcounts = labelcounts + 1

    generationtime = list()
    for entry in filetogenerate:
        mapentry, cam, idx, labelcounts = entry
        date = mapentry.split(' ')[1][0:10]
        seq = "{}_sync".format(mapentry.split(' ')[1])
        ind = "{}.png".format(str(idx).zfill(10))
        figpath = os.path.join(outputroot, date, seq, cam, ind)
        if os.path.exists(figpath):
            generationtime.append(os.path.getctime(figpath))
        else:
            generationtime.append(-1)
    generationtime = np.array(generationtime)
    if generationtime.max() > 0:
        startgenerationind = np.argmax(generationtime)
        startgenerationind = startgenerationind - 10
        if startgenerationind < 0:
            startgenerationind = 0
    else:
        startgenerationind = 0
    return filetogenerate, startgenerationind

st = time.time()
while(True):
    try:
        newcounts = 0
        cams = ['image_02', 'image_03']
        cammapping = {'image_02': 2, 'image_03': 3}
        filetogenerate, startgenerationind = getfiletogenerate(outputroot, mappings, cams)
        totnum = len(filetogenerate)
        print("Generation from: %d" % startgenerationind)

        for count in range(startgenerationind, totnum):
            mapentry, cam, idx, labelcounts = filetogenerate[count]
            semanseqind, kittiorg, startind, endind = mapentry.split(' ')

            kittiorg = '2011_09_26_drive_0014_sync'
            cam = 'image_02'
            P_velo2im, im_shape = retrieveProjectionM(os.path.join(kittiroot, kittiorg[0:10]), cammapping[cam])

            if not os.path.exists(os.path.join(kittiroot, kittiorg[0:10], "{}_sync".format(kittiorg), cam, 'data', "{}.png".format(str(idx).zfill(10)))):
                newcounts = newcounts + 1
                continue
            img = pil.open(os.path.join(kittiroot, kittiorg[0:10], "{}_sync".format(kittiorg), cam, 'data', "{}.png".format(str(idx).zfill(10))))
            depthgt = np.array(pil.open(os.path.join(filterlidarroot, kittiorg[0:10], "{}_sync".format(kittiorg), cam, "{}.png".format(str(idx).zfill(10))))).astype(np.int)
            velo = load_velodyne_points(os.path.join(kittiroot, kittiorg[0:10], "{}_sync".format(kittiorg), 'velodyne_points', "data", "{}.bin".format(str(idx).zfill(10))))

            label = np.fromfile(os.path.join(labelroot, 'sequences', semanseqind, 'labels', "{}.label".format(str(labelcounts).zfill(6))), dtype=np.uint32)
            label = label.reshape((-1))
            instancelabel = label >> 16
            assert velo.shape[0] == instancelabel.shape[0]

            P_velo2im, im_shape = retrieveProjectionM(os.path.join(kittiroot, kittiorg[0:10]), cammapping[cam])
            assert depthgt.shape[0] == im_shape[0] and depthgt.shape[1] == im_shape[1]
            projy, projx, inboundselector = retrieveveloind(velo, depthgt, P_velo2im)

            # dovls = np.random.randint(0, 10) == 0
            dovls = False
            figname = "{}_{}".format(kittiorg, str(idx).zfill(10))
            producedinstancelabel = labelin2d(projy, projx, velo[:, 2], inboundselector, instancelabel, im_shape, img, vlsroot, figname, dovls)
            savelabel(producedinstancelabel, np.array(img), figname, vlsroot_rgb, outputroot, cam)
            labelcounts = labelcounts + 1

            newcounts = newcounts + 1
            dr = time.time() - st
            lefttime = (totnum - count) * (dr / newcounts) / 60 / 60
            print("Finished: %d, remains hours: %f" % (count, lefttime))
        break
    except:
        print("Error occurred, restarting...")
