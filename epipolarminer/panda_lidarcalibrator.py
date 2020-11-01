import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

from epipolarminer import PandarSet
import torch.utils.data as data
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import PIL as pil
import matlab
import matlab.engine
from utils import *
import numba
from numba import jit, prange


positions = np.array([0.006250, -0.246875, -0.825000])
directions = np.zeros([3])
epspos = 0.1

def press(event):
    # print('press', event.key)
    # sys.stdout.flush()
    global epspos
    if event.key == 'd':
        positions[0] = positions[0] + epspos
    elif event.key == 'a':
        positions[0] = positions[0] - epspos
    elif event.key == 'w':
        positions[1] = positions[1] + epspos
    elif event.key == 'x':
        positions[1] = positions[1] - epspos
    elif event.key == 'r':
        positions[2] = positions[2] + epspos
    elif event.key == 't':
        positions[2] = positions[2] - epspos
    elif event.key == '+':
        epspos = epspos * 2
    elif event.key == '-':
        epspos = epspos / 2
    print('x: %f, y: %f, z: %f' % (positions[0], positions[1], positions[2]))
    return

def getrotM(dirs, pos):
    yaw = dirs[0]
    pitch = dirs[1]
    roll = dirs[2]

    yawM = np.eye(4)
    yawM[0, 0] = np.cos(yaw)
    yawM[0, 1] = -np.sin(yaw)
    yawM[1, 0] = np.sin(yaw)
    yawM[1, 1] = np.cos(yaw)

    pitchM = np.eye(4)
    pitchM[0, 0] = np.cos(pitch)
    pitchM[0, 2] = np.sin(pitch)
    pitchM[2, 0] = -np.sin(pitch)
    pitchM[2, 2] = np.cos(pitch)

    rollM = np.eye(4)
    rollM[1, 1] = np.cos(roll)
    rollM[1, 2] = -np.sin(roll)
    rollM[2, 1] = np.sin(roll)
    rollM[2, 2] = np.cos(roll)

    affM = yawM @ pitchM @ rollM
    affM[:3, 3] = pos
    return affM

@jit(nopython=True, parallel=False)
def edgeminer(depthmap, intrinsic, bsmvrec, h, w, sr):
    alignbs = 10
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    for m in range(h):
        for n in range(w):
            depthref = depthmap[m, n]
            if depthref > 0:
                alignment = depthref - alignbs
                for i in range(-sr, sr + 1):
                    for j in range(-sr, sr + 1):
                        if i == 0 and j == 0:
                            continue
                        sm = m + i
                        sn = n + j
                        if sm < 0 or sn < 0 or sm >= h or sn >= w:
                            continue
                        elif depthmap[sm, sn] > depthref:
                            pixeldist = np.sqrt((j / fx) ** 2 + (i / fy) ** 2)
                            if np.sqrt(j**2 + i**2) < sr:
                                bsmv = pixeldist / (1 / (depthref-alignment) - 1 / (depthmap[sm, sn]-alignment))
                                if bsmvrec[m, n] == 0:
                                    bsmvrec[m, n] = bsmv
                                elif bsmv < bsmvrec[m, n]:
                                    bsmvrec[m, n] = bsmv

datasetname = 'pandarset'
datasetroot = '/media/shengjie/disk1/data/padaset'
vlsroot = '/media/shengjie/disk1/visualization/pandaset_lidarcalibrated'

datasets = {'pandarset': PandarSet}
lidarname = 'solidlidar'
camname = 'front_camera'
loaded_dataset = datasets[datasetname](datasetroot)
loader = data.DataLoader(datasets[datasetname](datasetroot), 1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

do_calibration = False
do_visualization = True

for batch_idx, inputs in enumerate(loader):
    lidardata = inputs[lidarname].numpy()[0]
    intrinsic = inputs[camname]['intrinsic'].numpy()[0]
    extrinsic = inputs[camname]['extrinsic'].numpy()[0]
    rgbarr = inputs[camname]['rgbarr'].numpy()[0]

    lidardata_catones = np.concatenate([lidardata, np.ones([lidardata.shape[0], 1])], axis=1)

    h, w, _ = rgbarr.shape
    projectedpts = (intrinsic @ extrinsic @ lidardata_catones.T).T
    projectedpts[:, 0] = projectedpts[:, 0] / projectedpts[:, 2]
    projectedpts[:, 1] = projectedpts[:, 1] / projectedpts[:, 2]
    projectedpts = projectedpts[:, 0:3]

    valpts = (projectedpts[:, 2] > 0) * (projectedpts[:, 0] > 0) * (projectedpts[:, 0] < w) * (projectedpts[:, 1] > 0) * (projectedpts[:, 1] < h)
    projectedptsval = projectedpts[valpts, :]

    lidar_camcoord = (extrinsic @ lidardata_catones[valpts, :].T).T
    dummyintrinsic = intrinsic.copy()

    calibM = getrotM(dirs=directions, pos=positions)
    lidar_camcoord_projected = (dummyintrinsic @ calibM @ lidar_camcoord.T).T
    lidar_camcoord_projected[:, 0] = lidar_camcoord_projected[:, 0] / lidar_camcoord_projected[:, 2]
    lidar_camcoord_projected[:, 1] = lidar_camcoord_projected[:, 1] / lidar_camcoord_projected[:, 2]
    lidar_camcoord_projected = lidar_camcoord_projected[:, 0:3]
    distances = lidar_camcoord_projected[:, 2]
    colors = cm.jet(1 / distances * 10)

    # Create Depth map from Lidar View
    lidardepthmap = np.zeros([h, w])
    lidar_camcoord_projected_x = np.round(lidar_camcoord_projected[:, 0])
    lidar_camcoord_projected_y = np.round(lidar_camcoord_projected[:, 1])
    lidar_camcoord_projected_selector = (lidar_camcoord_projected_x >= 0) & (lidar_camcoord_projected_y >= 0) & (lidar_camcoord_projected_x < w) & (lidar_camcoord_projected_y < h)
    lidar_camcoord_projected_roundedval = np.stack([lidar_camcoord_projected_x, lidar_camcoord_projected_y, lidar_camcoord_projected[:, 2]], axis=1)[lidar_camcoord_projected_selector, :]
    lidardepthmapIndrec = np.ones([h, w], dtype=np.int) * -1
    lidardepthmapIndlist = np.array(list(range(np.sum(valpts))))

    # project to image
    lidardepthmap[lidar_camcoord_projected_roundedval[:, 1].astype(np.int), lidar_camcoord_projected_roundedval[:, 0].astype(np.int)] = lidar_camcoord_projected_roundedval[:, 2]
    lidardepthmapIndrec[lidar_camcoord_projected_roundedval[:, 1].astype(np.int), lidar_camcoord_projected_roundedval[:, 0].astype(np.int)] = lidardepthmapIndlist[lidar_camcoord_projected_selector]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(lidardepthmap.shape, lidar_camcoord_projected_roundedval[:, 1], lidar_camcoord_projected_roundedval[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    tmpindprjectedvalindreclist = lidardepthmapIndlist[lidar_camcoord_projected_selector]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(lidar_camcoord_projected_roundedval[pts[0], 0])
        y_loc = int(lidar_camcoord_projected_roundedval[pts[0], 1])
        lidardepthmap[y_loc, x_loc] = lidar_camcoord_projected_roundedval[pts, 2].min()
        lidardepthmapIndrec[y_loc, x_loc] = tmpindprjectedvalindreclist[pts[np.argmin(lidar_camcoord_projected_roundedval[pts, 2])]]
    lidardepthmap[lidardepthmap < 0] = 0
    lidardepthmapIndrec[lidardepthmap < 0] = -1

    vlsxx, vlsyy = np.meshgrid(range(w), range(h), indexing='xy')
    lidarxx = vlsxx[lidardepthmap > 0]
    lidaryy = vlsyy[lidardepthmap > 0]
    lidard = lidardepthmap[lidardepthmap > 0]
    colors = cm.jet(1 / lidard * 10)

    bsmvrec = np.zeros_like(lidardepthmap)
    sr = 12
    edgeminer(lidardepthmap, dummyintrinsic, bsmvrec, h, w, sr)
    edges = (bsmvrec > 0) * (bsmvrec < 0.06)
    edgexx = vlsxx[edges]
    edgeyy = vlsyy[edges]

    plt.figure()
    plt.scatter(lidarxx, lidaryy, color=colors, s=0.5)
    plt.scatter(edgexx, edgeyy, color='k', s=0.5)
    plt.xlim([0, w])
    plt.ylim([h, 0])

    edgeptsind = lidardepthmapIndrec[edges]
    edgepts3d = lidar_camcoord[edgeptsind, :]
    edgepts3dprojected = (intrinsic @ edgepts3d.T).T
    edgepts3dprojected[:, 0] = edgepts3dprojected[:, 0] / edgepts3dprojected[:, 2]
    edgepts3dprojected[:, 1] = edgepts3dprojected[:, 1] / edgepts3dprojected[:, 2]
    edgepts3dprojected = edgepts3dprojected[:, 0:3]
    plt.figure()
    plt.imshow(pil.fromarray(rgbarr))
    distances = edgepts3dprojected[:, 2]
    colors = cm.jet(1 / distances * 10)
    plt.gca().scatter(edgepts3dprojected[:, 0], edgepts3dprojected[:, 1], color=colors, s=0.5)
    plt.xlim([0, w])
    plt.ylim([h, 0])


    from utils import *
    import torch
    edges = (bsmvrec > 0) * (bsmvrec < 0.1)
    tensor2disp(torch.from_numpy(edges).float().unsqueeze(0).unsqueeze(0), vmax=1, ind=0).show()

    testsets = [[1610, 558, 1609, 554], [1610, 558, 1606, 558], [407, 618, 408, 613], [332, 587, 331, 583], [119, 743, 116, 734], [879, 651, 880, 645], [1336, 805, 1335, 805]]
    testsetind = 0
    alignbs = 10
    fx = dummyintrinsic[0, 0]
    fy = dummyintrinsic[1, 1]

    sxf, syf, sxb, syb = testsets[testsetind]
    df = lidardepthmap[syf, sxf]
    dv = lidardepthmap[syb, sxb]

    alignment = df - alignbs

    pixeldist = np.sqrt(((sxf - sxb) / fx) ** 2 + ((syf - syb) / fy) ** 2)
    bsmv = pixeldist / (1 / (df-alignment) - 1 / (dv-alignment))

    m = 805
    n = 1336
    depthref = lidardepthmap[m, n]
    if depthref > 0:
        for i in range(-sr, sr + 1):
            for j in range(-sr, sr + 1):
                if i == 0 and j == 0:
                    continue
                sm = m + i
                sn = n + j
                if sm < 0 or sn < 0 or sm >= h or sn >= w:
                    continue
                elif lidardepthmap[sm, sn] > depthref:
                    pixeldist = np.sqrt((j / fx) ** 2 + (i / fy) ** 2)
                    if np.sqrt(j ** 2 + i ** 2) < sr:
                        bsmv = pixeldist / (1 / depthref - 1 / lidardepthmap[sm, sn])
                        print("srcx: %d, srcy: %d, occx: %d, occy: %d, bs: %f" % (n, m, sn, sm, bsmv))

    if do_calibration:
        plt.ion()
        fig, ax = plt.subplots()
        sc = ax.scatter(lidar_camcoord_projected[:, 0], lidar_camcoord_projected[:, 1], color=colors, s=0.5)
        plt.xlim([0, w])
        plt.ylim([h, 0])
        fig.canvas.mpl_connect('key_press_event', press)
        plt.draw()
        for i in range(1000):
            calibM = getrotM(dirs=directions, pos=positions)
            lidar_camcoord_projected = (dummyintrinsic @ calibM @ lidar_camcoord.T).T
            lidar_camcoord_projected[:, 0] = lidar_camcoord_projected[:, 0] / lidar_camcoord_projected[:, 2]
            lidar_camcoord_projected[:, 1] = lidar_camcoord_projected[:, 1] / lidar_camcoord_projected[:, 2]
            lidar_camcoord_projected = lidar_camcoord_projected[:, 0:3]
            distances = lidar_camcoord_projected[:, 2]
            colors = cm.jet(1 / distances * 10)
            sc.set_offsets(lidar_camcoord_projected[:, 0:2])
            fig.canvas.draw_idle()
            plt.pause(0.1)
        plt.waitforbuttonpress()

    if do_visualization:
        fig, ax = plt.subplots(figsize=(20, 12))
        sc = ax.scatter(lidar_camcoord_projected[:, 0], lidar_camcoord_projected[:, 1], color=colors, s=0.5)
        plt.xlim([0, w])
        plt.ylim([h, 0])

        seqstr, framestr = loaded_dataset.filenames[batch_idx].split(' ')
        figname = "{}_{}.png".format(seqstr, framestr)
        plt.savefig(os.path.join(vlsroot, figname))
        plt.close()

    # # Validate solid lidar
    plt.figure()
    plt.imshow(pil.fromarray(rgbarr))
    distances = projectedptsval[:, 2]
    colors = cm.jet(1 / distances * 10)
    plt.gca().scatter(projectedptsval[:, 0], projectedptsval[:, 1], color=colors, s=0.5)
    plt.xlim([0, w])
    plt.ylim([h, 0])
