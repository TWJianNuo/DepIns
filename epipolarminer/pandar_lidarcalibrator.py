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
from epipolarminer.icp import icp


positions = np.array([0.006250, -0.246875, -0.825000])
# positions = np.array([0.006250, -0.246875, -0.325000])
directions = np.zeros([3])
epspos = 0.1
stopflag = False
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
    elif event.key == 'q':
        stopflag = True
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

datasetname = 'pandarset'
datasetroot = '/media/shengjie/disk1/data/pandarset'
vlsroot = '/media/shengjie/disk1/visualization/pandarset_vls'
rawlidarroot = '/media/shengjie/disk1/data/pandarset/rawlidarscan'

datasets = {'pandarset': PandarSet}
lidarname = 'solidlidar'
camname = 'front_camera'
loaded_dataset = datasets[datasetname](datasetroot, None)
loader = data.DataLoader(loaded_dataset, 1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)

do_calibration = True
do_visualization = True

for batch_idx in range(loader.dataset.__len__()):
    # batch_idx = 20
    inputs = loader.dataset.__getitem__(batch_idx)

    lidardata = inputs[lidarname]
    intrinsic = inputs[camname]['intrinsic']
    extrinsic = inputs[camname]['extrinsic']
    rgbarr = inputs[camname]['rgbarr']

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

    # rawsweeplidar = inputs['rawlidar']
    # rawsweeplidar = np.concatenate([rawsweeplidar, np.ones([rawsweeplidar.shape[0], 1])], axis=1).T
    # rawsweeplidar_wolrd = inputs['sweeplidar']
    # rawsweeplidar_wolrd = np.concatenate([rawsweeplidar_wolrd, np.ones([rawsweeplidar_wolrd.shape[0], 1])], axis=1).T
    # t_world_sweeplidar, distances, _ = icp(rawsweeplidar_wolrd[0:3, :].T, rawsweeplidar[0:3, :].T, max_iterations=200)
    # cvt_rawsweeplidar_wolrd = t_world_sweeplidar @ rawsweeplidar_wolrd
    # err = np.sqrt(np.sum((t_world_sweeplidar @ rawsweeplidar_wolrd - rawsweeplidar)[0:3, :] ** 2, axis=0)).mean()

    # import matlab
    # import matlab.engine
    #
    # eng = matlab.engine.start_matlab()
    # vlszero = matlab.double(np.zeros([1]).tolist())
    #
    # vlsx = matlab.double(lidardata[:, 0].tolist())
    # vlsy = matlab.double(lidardata[:, 1].tolist())
    # vlsz = matlab.double((-lidardata[:, 2]).tolist())
    #
    # eng.eval('figure()', nargout=0)
    # eng.scatter3(vlsx, vlsy, vlsz, 1, 'filled', nargout=0)
    # eng.eval('hold on', nargout=0)
    # eng.scatter3(vlszero, vlszero, vlszero, 10, 'filled', 'r', nargout=0)
    # eng.eval('axis equal', nargout=0)
    #
    # sampledIndex = int(np.random.randint(0, rawsweeplidar.shape[1] - 2, 1))
    # vlsxr = matlab.double(rawsweeplidar[0, :].tolist())
    # vlsyr = matlab.double(rawsweeplidar[1, :].tolist())
    # vlszr = matlab.double(rawsweeplidar[2, :].tolist())

    stopflag = False
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
            if stopflag is False:
                break
        # plt.waitforbuttonpress()
        plt.close()

    if do_visualization:
        fig, ax = plt.subplots(figsize=(20, 12))
        sc = ax.scatter(lidar_camcoord_projected[:, 0], lidar_camcoord_projected[:, 1], color=colors, s=0.5)
        plt.xlim([0, w])
        plt.ylim([h, 0])

        seqstr, framestr = loaded_dataset.filenames[batch_idx].split(' ')
        os.makedirs(os.path.join(vlsroot, 'lidarview', seqstr), exist_ok=True)
        figname = "{}.jpg".format(os.path.join(vlsroot, 'lidarview', seqstr, framestr))
        plt.savefig(figname)
        plt.close()

    # Validate solid lidar
    os.makedirs(os.path.join(vlsroot, 'cameraview', seqstr), exist_ok=True)
    figname = "{}.jpg".format(os.path.join(vlsroot, 'cameraview', seqstr, framestr))
    plt.figure(figsize=(20, 12))
    plt.imshow(pil.fromarray(rgbarr))
    distances = projectedptsval[:, 2]
    colors = cm.jet(1 / distances * 10)
    plt.gca().scatter(projectedptsval[:, 0], projectedptsval[:, 1], color=colors, s=0.5)
    plt.xlim([0, w])
    plt.ylim([h, 0])
    plt.savefig(figname)
    plt.close()
