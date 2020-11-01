from __future__ import absolute_import, division, print_function
import torch
import os
import random
import numpy as np
import PIL as pil
import torch.utils.data as data
import time
import glob
import pandas as pd
import epipolarminer.pandaset as pandaset
from matplotlib import pyplot as plt
import transforms3d as t3d

class PandarSet(data.Dataset):
    def __init__(self,
                 datasetroot
                 ):
        super(PandarSet, self).__init__()

        self.datasetroot = datasetroot
        self.dataset = pandaset.DataSet('/media/shengjie/disk1/data/padaset')
        self.__getfilenames()

        self.loadedseqind = -1
    def __getfilenames(self):
        self.filenames = list()
        seqs = sorted(glob.glob(os.path.join(self.datasetroot, '*')))
        for seq in seqs:
            fronimgpaths = sorted(glob.glob(os.path.join(seq, 'camera', 'front_camera', '*.jpg')))
            for imgpath in fronimgpaths:
                entry = "{} {}".format(imgpath.split('/')[-4].zfill(3), imgpath.split('/')[-1].split('.')[0].zfill(2))
                self.filenames.append(entry)

    def __getitem__(self, index):
        inputs = {}

        seqind = int(self.filenames[index].split(' ')[0])
        frameind = int(self.filenames[index].split(' ')[1])
        if self.loadedseqind != seqind:
            self.loadedseqind = seqind
            self.seqdata = self.dataset[str(seqind).zfill(3)]
            self.seqdata.load()

        camera_to_load = ['front_camera']
        for camname in camera_to_load:
            cameradata = dict()

            cameradata['rgbarr'] = np.array(self.seqdata.camera[camname][frameind])

            camera_pose = self.seqdata.camera[camname].poses[frameind]
            camera_heading = camera_pose['heading']
            camera_position = camera_pose['position']
            extrinsic = np.linalg.inv(self._heading_position_to_mat(camera_heading, camera_position))

            cameradata['extrinsic'] = extrinsic
            cameradata['intrinsic'] = self._cvt2intrinsic(self.seqdata.camera[camname].intrinsics)

            inputs[camname] = cameradata

        self.seqdata.lidar.set_sensor(0)
        inputs['sweeplidar'] = self.seqdata.lidar.data[frameind].to_numpy()[:, :3]

        self.seqdata.lidar.set_sensor(1)
        inputs['solidlidar'] = self.seqdata.lidar.data[frameind].to_numpy()[:, :3]

        return inputs

    def _heading_position_to_mat(self, heading, position):
        quat = np.array([heading["w"], heading["x"], heading["y"], heading["z"]])
        pos = np.array([position["x"], position["y"], position["z"]])
        transform_matrix = t3d.affines.compose(np.array(pos), t3d.quaternions.quat2mat(quat), [1.0, 1.0, 1.0])
        return transform_matrix

    def _cvt2intrinsic(self, cameraintrinsic):
        K = np.eye(4, dtype=np.float64)
        K[0, 0] = cameraintrinsic.fx
        K[1, 1] = cameraintrinsic.fy
        K[0, 2] = cameraintrinsic.cx
        K[1, 2] = cameraintrinsic.cy
        return K

    def __len__(self):
        return len(self.filenames)


