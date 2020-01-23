# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset
from kitti_utils import read_calib_file, load_velodyne_points
from kitti_utils import labels
from utils import *
import cv2

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        if os.path.isfile(velo_filename):
            depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
            depth_gt = skimage.transform.resize(
                depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

            if do_flip:
                depth_gt = np.fliplr(depth_gt)
        else:
            depth_gt = np.zeros(self.full_res_shape[::-1]).astype(np.float32)

        return depth_gt

    def get_detection(self, folder, frame_index, side, do_flip, imorgSize):
        if side == 'l':
            camid = '2'
        else:
            camid = '3'
        label_path = os.path.join(self.detect_path, folder, 'detection_label', 'image_0' + str(camid), str(frame_index).zfill(10) + '.txt')
        with open(label_path) as f:
            labels = f.readlines()
        box2d = self.extract_2dbox_from_detectlabel(labels, imorgSize, do_flip)
        return box2d

    def extract_2dbox_from_detectlabel(self, labels, imorgSize, do_flip):
        box2d = list()
        box2dn = np.ones([self.maxLoad, 4]) * -1
        ratioh = imorgSize[0] / self.height
        ratiow = imorgSize[1] / self.width
        for label in labels:
            comps = label.split(' ')
            x1 = np.clip(float(comps[4]), 0, imorgSize[1] - 1)
            y1 = np.clip(float(comps[5]), 0, imorgSize[0] - 1)
            x2 = np.clip(float(comps[6]), 0, imorgSize[1] - 1)
            y2 = np.clip(float(comps[7]), 0, imorgSize[0] - 1)

            if do_flip:
                x1 = imorgSize[1] - x1
                x2 = imorgSize[1] - x2

                # Switch x1 and x2 to maintain order
                tmp = x2
                x2 = x1
                x1 = tmp
            box2d.append([x1 / ratiow, y1 / ratioh, x2 / ratiow, y2 / ratioh])
        if len(box2d) > 0:
            box2d = np.array(box2d)
            box2dn[0 : box2d.shape[0], :] = box2d
        box2dn = box2dn.astype(np.float32)
        return box2dn

    def get_flipMat(self, do_flip):
        if do_flip:
            flipMat = np.array([[-1, 0, self.width, 1],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
        else:
            flipMat = np.eye(4)
        return flipMat

    def get_rescaleMat(self, im_shape):
        height, width = im_shape
        fx = self.width / width
        fy = self.height / height
        rescaleMat = np.array([[fx, 0, 0, 0], [0, fy, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return rescaleMat

    def get_velo(self, velo_filename):
        velo = np.zeros([100000, 4]).astype(np.float32)
        velo_pts = load_velodyne_points(velo_filename)
        velo_pts = velo_pts[velo_pts[:, 0] > 0, :]
        np.random.shuffle(velo_pts)
        copyNum = np.min([velo_pts.shape[0], velo.shape[0]])
        velo[0 : copyNum, :] = velo_pts[0 : copyNum, :]

        return velo

    def get_camK(self, folder, frame_index, side, do_flip):
        outputs = {}

        sidemap = {'l' : 2, 'r' : 3}

        calib_dir = os.path.join(self.data_path, folder.split("/")[0])
        velo_filename = os.path.join(self.data_path, folder, "velodyne_points", "data",
                                     "{:010d}.bin".format(int(frame_index)))
        cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # get image shape
        im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

        # get formation mat
        flipMat = self.get_flipMat(do_flip)
        rescaleMat = self.get_rescaleMat(im_shape)

        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_rect = cam2cam['P_rect_0' + str(sidemap[side])].reshape(3, 4)
        P_rect = np.append(P_rect, [[0, 0, 0, 1]], axis = 0)

        realIn = flipMat @ rescaleMat @ P_rect
        realEx = R_cam2rect @ velo2cam
        camK = realIn @ realEx
        invcamK = np.linalg.inv(camK)

        velo = self.get_velo(velo_filename)

        outputs['camK'] = camK.astype(np.float32)
        outputs['invcamK'] = invcamK.astype(np.float32)
        outputs['realIn'] = realIn.astype(np.float32)
        outputs['realEx'] = realEx.astype(np.float32)
        outputs['velo'] = velo

        return outputs

    def get_seman(self, folder, frame_index, side, do_flip):
        rgb_path = self.get_image_path(folder, frame_index, side)
        if 'image_03' in rgb_path:
            semantic_label_path = rgb_path.replace('image_03/data', 'semantic_prediction/image_03')
        elif 'image_02' in rgb_path:
            semantic_label_path = rgb_path.replace('image_02/data', 'semantic_prediction/image_02')

        semantic_label = pil.open(semantic_label_path)
        if do_flip:
            semantic_label = semantic_label.transpose(pil.FLIP_LEFT_RIGHT)
        semantic_label_copy = np.array(semantic_label.copy())
        for k in np.unique(semantic_label):
            semantic_label_copy[semantic_label_copy == k] = labels[k].trainId
        # visualize_semantic(semantic_label_copy)
        semantic_label_copy = np.expand_dims(semantic_label_copy, axis=0)
        return semantic_label_copy

    def get_bundlePose(self, folder, frame_index):
        poseM = list()
        for i in range(len(self.frame_idxs)):
            poseM.append(self.get_pose(folder, frame_index + i))
        return np.stack(poseM, axis=0)

    def get_pose(self, folder, frame_index):
        filePath = os.path.join(self.data_path, folder, 'oxts', 'data', str(frame_index).zfill(10) + '.txt')
        with open(filePath) as f:
            gps_enntry = f.readlines()[0]
        oxts = gps_enntry.split(' ')

        # Start to compute
        scale = latToScale(float(oxts[0]))
        t1, t2 = latlonToMercator(float(oxts[0]), float(oxts[1]), scale)
        t3 = float(oxts[2])
        T = np.transpose(np.array([t1, t2, t3]))

        rx = float(oxts[3])
        ry = float(oxts[4])
        rz = float(oxts[5])

        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx

        poseM = np.eye(4)
        poseM[0:3, 0:3] = R
        poseM[0:3, 3] = T

        poseM = poseM.astype(np.float32)
        return poseM

    def get_predDepth(self, folder, frame_index, side, do_flip):
        mapping = {'l' : 'image_02', 'r' : 'image_03'}
        filePath = os.path.join(self.predDepthPath, folder, mapping[side], str(frame_index).zfill(10) + '.png')
        predDepth = np.array(cv2.imread(filePath, -1))
        if do_flip:
            predDepth = np.fliplr(predDepth)
        predDepth = predDepth.astype(np.float32) / 256
        predDepth = np.expand_dims(predDepth, axis=0)
        return predDepth

class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
