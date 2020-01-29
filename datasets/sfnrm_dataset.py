import os.path
from datasets.base_dataset import BaseDataset, get_transform
from datasets.image_folder import make_dataset
from PIL import Image
import random
from utils import readlines
import torch.utils.data as data
import cv2
import numpy as np
from kitti_utils import read_calib_file, load_velodyne_points
class SFNormDataset(data.Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(SFNormDataset, self).__init__()
        self.A_paths = readlines(os.path.join('../splits', opt.split, opt.phase + 'A.txt'))
        self.B_paths = readlines(os.path.join('../splits', opt.split, opt.phase + 'B.txt'))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        self.opt = opt
        self.mapping = {'l' : 'image_02', 'r' : 'image_03', 'm' : ''}
        self.init_datasetB_extrinsics()

    def init_datasetB_extrinsics(self):
        seqs = ['0001', '0002', '0006', '0018', '0020']
        self.extrinsics_B = dict()
        for seq in seqs:
            with open(os.path.join(self.opt.dataB_path, seq, 'extrinsicsgt.txt')) as f:
                lines = f.readlines()[1:]
            self.extrinsics_B[seq] = lines

    def get_index(self, index):
        if self.opt.serial_batches:
            index_A = index % self.A_size
            index_B = index % self.B_size
        else:
            index_A = random.randint(0, self.A_size - 1)
            index_B = random.randint(0, self.B_size - 1)

        return index_A, index_B


    def get_camK_A(self, folder, side):
        sidemap = {'l' : 2, 'r' : 3}

        calib_dir = os.path.join(self.opt.dataA_path, folder.split("/")[0])
        cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
        velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_rect = cam2cam['P_rect_0' + str(sidemap[side])].reshape(3, 4)
        P_rect = np.append(P_rect, [[0, 0, 0, 1]], axis = 0)

        intrinsic_A = P_rect
        extrinsic_A = R_cam2rect @ velo2cam

        return intrinsic_A, extrinsic_A

    def get_camK_B(self, seq, frame):
        intrinsic_B = np.array([[725, 0, 620.5, 0], [0, 725, 187.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        extrinsic_B = self.extrinsics_B[seq][int(frame)]
        extrinsic_B = np.fromstring(extrinsic_B, dtype=np.float32, sep=' ')
        extrinsic_B = np.reshape(extrinsic_B[1 : :], [4,4])
        return intrinsic_B, extrinsic_B

    def normalize_depth(self, depthmap):
        depthmap = np.clip(depthmap, a_min=self.opt.min_depth, a_max=self.opt.max_depth)
        depthmap = (depthmap - self.opt.min_depth) / (self.opt.max_depth - self.opt.min_depth)
        depthmap = (depthmap - 0.5) / 0.5
        return depthmap

    def get_flipMat(self, do_flip):
        if do_flip:
            flipMat = np.array([[-1, 0, self.opt.width, 1],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])
        else:
            flipMat = np.eye(4)
        return flipMat.astype(np.float32)

    def get_rescaleMat(self, height, width):
        fx = self.opt.width / width
        fy = self.opt.height / height
        rescaleMat = np.array([[fx, 0, 0, 0], [0, fy, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return rescaleMat.astype(np.float32)

    def get_RGB_Depth_Intrincis_Extrinsic(self, index_A, index_B):
        inputs = {}
        strA = self.A_paths[index_A]
        strAComps = strA.split(' ')
        strB = self.B_paths[index_B]
        strBComps = strB.split(' ')


        # Read RGB
        A_path = os.path.join(self.opt.dataA_path, strAComps[0], self.mapping[strAComps[2]], 'data',
                              strAComps[1] + '.png')
        B_path = os.path.join(self.opt.dataB_path, strBComps[0], self.mapping[strBComps[2]], 'rgb',
                              strBComps[1] + '.png')

        A_rgb = np.array(Image.open(A_path).convert('RGB')).astype(np.float32) / 255
        B_rgb = np.array(Image.open(B_path).convert('RGB')).astype(np.float32) / 255

        # Read Depth
        A_path = os.path.join(self.opt.predDepth_path, strAComps[0], self.mapping[strAComps[2]],
                              strAComps[1] + '.png')
        B_path = os.path.join(self.opt.dataB_path, strBComps[0], self.mapping[strBComps[2]], 'depthgt',
                              strBComps[1] + '.png')

        A_depth = np.array(cv2.imread(A_path, -1)).astype(np.float32) / 256 # max is 80 meters
        B_depth = np.array(cv2.imread(B_path, -1)).astype(np.float32) / 100 # 1 intensity inidicates 1 cm, max is 655.35 meters

        A_depth = self.normalize_depth(A_depth)
        B_depth = self.normalize_depth(B_depth)

        # Read Intrinsic and Extrinsic
        intrinsic_A, extrinsic_A = self.get_camK_A(strAComps[0], strAComps[2])
        intrinsic_B, extrinsic_B = self.get_camK_B(strBComps[0], strBComps[1])

        # Read gtDepth
        gt_depth_path = os.path.join(self.opt.gtDepth_path, strAComps[0], self.mapping[strAComps[2]],
                              strAComps[1] + '.png')
        gt_depth = np.array(cv2.imread(gt_depth_path, -1)).astype(np.float32) / 256

        inputs['A_rgb'] = A_rgb
        inputs['B_rgb'] = B_rgb

        inputs['A_depth'] = A_depth
        inputs['B_depth'] = B_depth

        inputs['intrinsic_A'] = intrinsic_A.astype(np.float32)
        inputs['extrinsic_A'] = extrinsic_A.astype(np.float32)

        inputs['intrinsic_B'] = intrinsic_B.astype(np.float32)
        inputs['extrinsic_B'] = extrinsic_B.astype(np.float32)

        inputs['gt_depth'] = gt_depth
        return inputs

    def do_agumentation(self, inputs):
        resize_imgs = {'A_rgb', 'B_rgb', 'A_depth', 'B_depth'}
        types = {'A', 'B'}
        do_flip = np.random.uniform(0,1,1)[0] > 0.5

        flip_mat = self.get_flipMat(do_flip)

        for type in types:
            height, width, _ = inputs[type + '_rgb'].shape
            resize_mat = self.get_rescaleMat(height = height, width = width)
            inputs['intrinsic_' + type] = flip_mat @ resize_mat @ inputs['intrinsic_' + type]

        for imgentry in resize_imgs:
            inputs[imgentry] = cv2.resize(inputs[imgentry], dsize=(self.opt.width, self.opt.height), interpolation=cv2.INTER_LINEAR)
            if do_flip:
                inputs[imgentry] = np.fliplr(inputs[imgentry]).copy()


        for imgentry in resize_imgs:
            if len(inputs[imgentry].shape) > 2:
                inputs[imgentry] = np.moveaxis(inputs[imgentry], [0,1,2], [1,2,0])
            else:
                inputs[imgentry] = np.expand_dims(inputs[imgentry], 0)


        if do_flip:
            inputs['gt_depth'] = np.fliplr(inputs['gt_depth']).copy()
        inputs['gt_depth'] = np.expand_dims(inputs['gt_depth'], 0)

        return inputs

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """

        inputs = {}

        index_A, index_B = self.get_index(index)

        # Read RGB, Depth, Intrinsic, Extrinsic
        inputs.update(self.get_RGB_Depth_Intrincis_Extrinsic(index_A = index_A, index_B = index_B))

        inputs = self.do_agumentation(inputs)

        return inputs

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
