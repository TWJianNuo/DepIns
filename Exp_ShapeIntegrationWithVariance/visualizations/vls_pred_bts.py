from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from torch.utils.data import DataLoader

# Resolve Tensorbard Confliction across pytorch version
import torch
import warnings

from Exp_PreSIL.dataloader_kitti import KittiDataset

import networks


from layers import *
from networks import *
from kitti_utils import read_calib_file

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

class Trainer:
    def __init__(self):
        self.sfnormOptimizer = SurfaceNormalOptimizer(height=352, width=1216, batch_size=1)

    def val(self):
        """Validate the model on a single minibatch
        """
        fpath = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen/test_files.txt'
        val_filenames = readlines(fpath)

        btspred = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Bts_Pred/result_bts_eigen_v2_pytorch_densenet161/pred'
        kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
        vlsroot = '/media/shengjie/disk1/visualization/shapeintegrationpred/bts'
        os.makedirs(vlsroot, exist_ok=True)

        crph = 352
        crpw = 1216

        count = 0
        for entry in val_filenames:
            seq, index, dir = entry.split(' ')

            rgb = pil.open(os.path.join(kittiroot, seq, 'image_02', "data", "{}.png".format(index)))

            w, h = rgb.size
            top = int(h - crph)
            left = int((w - crpw) / 2)

            calibpath = os.path.join(kittiroot, seq.split('/')[0], 'calib_cam_to_cam.txt')
            cam2cam = read_calib_file(calibpath)
            K = np.eye(4)
            K[0:3, :] = cam2cam['P_rect_02'].reshape(3, 4)
            K[0, 2] = K[0, 2] - left
            K[1, 2] = K[1, 2] - top
            K = torch.from_numpy(K).unsqueeze(0).float()

            pred = pil.open(os.path.join(btspred, "{}_{}.png".format(seq.split('/')[1], index)))
            pred = np.array(pred).astype(np.float) / 256.0

            predtorch = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
            prednorm = self.sfnormOptimizer.depth2norm(depthMap=predtorch, intrinsic=K)

            fig1 = tensor2disp(1 / predtorch, vmax=0.15, ind=0)
            fig2 = tensor2rgb((prednorm + 1) / 2, ind=0)

            fig = np.concatenate([np.array(fig1), np.array(fig2)], axis=1)
            pil.fromarray(fig).save(os.path.join(vlsroot, "{}_{}.png".format(seq.split('/')[1], str(index).zfill(10))))
            count = count + 1
            print("%s finished" % entry)

    def bts_shapeVarInt(self):
        fpath = '/home/shengjie/Documents/Project_SemanticDepth/splits/eigen/test_files.txt'
        val_filenames = readlines(fpath)

        kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
        bsvls = '/media/shengjie/disk1/visualization/shapeintegrationpred/kitti_depth_l1_baseline_semidense/'
        shapeVarIntVls = '/media/shengjie/disk1/visualization/shapeintegrationpred/crfintwvariance_bansemantics_banground_1'
        vlsroot = '/media/shengjie/disk1/visualization/shapeintegrationpred/shapeVarInt_bs'

        crph = 365
        crpw = 1220

        os.makedirs(vlsroot, exist_ok=True)
        count = 0
        for entry in val_filenames:
            seq, index, dir = entry.split(' ')

            rgb = pil.open(os.path.join(kittiroot, seq, 'image_02', "data", "{}.png".format(index)))

            w, h = rgb.size

            left = int((w - crpw) / 2)
            top = int((h - crph) / 2)
            imgcropped = rgb.crop((left, top, left + crpw, top + crph))

            fig0 = np.concatenate([np.array(imgcropped), np.array(imgcropped)], axis=1)
            fig1 = pil.open(os.path.join(bsvls, "{}_{}.png".format(seq.split('/')[1], str(index).zfill(10))))
            fig2 = pil.open(os.path.join(shapeVarIntVls, "{}_{}.png".format(seq.split('/')[1], str(index).zfill(10))))
            # fig2 =fig2.resize([fig1.size[0], int(fig1.size[0] / fig2.size[0] * fig2.size[1])])

            fig = np.concatenate([np.array(fig0), np.array(fig2), np.array(fig1)], axis=0)
            pil.fromarray(fig).save(os.path.join(vlsroot, "{}_{}.png".format(seq.split('/')[1], str(index).zfill(10))))
            count = count + 1
            print("%s finished" % entry)

if __name__ == "__main__":
    trainer = Trainer()
    # trainer.val()
    trainer.bts_shapeVarInt()
