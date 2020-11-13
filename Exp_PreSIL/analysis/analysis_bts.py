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

import argparse

default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",                  type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                    type=str,                               help="path to kitti gt file")
parser.add_argument("--semanticspred_path",         type=str,   default='None',             help="path to kitti semantics prediction file")
parser.add_argument("--instancepred_path",          type=str,   default='None',             help="path to kitti instance prediction file")
parser.add_argument("--split",                      type=str,                               help="train/val split to use")
parser.add_argument("--num_layers",                 type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                     type=int,   default=320,                help="input image height")
parser.add_argument("--width",                      type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                       type=int,   default=352,                help="cropped image height")
parser.add_argument("--crpw",                       type=int,   default=1216,               help="cropped image width")
parser.add_argument("--min_depth",                  type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",                  type=float, default=100.0,              help="maximum depth")
parser.add_argument("--vlsfold",                    type=str)

# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=1,                 help="batch size")
parser.add_argument("--load_weights_folder_depth",  type=str,   default=None,               help="name of models to load")
parser.add_argument("--load_weights_folder_depthbs",   type=str,   default=None,               help="name of models to load")

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
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = "cuda"

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()

    def val(self):
        """Validate the model on a single minibatch
        """
        fpath = os.path.join(os.getcwd(), "splits", self.opt.split, "{}_files.txt")
        val_filenames = readlines(fpath.format("test"))

        semidenseroot = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt'
        filterlidaroot = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'
        btspred = '/home/shengjie/Documents/bts/result_bts_eigen_v2_pytorch_densenet161/pred'
        kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
        deptherrpred = list()

        xx, yy = np.meshgrid(range(1216), range(352), indexing='xy')
        cm = plt.cm.get_cmap('seismic')

        count = 0
        for entry in val_filenames:
            seq, index, dir = entry.split(' ')
            semidensegt = pil.open(os.path.join(filterlidaroot, seq, 'image_02', "{}.png".format(index)))

            rgb = pil.open(os.path.join(kittiroot, seq, 'image_02', "data", "{}.png".format(index)))

            w, h = semidensegt.size
            top = int(h - self.opt.crph)
            left = int((w - self.opt.crpw) / 2)

            semidensegt = semidensegt.crop((left, top, left + self.opt.crpw, top + self.opt.crph))
            semidensegt = np.array(semidensegt).astype(np.float)
            semidensegt = semidensegt / 256.0

            rgb = rgb.crop((left, top, left + self.opt.crpw, top + self.opt.crph))

            pred = pil.open(os.path.join(btspred, "{}_{}.png".format(seq.split('/')[1], index)))
            pred = np.array(pred).astype(np.float) / 256.0
            pred = np.clip(pred, a_min=self.MIN_DEPTH, a_max=self.MAX_DEPTH)

            predtorch = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)

            instancepred = pil.open(os.path.join(self.opt.instancepred_path, seq, 'image_02', "{}.png".format(index)))
            instancepred = instancepred.crop((left, top, left + 1216, top + 352))
            instancepred = np.array(instancepred)

            mask = (semidensegt > self.MIN_DEPTH) * (semidensegt < self.MAX_DEPTH)
            cropmask = np.zeros_like(mask)
            cropmask[int(0.40810811 * self.opt.crph): int(0.99189189 * self.opt.crph), int(0.03594771 * self.opt.crpw): int(0.96405229 * self.opt.crpw)] = 1
            mask[cropmask == 0] = 0
            mask = mask == 1

            if np.sum(mask) == 0:
                continue

            gtnp = semidensegt[mask]
            prednp = pred[mask]

            deptherrpred.append(compute_errors(gtnp, prednp))

            vlsxx = xx[mask == 1]
            vlsyy = yy[mask == 1]
            vlscolor = cm((prednp / gtnp - 1) + 0.5)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
            ax1.imshow(rgb)
            ax1.scatter(vlsxx, vlsyy, 4, vlscolor, '.')
            ax2.imshow(tensor2disp(1 / predtorch, vmax=0.2, ind=0))
            plt.savefig(os.path.join('/media/shengjie/disk1/visualization/bts_erranalysis', "{}.png".format(str(count).zfill(3))))
            plt.close()
            count = count + 1
            print("%s finished" % entry)


        err1 = np.array(deptherrpred)
        err1 = np.mean(err1, axis=0)
        print("\nBaseline Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*err1.tolist()) + "\\\\")


if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
