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
        weightsx = torch.Tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0).expand([3, -1, -1, -1])
        weightsx = weightsx / 4 / 2

        weightsy = torch.Tensor([[-1., -2., -1.],
                                 [0., 0., 0.],
                                 [1., 2., 1.]]).unsqueeze(0).unsqueeze(0).expand([3, -1, -1, -1])
        weightsy = weightsy / 4 / 2
        self.diffx = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, groups=3)
        self.diffy = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, bias=False, groups=3)

        self.diffx.weight = nn.Parameter(weightsx, requires_grad=False)
        self.diffy.weight = nn.Parameter(weightsy, requires_grad=False)

        self.diffx = self.diffx.cuda()
        self.diffy = self.diffy.cuda()

        intensitymeasurew = torch.Tensor([[1., 1., 1.],
                                          [1., 1., 1.],
                                          [1., 1., 1.]]).unsqueeze(0).unsqueeze(0) / 9
        intensitymeasurew = intensitymeasurew.expand([-1, 3, -1, -1]) / 3
        self.intensityconv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.intensityconv.weight = nn.Parameter(intensitymeasurew, requires_grad=False)
        self.intensityconv = self.intensityconv.cuda()

        from kitti_utils import labels
        fpath = os.path.join(os.getcwd(), "splits", self.opt.split, "{}_files.txt")
        val_filenames = readlines(fpath.format("test"))
        # vlsnum = 300
        # val_filenames = random.sample(val_filenames, vlsnum)

        semidenseroot = '/home/shengjie/Documents/Data/Kitti/kitti_predSemantics'
        filterlidaroot = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'
        btspred = '/home/shengjie/Documents/bts/result_bts_eigen_v2_pytorch_densenet161/pred'
        kittiroot = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
        deptherrpred = list()

        xx, yy = np.meshgrid(range(1216), range(352), indexing='xy')
        cm = plt.cm.get_cmap('seismic')

        count = 0
        for entry in val_filenames:
            seq, index, dir = entry.split(' ')

            rgb = pil.open(os.path.join(kittiroot, seq, 'image_02', "data", "{}.png".format(index.zfill(10))))

            w, h = rgb.size
            top = int(h - self.opt.crph)
            left = int((w - self.opt.crpw) / 2)

            semantics = pil.open(os.path.join(semidenseroot, seq, 'semantic_prediction/image_02', "{}.png".format(index.zfill(10))))
            semantics = semantics.resize([w, h], pil.NEAREST)

            rgb = rgb.crop((left, top, left + self.opt.crpw, top + self.opt.crph))
            semantics = np.array(semantics.crop((left, top, left + self.opt.crpw, top + self.opt.crph)))

            cvtsemantics = semantics.copy()
            for l in np.unique(semantics):
                cvtsemantics[semantics == l] = labels[l].trainId
            semantics = cvtsemantics
            semanticstorch = torch.from_numpy(semantics.astype(np.float32)).unsqueeze(0).unsqueeze(0)


            rgbtorch = torch.from_numpy(np.array(rgb)).unsqueeze(0).permute([0, 3, 1, 2]).float().cuda() / 255.0
            gradx = self.diffx(rgbtorch)
            gradx = torch.mean(torch.abs(gradx), dim=1, keepdim=True)
            grady = self.diffy(rgbtorch)
            grady = torch.mean(torch.abs(grady), dim=1, keepdim=True)
            intensity = self.intensityconv(rgbtorch)

            semancat = [2, 3, 4]
            semanmask = np.zeros_like(semantics)
            for l in np.unique(semantics):
                if l in semancat:
                    semanmask[semantics == l] = 1
            semanmask = semanmask.astype(np.float)
            maskedge = (gradx + grady) / (intensity + 1e-3) > 0.15
            maskedge = maskedge.cpu().detach().numpy()[0,0,:,:]
            maskedge = maskedge.astype(np.float)
            semanedgemask = semanmask * maskedge
            semanedgemask = np.stack([semanedgemask, semanedgemask, semanedgemask], axis=2)

            figsemantics = tensor2semantic(semanticstorch, ind=0)
            figsemantics = pil.fromarray((np.array(figsemantics).astype(np.float) * (1 - semanedgemask)).astype(np.uint8))


            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
            ax1.imshow(rgb)
            ax2.imshow(figsemantics)
            plt.savefig(os.path.join('/media/shengjie/disk1/visualization/poleconnectionvls', "{}.png".format(str(count).zfill(3))))
            plt.close()
            count = count + 1
            print("%s finished" % entry)



if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
