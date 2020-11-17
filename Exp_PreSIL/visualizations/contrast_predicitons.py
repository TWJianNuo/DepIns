from __future__ import absolute_import, division, print_function

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

parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",                  type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                    type=str,                               help="path to kitti gt file")
parser.add_argument("--semanticspred_path",         type=str,   default='None',             help="path to kitti semantics prediction file")
parser.add_argument("--split",                      type=str,                               help="train/val split to use")
parser.add_argument("--num_layers",                 type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                     type=int,   default=320,                help="input image height")
parser.add_argument("--width",                      type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                       type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                       type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",                  type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",                  type=float, default=100.0,              help="maximum depth")
parser.add_argument("--vlsfold",                    type=str)


# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=1,                 help="batch size")
parser.add_argument("--load_weights_folder",  type=str,   default=None,              help="name of models to load")

class Trainer:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.set_dataset()

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.minabsrel = 1e10
        self.maxa1 = -1e10

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()

        self.device = "cuda"
    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.getcwd(), "splits", self.opt.split, "{}_files.txt")
        val_filenames = readlines(fpath.format("test"))

        nummax = 300
        random.seed(0)
        random.shuffle(val_filenames)
        val_filenames = val_filenames[0:nummax]
        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False
        )

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False)

        self.val_num = val_dataset.__len__()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        modelnames = ['intconstrainWallPoleBs', 'intconstrainPole', 'intconstrainWall', 'intconstrainWallPole', 'intconstrainPole2', 'intconstrainPole4', 'intconstrainPole3', 'intconstrainPole5']

        vlsroot = os.path.join(self.opt.vlsfold, 'vls')
        os.makedirs(vlsroot, exist_ok=True)
        for batch_idx, inputs in enumerate(self.val_loader):
            for key, ipt in inputs.items():
                if not key == 'tag':
                    inputs[key] = ipt.to(self.device)

            imgs = list()
            figname = "{}_{}.png".format(inputs['tag'][0].split(' ')[0].split('/')[1], inputs['tag'][0].split(' ')[1])
            for k in range(len(modelnames)):
                predepth = pil.open(os.path.join(self.opt.vlsfold, modelnames[k], figname))
                predepth = np.array(predepth).astype(np.float) / 256.0
                predepth = torch.from_numpy(predepth).unsqueeze(0).unsqueeze(0)
                imgs.append(np.array(tensor2disp(1 / predepth, vmax=0.2, ind=0)))
            imgl = np.concatenate(imgs[0::2], axis=0)
            imgr = np.concatenate(imgs[1::2], axis=0)
            imgvls = np.concatenate([imgl, imgr], axis=1)
            # imgvls = np.concatenate(imgs, axis=0)
            pil.fromarray(imgvls).save(os.path.join(vlsroot, figname))


if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
