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

import time

from layers import *
from networks import *

import argparse

from collections import namedtuple
from kitti_utils import labels

default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",              type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                type=str,                               help="path to kitti gt file")
parser.add_argument("--predang_path",           type=str,                               help="path to kitti gt file")
parser.add_argument("--semanticspred_path",     type=str,   default='None',             help="path to kitti semantics prediction file")
parser.add_argument("--val_gt_path",            type=str,                               help="path to validation gt file")
parser.add_argument("--model_name",             type=str,                               help="name of the model")
parser.add_argument("--split",                  type=str,                               help="train/val split to use")
parser.add_argument("--num_layers",             type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                 type=int,   default=320,                help="input image height")
parser.add_argument("--width",                  type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                   type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                   type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",              type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",              type=float, default=100.0,              help="maximum depth")
parser.add_argument("--variancefold",           type=float, default=1)
parser.add_argument("--threeinput",             action='store_true')


# OPTIMIZATION options
parser.add_argument("--batch_size",             type=int,   default=12,                 help="batch size")
parser.add_argument("--num_epochs",             type=int,   default=20,                 help="number of epochs")
parser.add_argument("--load_weights_folder",    type=str,   default=None,               help="name of models to load")
parser.add_argument("--num_workers",            type=int,   default=6,                  help="number of dataloader workers")


ShapeCat = namedtuple('ShapeCat' , [
    'name',         # The identifier of this catogery
    'trainId',      # Original TrainID this catogery
    'categoryId',   # New assigned CategoryID specified to a vector
    'varianceBar',  # Variance Bar to stop receptive field
    'color',        # Color for visualization
    ])

shapecats = [
    ShapeCat(['road', 'sidewalk', 'terrain', 'sky'],        [0, 1, 9, 10],      0,  [-1.00,     -1.00], (0,    0,    0)),
    ShapeCat(['building'],                                  [2],                1,  [0.500,     0.500], (70,   70,   70)),
    ShapeCat(['wall'],                                      [3],                2,  [0.500,     0.500], (102,  102,  156)),
    ShapeCat(['fence'],                                     [4],                3,  [0.500,     0.500], (190,  153,  153)),
    ShapeCat(['pole'],                                      [5],                4,  [0.500,     0.500], (153,  153,  153)),
    ShapeCat(['traffic light'],                             [6],                5,  [0.500,     0.500], (250,  170,  30)),
    ShapeCat(['traffic sign'],                              [7],                6,  [0.500,     0.500], (220,  220,  0)),
    ShapeCat(['vegetation'],                                [8],                7,  [-1.00,     0.030], (107,  142,  35)),
    ShapeCat(['person', 'rider', 'motorcycle', 'bicycle'],  [11, 12, 17, 18],   8,  [0.500,     0.500], (220,  20,   60)),
    ShapeCat(['car'],                                       [13],               9,  [0.500,     0.500], (0,    0,    142)),
    ShapeCat(['truck'],                                     [14],               10, [0.500,     0.500], (0,    0,    70)),
    ShapeCat(['bus'],                                       [15],               11, [0.500,     0.500], (0,    60,   100)),
    ShapeCat(['train'],                                     [16],               12, [0.500,     0.500], (0,    80,   100))
]

totcat = 19
catmapp = dict()
variancebar = np.zeros([len(shapecats), 2])
for cat in shapecats:
    for idx, entry in enumerate(cat.name):
        for l in labels:
            if l.name == entry:
                assert l.trainId == cat.trainId[idx]
                catmapp[l.trainId] = cat.categoryId
                varh, varv = cat.varianceBar
                variancebar[int(cat.categoryId), 0] = varh
                variancebar[int(cat.categoryId), 1] = varv


class Trainer:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = "cuda"

        if self.opt.threeinput:
            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True, num_input_channels=8)
            self.models["encoder"].to(self.device)
        else:
            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
            self.models["encoder"].to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"] = DepthDecoder(self.models["encoder"].num_ch_enc, num_output_channels=1)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        self.models["confidence"] = ConfidenceDecoder(self.models["encoder"].num_ch_enc, num_output_channels=1)
        self.models["confidence"].to(self.device)

        self.set_dataset()

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\t  ", self.opt.split)

        self.load_model()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()

        self.variancebar = torch.from_numpy(variancebar).cuda().float()
        self.variancebar[self.variancebar > 0] = 1e3

        from integrationModule import IntegrationFunction
        self.integrationFunction = IntegrationFunction.apply

    def set_dataset(self):
        """properly handle multiple dataset situation
        """
        test_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "eigen", "test_files.txt")
        val_filenames = readlines(test_fpath)


        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.val_gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path, threeinput=self.opt.threeinput
        )

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.val_num = val_dataset.__len__()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if not key == 'tag':
                inputs[key] = ipt.to(self.device)

        outputs = dict()
        losses = dict()

        if not self.opt.threeinput:
            encoder_feature = self.models['encoder'](inputs['color_aug'])
        else:
            inputdata = torch.cat([inputs['color_aug'], inputs['semanticspred_cat_vls'], inputs['angh_normed'], inputs['angv_normed']], dim=1)
            encoder_feature = self.models['encoder'](inputdata)

        outputs.update(self.models['depth'](encoder_feature))
        outputs.update(self.models['confidence'](encoder_feature))

        # Depth Branch
        losses.update(self.depth_compute_losses(inputs, outputs))

        losses['totLoss'] = losses['depthloss']

        return outputs, losses

    def depth_compute_losses(self, inputs, outputs, itnum):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}

        pred_ang = torch.cat([inputs['angh'], inputs['angv']], dim=1).contiguous()
        pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang).contiguous()

        edge = self.sfnormOptimizer.ang2edge(ang=pred_ang, intrinsic=inputs['K']).int()
        mask = torch.zeros_like(edge, dtype=torch.int)
        mask[:, :, int(0.40810811 * self.opt.crph) : int(0.99189189 * self.opt.crph), :] = 1
        # mask = mask * (1 - edge)
        mask = mask.contiguous()

        scale = 0
        pred_confidence = torch.ones_like(inputs['depthgt'])

        scaled_disp, pred_depth = disp_to_depth(outputs[('disp', scale)], min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
        pred_depth = F.interpolate(pred_depth, [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True)
        pred_depth = pred_depth * self.STEREO_SCALE_FACTOR
        outputs[('depth', scale)] = pred_depth

        for i in range(itnum):
            if i == 0 or i == itnum - 1 or i == 1:
                outputs[('depth_opted', i)] = pred_depth.clone()
            pred_depth_opted = self.integrationFunction(pred_ang, pred_log, pred_confidence, inputs['semanticspred_cat'], mask, pred_depth, self.variancebar, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            pred_depth = pred_depth_opted.clone()

        return losses

    def vls(self, itnum):
        """Validate the model on a single minibatch
        """
        vind = 0
        self.set_eval()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                outputs = dict()
                if not self.opt.threeinput:
                    encoder_feature = self.models['encoder'](inputs['color'])
                else:
                    inputdata = torch.cat([inputs['color'], inputs['semanticspred_cat_vls'], inputs['angh_normed'], inputs['angv_normed']], dim=1)
                    encoder_feature = self.models['encoder'](inputdata)

                outputs.update(self.models['depth'](encoder_feature))

                self.depth_compute_losses(inputs, outputs, itnum=itnum)

                sfnorm_opted_figs = list()
                depth_opted_figs = list()
                for i in range(itnum):
                    if i == 0 or i == itnum - 1 or i == 1:
                        sfnorm_opted = self.sfnormOptimizer.depth2norm(outputs[('depth_opted', i)], intrinsic=inputs['K'])
                        sfnorm_opted_figs.append(np.array(tensor2rgb((sfnorm_opted + 1) / 2, ind=vind)))

                        figd = tensor2disp(1 / outputs[('depth_opted', i)], vmax=0.25, ind=vind)
                        depth_opted_figs.append(np.array(figd))

                sfnorm_opted_fig = np.concatenate(sfnorm_opted_figs, axis=0)
                depth_opted_fig = np.concatenate(depth_opted_figs, axis=0)
                figanghoview = np.concatenate([sfnorm_opted_fig, depth_opted_fig], axis=1)

                figname = inputs['tag'][0].split(' ')[0].split('/')[1] + '_' + inputs['tag'][0].split(' ')[1]
                pil.fromarray(figanghoview).save(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/depthintegrationNum', '{}.png'.format(figname)))

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        models_to_load = ['encoder', 'depth']
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.vls(itnum=50)
