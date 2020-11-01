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
parser.add_argument("--trainmapping_fold",      type=str)
parser.add_argument("--instancelabel_path",     type=str)
parser.add_argument("--vlsfold",                type=str)

# OPTIMIZATION options
parser.add_argument("--num_epochs",             type=int,   default=20,                 help="number of epochs")
parser.add_argument("--load_weights_folder",    type=str,   default=None,               help="name of models to load")
parser.add_argument("--load_sfnweights_folder", type=str,   default=None,               help="name of models to load")
parser.add_argument("--load_angweights_folder", type=str,   default=None,               help="name of models to load")
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

        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder"].to(self.device)

        self.models["depth"] = DepthDecoder(self.models["encoder"].num_ch_enc, num_output_channels=1)
        self.models["depth"].to(self.device)

        self.models["sfmencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["sfmencoder"].to(self.device)

        self.models["sfmdecoder"] = DepthDecoder(self.models["encoder"].num_ch_enc, num_output_channels=3)
        self.models["sfmdecoder"].to(self.device)

        self.models["angencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["angencoder"].to(self.device)

        self.models["angdecoder"] = DepthDecoder(self.models["encoder"].num_ch_enc, num_output_channels=2)
        self.models["angdecoder"].to(self.device)

        self.set_dataset()

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\t  ", self.opt.split)

        self.load_model()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=1).cuda()

        self.variancebar = torch.from_numpy(variancebar).cuda().float()
        self.variancebar[self.variancebar > 0] = 1e8

        from integrationModule import IntegrationFunction
        self.integrationFunction = IntegrationFunction.apply

    def get_indmapping(self, mapping):
        wins_ind = dict()
        for idx, m in enumerate(mapping):
            if len(m) > 1:
                wins_ind[mapping[idx]] = idx
        return wins_ind

    def set_dataset(self):
        """properly handle multiple dataset situation
        """
        test_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "test_files.txt")
        self.val_filenames = readlines(test_fpath)

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.val_gt_path, self.val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path, threeinput=self.opt.threeinput
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.val_num = val_dataset.__len__()

        mapping = readlines(os.path.join(self.opt.trainmapping_fold, 'training_mapping.txt'))
        self.indmapping = self.get_indmapping(mapping)

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

        encoder_feature = self.models['encoder'](inputs['color_aug'])

        outputs.update(self.models['depth'](encoder_feature))
        outputs.update(self.models['confidence'](encoder_feature))

        # Depth Branch
        losses.update(self.depth_compute_losses(inputs, outputs))

        losses['totLoss'] = losses['depthloss']

        return outputs, losses

    def get_instanceLabel(self, batch_idx):
        entry = '{} {} {}'.format(self.val_filenames[batch_idx].split(' ')[0].split('/')[0], self.val_filenames[batch_idx].split(' ')[0].split('/')[1], self.val_filenames[batch_idx].split(' ')[1].zfill(10))
        ind = self.indmapping[entry]
        instancelabel = pil.open(os.path.join(self.opt.instancelabel_path, '{}_10.png'.format(str(ind).zfill(6))))

        rgbpath = os.path.join(self.opt.data_path, self.val_filenames[batch_idx].split(' ')[0], 'image_02', 'data', '{}.png'.format(self.val_filenames[batch_idx].split(' ')[1].zfill(10)))
        rgb = pil.open(rgbpath)

        assert instancelabel.size == rgb.size

        w, h = instancelabel.size
        left = int((w - self.opt.crpw) / 2)
        top = int((h - self.opt.crph) / 2)
        instancelabel = instancelabel.crop((left, top, left + self.opt.crpw, top + self.opt.crph))
        semanticlabel = (np.array(instancelabel) / 256).astype(np.int)
        instancelabel = np.array(instancelabel) % 256

        instancelabel = torch.from_numpy(instancelabel).unsqueeze(0).unsqueeze(0).int().cuda().contiguous()

        return semanticlabel, instancelabel

    def draw_instanceLabel(self, semanticlabel, instancelabel):
        from kitti_utils import labels
        cropped_semanticspred_copy = np.array(semanticlabel.copy())
        for l in np.unique(np.array(cropped_semanticspred_copy)):
            cropped_semanticspred_copy[cropped_semanticspred_copy == l] = labels[l].trainId
        cropped_semanticspred_copy = torch.from_numpy(cropped_semanticspred_copy).unsqueeze(0).unsqueeze(0).cuda().float()
        figseman = tensor2semantic(cropped_semanticspred_copy, shapeCat=False)
        figseman = np.array(figseman)

        instancelabelnp = instancelabel[0, 0].cpu().numpy()
        for idx in np.unique(instancelabelnp):
            if idx > 0:
                rndcolor = (np.random.random([3]) * 255).astype(np.int)
                figseman[instancelabelnp == idx, 0] = rndcolor[0]
                figseman[instancelabelnp == idx, 1] = rndcolor[1]
                figseman[instancelabelnp == idx, 2] = rndcolor[2]
        return figseman

    def prepare_variancebar(self, instancelabel):
        variancebar_perframe = torch.zeros([instancelabel.max() + 1, 2])
        variancebar_perframe[0, :] = -1
        variancebar_perframe[1:instancelabel.max() + 1, :] = 1e3
        variancebar_perframe = variancebar_perframe.cuda()
        return variancebar_perframe

    def vls(self, itnum):
        """Validate the model on a single minibatch
        """
        vind = 0
        self.set_eval()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                # if batch_idx != self.val_filenames.index('2011_09_26/2011_09_26_drive_0009_sync 364 l'):
                #     continue
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                outputs = dict()
                encoder_feature = self.models['encoder'](inputs['color'])
                outputs.update(self.models['depth'](encoder_feature))

                outputsnorm = dict()
                outputsnorm.update(self.models['angdecoder'](self.models['angencoder'](inputs['color'])))
                ang_gtsize = (F.interpolate(outputsnorm[('disp', 0)], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True) - 0.5) * 2 * np.pi
                inputs['angh'] = ang_gtsize[:, 0, :, :].unsqueeze(1)
                inputs['angv'] = ang_gtsize[:, 1, :, :].unsqueeze(1)

                outputsnorm = dict()
                outputsnorm.update(self.models['sfmdecoder'](self.models['sfmencoder'](inputs['color'])))
                norm_blur_gtsize = F.interpolate(outputsnorm[('disp', 0)], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True)
                norm_ang = self.sfnormOptimizer.ang2normal(torch.cat([inputs['angh'], inputs['angv']], dim=1), intrinsic=inputs['K'])

                semanticlabel, instancelabel = self.get_instanceLabel(batch_idx)
                # variancebar_perframe = self.prepare_variancebar(instancelabel)

                pred_ang = torch.cat([inputs['angh'], inputs['angv']], dim=1).contiguous()
                pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang).contiguous()

                edge = self.sfnormOptimizer.ang2edge(ang=pred_ang, intrinsic=inputs['K']).int()
                mask = torch.zeros_like(edge, dtype=torch.int)
                mask[:, :, int(0.40810811 * self.opt.crph): -1, :] = 1
                mask = mask.contiguous()
                mask = (mask * instancelabel > 0).int()

                scale = 0

                scaled_disp, pred_depth = disp_to_depth(outputs[('disp', scale)], min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
                pred_depth = F.interpolate(pred_depth, [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True)
                pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

                lamb = 0.05
                depth_optedin = pred_depth.clone()
                depth_optedout = torch.zeros_like(depth_optedin)
                import shapeintegration_cuda
                for i in range(itnum):
                    if i == 0:
                        outputs[('depth_opted', i)] = depth_optedin.clone()
                    elif i == itnum - 1:
                        outputs[('depth_opted', i)] = depth_optedout.clone()
                    shapeintegration_cuda.shapeIntegration_crf_forward(pred_ang, pred_log, instancelabel, mask, pred_depth, depth_optedin, depth_optedout, self.opt.crph, self.opt.crpw, 1, lamb)
                    depth_optedin = depth_optedout.clone()

                # vls
                sfnorm_opted_figs = list()
                depth_opted_figs = list()
                for i in range(itnum):
                    if i == 0 or i == itnum - 1:
                        sfnorm_opted = self.sfnormOptimizer.depth2norm(outputs[('depth_opted', i)], intrinsic=inputs['K'])
                        sfnorm_opted_figs.append(np.array(tensor2rgb((sfnorm_opted + 1) / 2, ind=vind)))
                        figd = tensor2disp(1 / outputs[('depth_opted', i)], vmax=0.25, ind=vind)
                        depth_opted_figs.append(np.array(figd))

                sfnorm_opted_fig = np.concatenate(sfnorm_opted_figs, axis=0)
                depth_opted_fig = np.concatenate(depth_opted_figs, axis=0)

                fig_normblur = tensor2rgb(norm_blur_gtsize, ind=vind)
                fig_ang = tensor2rgb((norm_ang + 1) / 2, ind=vind)
                fig_norm = np.concatenate([np.array(fig_normblur), np.array(fig_ang)], axis=1)

                figrgb = tensor2rgb(F.interpolate(inputs['color'], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True), ind=vind)
                figinstance = self.draw_instanceLabel(semanticlabel, instancelabel)
                figscene = np.concatenate([np.array(figrgb), np.array(figinstance)], axis=1)

                figanghoview = np.concatenate([sfnorm_opted_fig, depth_opted_fig], axis=1)
                figanghoview = np.concatenate([figanghoview, fig_norm, figscene], axis=0)
                figname = inputs['tag'][0].split(' ')[0].split('/')[1] + '_' + inputs['tag'][0].split(' ')[1]
                pil.fromarray(figanghoview).save(os.path.join(self.opt.vlsfold, '{}.png'.format(figname)))

    def load_model(self):
        """Load model(s) from disk
        """
        load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(load_weights_folder), "Cannot find folder {}".format(load_weights_folder)
        print("loading model from folder {}".format(load_weights_folder))

        models_to_load = ['encoder', 'depth']
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}.pth".format(n))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        load_weights_folder = os.path.expanduser(self.opt.load_sfnweights_folder)

        assert os.path.isdir(load_weights_folder), "Cannot find folder {}".format(load_weights_folder)
        print("loading model from folder {}".format(load_weights_folder))

        models_to_load = ['sfmencoder', 'sfmdecoder']
        modelnameMap = {'sfmencoder': 'encoder', 'sfmdecoder': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}.pth".format(modelnameMap[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        load_weights_folder = os.path.expanduser(self.opt.load_angweights_folder)

        assert os.path.isdir(load_weights_folder), "Cannot find folder {}".format(load_weights_folder)
        print("loading model from folder {}".format(load_weights_folder))

        models_to_load = ['angencoder', 'angdecoder']
        modelnameMap = {'angencoder': 'encoder', 'angdecoder': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}.pth".format(modelnameMap[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.vls(itnum=5)
