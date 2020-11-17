from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from torch.utils.data import DataLoader

# Resolve Tensorbard Confliction across pytorch version
from torch.utils.tensorboard import SummaryWriter
from Exp_PreSIL.dataloader_kitti import KittiDataset

import networks

import time
import json

from layers import *
from networks import *

import argparse


default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",              type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                type=str,                               help="path to kitti gt file")
parser.add_argument("--predang_path",           type=str,                               help="path to kitti gt file")
parser.add_argument("--semanticspred_path",     type=str,                               help="path to kitti gt file")
parser.add_argument("--val_gt_path",            type=str,                               help="path to validation gt file")
parser.add_argument("--model_name",             type=str,                               help="name of the model")
parser.add_argument("--split",                  type=str,                               help="train/val split to use")
parser.add_argument("--log_dir",                type=str,   default=default_logpath,    help="path to log file")
parser.add_argument("--num_layers",             type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                 type=int,   default=320,                help="input image height")
parser.add_argument("--width",                  type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                   type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                   type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",              type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",              type=float, default=100.0,              help="maximum depth")
parser.add_argument("--print_freq",             type=int,   default=50)
parser.add_argument("--val_frequency",          type=int,   default=10)
parser.add_argument("--integrationlossw",       type=float, default=1)
parser.add_argument("--banshuffle",             action='store_true')
parser.add_argument("--addpole",                action='store_true')
parser.add_argument("--addbuilding",            action='store_true')



# OPTIMIZATION options
parser.add_argument("--batch_size",             type=int,   default=12,                 help="batch size")
parser.add_argument("--learning_rate",          type=float, default=1e-4,               help="learning rate")
parser.add_argument("--num_epochs",             type=int,   default=20,                 help="number of epochs")
parser.add_argument("--scheduler_step_size",    type=int,   default=15,                 help="step size of the scheduler")
parser.add_argument("--load_weights_folder",    type=str,   default=None,               help="name of models to load")
parser.add_argument("--num_workers",            type=int,   default=6,                  help="number of dataloader workers")

# LOGGING options
parser.add_argument("--log_frequency",          type=int,   default=250,                help="number of batches between each tensorboard log")
parser.add_argument("--save_frequency",         type=int,   default=1,                  help="number of epochs between each save")

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
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = "cuda"

        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder"].to(self.device)

        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"] = DepthDecoder(self.models["encoder"].num_ch_enc, num_output_channels=1)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        print("Training model named:\t", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\t", self.opt.log_dir)
        print("Training is using:\t", self.device)

        self.set_dataset()
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\t  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items".format(self.train_num, self.val_num))

        if self.opt.load_weights_folder is not None:
            self.load_model()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.minabsrel = 1e10
        self.maxa1 = -1e10

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()

        from integrationModule import IntegrationConstrainFunction
        self.integrationConstrainFunction = IntegrationConstrainFunction.apply

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
    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.getcwd(), "splits", self.opt.split, "{}_files.txt")
        test_fpath = os.path.join(os.getcwd(), "splits", self.opt.split, "test_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(test_fpath)

        train_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, train_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path
        )

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.val_gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path
        )

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.train_num = train_dataset.__len__()
        self.val_num = val_dataset.__len__()
        self.num_total_steps = self.train_num // self.opt.batch_size * self.opt.num_epochs

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
        self.set_train()

        for batch_idx in (range(self.train_loader.dataset.__len__())):
            batch_idx = self.train_loader.dataset.filenames.index("2011_09_26/2011_09_26_drive_0060_sync 6 l")
            inputs = self.train_loader.dataset.__getitem__(batch_idx)

            for key, ipt in inputs.items():
                if not key == 'tag':
                    if not key == 'K':
                        inputs[key] = ipt.unsqueeze(0).to(self.device).repeat([2, 1, 1, 1])
                    else:
                        inputs[key] = ipt.unsqueeze(0).to(self.device).repeat([2, 1, 1])

            pred_ang = torch.cat([inputs['angh'], inputs['angv']], dim=1).contiguous()
            pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang).contiguous()

            semanticspred = inputs['semanticspred'].int().contiguous()

            resizedcolor = F.interpolate(inputs['color'], (self.opt.crph, self.opt.crpw), mode='nearest')
            gradx = self.diffx(resizedcolor)
            gradx = torch.mean(torch.abs(gradx), dim=1, keepdim=True)
            grady = self.diffy(resizedcolor)
            grady = torch.mean(torch.abs(grady), dim=1, keepdim=True)
            intensity = self.intensityconv(resizedcolor)
            maskedge = ((gradx + grady) / (intensity + 1e-3) > 0.15).float()
            semanmask = torch.zeros_like(semanticspred)
            semanedgemask = torch.zeros_like(semanticspred)

            if self.opt.addbuilding:
                semancat = [2, 3, 4]
                for l in torch.unique(inputs['semanticspred']):
                    if l in semancat:
                        semanmask[inputs['semanticspred'] == l] = 1
                semanedgemask = semanmask * (1 - maskedge)

            if self.opt.addpole:
                semancat = [5, 6, 7]
                for l in torch.unique(inputs['semanticspred']):
                    if l in semancat:
                        semanedgemask[inputs['semanticspred'] == l] = 1

            # exclude the top area
            semanedgemask[:, :, 0:int(0.40 * self.opt.crph), :] = 0

            # exclude up normal
            normfromang = self.sfnormOptimizer.ang2normal(ang=pred_ang, intrinsic=inputs['K'])
            semanedgemask = semanedgemask * (normfromang[:, 1, :, :].unsqueeze(1) < 0.8).int()

            # exclude singular normal point
            singularnorm = self.sfnormOptimizer.ang2edge(ang=pred_ang, intrinsic=inputs['K'])
            semanedgemask = semanedgemask * (1 - singularnorm).int()
            semanedgemask = semanedgemask.int().contiguous()

            vallidarmask = (inputs['depthgt'] > 0).float()

            for k in range(500):
                outputs = dict()
                encoder_feature = self.models['encoder'](inputs['color'])
                outputs.update(self.models['depth'](encoder_feature))

                depthloss_plain = 0
                depthloss_int = 0
                for scale in range(4):
                    scaled_disp, pred_depth = disp_to_depth(outputs[('disp', scale)], min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
                    pred_depth = F.interpolate(pred_depth, [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=False)
                    pred_depth = pred_depth * self.STEREO_SCALE_FACTOR
                    outputs[('depth', scale)] = pred_depth

                    shape_constrain = self.integrationConstrainFunction(pred_log, semanticspred, semanedgemask, pred_depth, self.opt.crph, self.opt.crpw, self.opt.batch_size)
                    outputs[('shapeconstrain', scale)] = shape_constrain

                    depthloss_int = depthloss_int + shape_constrain.mean()
                    depthloss_plain = depthloss_plain + torch.sum(torch.abs(pred_depth - inputs['depthgt']) * vallidarmask) / (torch.sum(vallidarmask) + 1)

                depthloss_int = depthloss_int / 4
                depthloss_plain = depthloss_plain / 4

                self.model_optimizer.zero_grad()
                (depthloss_plain + depthloss_int * self.opt.integrationlossw).backward()
                self.model_optimizer.step()
                print("Iteration: %d, plainloss: %f, constainloss: %f" % (k, float((depthloss_plain).detach().cpu().numpy()), float(depthloss_int.detach().cpu().numpy())))


            tensor2disp(1 / pred_depth, vmax=0.2, ind=0).save(os.path.join('/media/shengjie/disk1/visualization/offlinefinetune', "{}.png".format(str(self.opt.integrationlossw)).zfill(3)))

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

        # Depth Branch
        losses.update(self.depth_compute_losses(inputs, outputs))

        losses['totLoss'] = losses['depthloss_plain'] + losses['depthloss_int'] * self.opt.integrationlossw

        return outputs, losses
    def record_img(self, inputs, outputs, recoder='train'):
        vind = 0

        vlscolor = F.interpolate(inputs['color'], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=False)
        figrgb = np.array(tensor2rgb(vlscolor, ind=vind))
        figseman = tensor2semantic(inputs['semanticspred'], ind=0)

        ratio = 0.3
        combined = np.array(figrgb).astype(np.float) * (1 - ratio) + (np.array(figseman)).astype(np.float) * ratio
        combined = combined.astype(np.uint8)

        figd1 = tensor2disp(1 / outputs[('depth', 0)], vmax=0.25, ind=vind)
        figd2 = tensor2disp(outputs[('shapeconstrain', 0)], vmax=2, ind=vind)
        figmask = tensor2disp(outputs['mask'], vmax=1, ind=vind)

        surfnorm = self.sfnormOptimizer.depth2norm(outputs[('depth', 0)], intrinsic=inputs['K'])
        surfnormref = self.sfnormOptimizer.ang2normal(torch.cat([inputs['angh'], inputs['angv']], dim=1), intrinsic=inputs['K'])
        fignorm = tensor2rgb((surfnorm + 1) / 2, ind=vind)
        fignormref = tensor2rgb((surfnormref + 1) / 2, ind=vind)

        figoview = np.concatenate([np.array(figd1), np.array(figmask), np.array(fignorm)], axis=0)
        figanghoview = np.concatenate([np.array(combined), np.array(figd2), np.array(fignormref)], axis=0)

        figcombined = np.concatenate([figoview, figanghoview], axis=1)
        self.writers[recoder].add_image('overview', (torch.from_numpy(figcombined).float() / 255).permute([2, 0, 1]), self.step)

    def log(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

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

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.train()
