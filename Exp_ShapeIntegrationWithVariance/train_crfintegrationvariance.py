from __future__ import absolute_import, division, print_function

import os, sys
project_rootdir = os.getcwd()
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from Exp_ShapeIntegrationWithVariance.dataloader_kitti import KittiDataset

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
parser.add_argument("--predsemantics_path",     type=str,   default='None',             help="path to kitti semantics prediction file")
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
parser.add_argument("--banshuffle",             action="store_true")

# Integration options
parser.add_argument("--lam",                    type=float,   default=0.05)
parser.add_argument("--inttimes",               type=int,     default=1)
parser.add_argument("--clipvariance",           type=float,   default=5)
parser.add_argument("--maxrange",               type=float,   default=100)

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
        self.models["depth"] = MultiModalityDecoder(self.models["encoder"].num_ch_enc, modalities=['depth', 'variance'], nchannelout=[1, 1], additionalblocks=1)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        print("Training model named:\t", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\t", self.opt.log_dir)
        print("Training is using:\t", self.device)

        self.set_dataset()
        self.writers = SummaryWriter(os.path.join(self.log_path))

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\t  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items".format(self.train_num, self.val_num))

        if self.opt.load_weights_folder is not None:
            self.load_model()

        self.save_opts()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.minabsrel = 1e10
        self.maxa1 = -1e10

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()

        from integrationModule import CRFIntegrationModule
        self.crfintmodule = CRFIntegrationModule(clipvariance=self.opt.clipvariance, maxrange=self.opt.maxrange, lam=self.opt.lam)

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.getcwd(), "splits", self.opt.split, "{}_files.txt")
        test_fpath = os.path.join(os.getcwd(), "splits", self.opt.split, "test_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(test_fpath)

        train_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, train_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=True, predang_path=self.opt.predang_path,
            predsemantics_path=self.opt.predsemantics_path
        )

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.val_gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, predang_path=self.opt.predang_path,
            predsemantics_path=self.opt.predsemantics_path
        )

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=not self.opt.banshuffle,
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

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses['totLoss'].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            if self.step % 1000 == 0:
                self.record_img(inputs, outputs)

            if self.step % 100 == 0:
                self.log_time(batch_idx, duration, losses["totLoss"])

            if self.step % 1 == 0:
                self.log("train", losses)

            if self.step % 2000 == 0 and self.step > 1:
                self.val()

            self.step += 1

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

        losses['totLoss'] = losses['depthloss']

        return outputs, losses

    def depth_compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        depthloss = 0

        for scale in range(4):
            scaled_disp, pred_depth = disp_to_depth(outputs[('depth', scale)], min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
            pred_depth = F.interpolate(pred_depth, [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True)
            pred_depth = pred_depth * self.STEREO_SCALE_FACTOR
            outputs[('pred_depth', scale)] = pred_depth

            pred_variance = F.interpolate(outputs[('variance', scale)], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True) * (self.opt.maxrange + 1)
            outputs[('pred_variance', scale)] = pred_variance

        pred_ang = torch.cat([inputs['angh'], inputs['angv']], dim=1).contiguous()
        pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang).contiguous()
        normfromang = self.sfnormOptimizer.ang2normal(ang=pred_ang, intrinsic=inputs['K'])
        singularnorm = self.sfnormOptimizer.ang2edge(ang=pred_ang, intrinsic=inputs['K'])

        semanticspred = inputs['semanticspred'].int().contiguous()
        semanedgemask = torch.ones_like(semanticspred)

        # exclude the top area
        semanedgemask[:, :, 0:int(0.40810811 * self.opt.crph), :] = 0

        # exclude distant places
        semanedgemask = semanedgemask * (outputs[('pred_depth', 0)] < 30).int()

        # exclude up normal
        semanedgemask = semanedgemask * (normfromang[:, 1, :, :].unsqueeze(1) < 0.8).int()

        # exclup singular value of the normal
        semanedgemask = semanedgemask * (1 - singularnorm).int()

        outputs['semanedgemask'] = semanedgemask

        vallidarmask = (inputs['depthgt'] > 0).float()
        for scale in range(4):
            pred_depth_integrated = self.crfintmodule(pred_log, semanticspred, semanedgemask, outputs[('pred_variance', scale)], outputs[('pred_depth', scale)], times=self.opt.inttimes)
            outputs[('pred_depth_integrated', scale)] = pred_depth_integrated

            depthloss += torch.sum(torch.abs(outputs[('pred_depth_integrated', scale)] - inputs['depthgt']) * vallidarmask) / (torch.sum(vallidarmask) + 1)

        depthloss = depthloss / 4
        losses['depthloss'] = depthloss

        return losses

    def val(self):
        """Validate the model on a single minibatch
        """
        count = 0
        self.set_eval()
        errors = list()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                outputs = dict()
                encoder_feature = self.models['encoder'](inputs['color'])
                outputs.update(self.models['depth'](encoder_feature))

                self.depth_compute_losses(inputs, outputs)
                for i in range(inputs['color'].shape[0]):
                    gt_depth = inputs['depthgt'][i, 0, :, :].cpu().numpy()
                    gt_height, gt_width = gt_depth.shape
                    cur_pred_depth = outputs[('pred_depth_integrated', 0)][i:i+1,:,:,:].detach()
                    cur_pred_depth = F.interpolate(cur_pred_depth, [gt_height,gt_width], mode='bilinear', align_corners=True)
                    cur_pred_depth = cur_pred_depth[0,0,:,:].cpu().numpy()

                    mask = np.logical_and(gt_depth > self.MIN_DEPTH, gt_depth < self.MAX_DEPTH)
                    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                     0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

                    cur_pred_depth = cur_pred_depth[mask]
                    gt_depth = gt_depth[mask]

                    cur_pred_depth[cur_pred_depth < self.MIN_DEPTH] = self.MIN_DEPTH
                    cur_pred_depth[cur_pred_depth > self.MAX_DEPTH] = self.MAX_DEPTH

                    errors.append(compute_errors(gt_depth, cur_pred_depth))
                    count = count + 1

        mean_errors = np.array(errors).mean(0)

        if mean_errors[0] < self.minabsrel:
            self.minabsrel_perform = mean_errors
            self.minabsrel = mean_errors[0]
            self.save_model("best_absrel_models")
        if mean_errors[4] > self.maxa1:
            self.maxa1_perform = mean_errors
            self.maxa1 = mean_errors[4]
            self.save_model("best_a1_models")

        print("\nCurrent Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

        print("\nBest Absolute Relative Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*self.minabsrel_perform.tolist()) + "\\\\")

        print("\nBest A1 Relative Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*self.maxa1_perform.tolist()) + "\\\\")

        self.set_train()

    def log_time(self, batch_idx, duration, loss_tot):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}\nloss_tot: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss_tot, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def record_img(self, inputs, outputs, recoder='train'):
        vind = 0

        vlscolor = F.interpolate(inputs['color'], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True)
        figrgb = np.array(tensor2rgb(vlscolor, ind=vind))

        semanticsfig = tensor2semantic(inputs['semanticspred'], ind=0)

        ratio = 0.3
        combined = np.array(figrgb).astype(np.float) * (1 - ratio) + np.array(semanticsfig).astype(np.float) * ratio
        combined = combined.astype(np.uint8)

        figd1 = tensor2disp(1 / outputs[('pred_depth', 0)], vmax=0.25, ind=vind)
        figd2 = tensor2disp(1 / outputs[('pred_depth_integrated', 0)], vmax=0.25, ind=vind)
        figd3 = tensor2disp(1 / outputs[('pred_variance', 0)], vmax=1, ind=vind)
        figmask = tensor2disp(outputs['semanedgemask'], vmax=1, ind=vind)

        surfnorm = self.sfnormOptimizer.depth2norm(outputs[('pred_depth_integrated', 0)], intrinsic=inputs['K'])
        fignorm = tensor2rgb((surfnorm + 1) / 2, ind=vind)
        # surfnormref = self.sfnormOptimizer.ang2normal(torch.cat([inputs['angh'], inputs['angv']], dim=1), intrinsic=inputs['K'])
        # fignormref = tensor2rgb((surfnormref + 1) / 2, ind=vind)

        figoview = np.concatenate([np.array(figd1), np.array(figd2), np.array(figd3)], axis=0)
        figanghoview = np.concatenate([np.array(combined), np.array(fignorm), np.array(figmask)], axis=0)

        figcombined = np.concatenate([figoview, figanghoview], axis=1)
        self.writers.add_image('overview', (torch.from_numpy(figcombined).float() / 255).permute([2, 0, 1]), self.step)

    def log(self, mode, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, keyword):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "{}".format(keyword))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        print("save to %s" % save_folder)

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
