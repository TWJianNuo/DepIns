from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from torch.utils.data import DataLoader

# Resolve Tensorbard Confliction across pytorch version
import torch
import warnings
version_num = torch.__version__
version_num = ''.join(i for i in version_num if i.isdigit())
version_num = int(version_num.ljust(10, '0'))
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    if version_num > 1100000000:
        from torch.utils.tensorboard import SummaryWriter
    else:
        from tensorboardX import SummaryWriter

from Exp_PreSIL.dataloader_PreSIL import PreSILDataset

import networks

import time
import json

from layers import *
from networks import *

import argparse

default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",              type=str,                               help="path to dataset")
parser.add_argument("--model_name",             type=str,                               help="name of the model")
parser.add_argument("--split",                  type=str,                               help="train/val split to use")
parser.add_argument("--log_dir",                type=str,   default=default_logpath,    help="path to log file")
parser.add_argument("--num_layers",             type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                 type=int,   default=192,                help="input image height")
parser.add_argument("--width",                  type=int,   default=640,                help="input image width")
parser.add_argument("--min_depth",              type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",              type=float, default=100.0,              help="maximum depth")
parser.add_argument("--print_freq",             type=int,   default=50)
parser.add_argument("--val_frequency",          type=int,   default=10)
parser.add_argument("--as_lidar",                action="store_true")

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
        self.models["depth"] = DepthDecoder(self.models["encoder"].num_ch_enc, num_output_channels=3)
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

        self.save_opts()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.best_abs = 1e10

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.height, width=self.opt.width, batch_size=self.opt.batch_size).cuda()

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "{}_files.txt")
        test_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "val_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(test_fpath)

        train_dataset = PreSILDataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            is_train=True, as_lidar=self.opt.as_lidar
        )

        val_dataset = PreSILDataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            is_train=False, as_lidar=self.opt.as_lidar
        )

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=True,
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
            self.save_model("weight_{}".format(self.epoch))

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

            if self.step % 100 == 0:
                self.log("train", inputs, outputs, losses, writeImage=False)

            if self.step % 2000 == 0 and self.step > 1999:
                self.val()

            self.step += 1

    def process_batch(self, inputs, isval = False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if not key == 'tag':
                inputs[key] = ipt.to(self.device)
        inputs["normgt"] = self.sfnormOptimizer.depth2norm(inputs["depthgt"], inputs["K"])

        outputs = dict()
        losses = dict()

        outputs.update(self.models['depth'](self.models['encoder'](inputs['color_aug'])))

        # Normal Branch
        # losses.update(self.theta_compute_losses(inputs, outputs))

        # Depth Branch
        losses.update(self.norm_compute_losses(inputs, outputs))

        # Constrain Branch
        # losses.update(self.constrain_compute_losses(inputs, outputs))

        losses['totLoss'] = losses['norm_l1loss']
        return outputs, losses

    def constrain_compute_losses(self, inputs, outputs):
        losses = dict()
        l1constrain = 0
        htheta_pred_detached = outputs['htheta_pred'].detach()
        vtheta_pred_detached = outputs['vtheta_pred'].detach()

        # htheta, vtheta = self.presil_localthetadesp.get_theta(inputs['pSIL_depth'])
        # self.presil_localthetadesp.depth_localgeom_consistency(inputs['pSIL_depth'], htheta, vtheta)
        for i in range(len(self.opt.scales)):
            scaledDepth = F.interpolate(outputs[('depth', 0, i)] * self.STEREO_SCALE_FACTOR, [self.opt.height, self.opt.width], mode='bilinear', align_corners=True)
            l1constrain = l1constrain + self.localthetadespKitti_scaled.depth_localgeom_consistency(scaledDepth, htheta_pred_detached, vtheta_pred_detached, mask=self.thetalossmap)
        l1constrain = l1constrain / len(self.opt.scales)
        losses['norm_l1loss'] = l1constrain
        return losses

    def theta_compute_losses(self, inputs, outputs):
        losses = dict()
        ltheta = 0
        sclLoss = 0
        for i in range(len(self.opt.scales)):
            pred_theta = outputs[('disp', i)][:,0:2,:,:]
            pred_theta = F.interpolate(pred_theta, [self.kittih, self.kittiw], mode='bilinear', align_corners=True)
            pred_theta = pred_theta * float(np.pi) * 2
            htheta_pred = pred_theta[:, 0:1, :, :]
            vtheta_pred = pred_theta[:, 1:2, :, :]

            inbl, outbl, scl = self.localthetadespKitti.inplacePath_loss(depthmap=inputs['depthgt'], htheta=htheta_pred, vtheta=vtheta_pred, balancew = self.opt.balancew)

            if i == 0:
                outputs['htheta_pred'] = outputs[('disp', i)][:, 0:1, :, :] * float(np.pi) * 2
                outputs['vtheta_pred'] = outputs[('disp', i)][:, 1:2, :, :] * float(np.pi) * 2

            ltheta = ltheta + inbl + outbl / 10
            sclLoss = sclLoss + scl

        ltheta = ltheta / 4
        sclLoss = sclLoss / 4

        losses['ltheta'] = ltheta
        losses['sclLoss'] = sclLoss

        return losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        toterr = 0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                normgt = self.sfnormOptimizer.depth2norm(inputs["depthgt"], inputs["K"])

                outputs = self.models['depth'](self.models['encoder'](inputs["color"]))

                prednorm = (outputs[('disp', 0)] - 0.5) * 2
                prednorm = prednorm / torch.norm(prednorm, dim=1, keepdim=True).expand([-1, 3, -1, -1])

                toterr = torch.mean(1 - torch.sum(prednorm * normgt, dim=1, keepdim=True))
            del inputs, outputs
        mean_err = toterr / self.val_loader.__len__()

        if mean_err < self.best_abs:
            self.best_abs = mean_err
            self.save_model("best_abs_models")

        print("\nCurrent Performance: %f" % mean_err)
        print("\nBest Performance: %f" % self.best_abs)
        self.set_train()

    def norm_compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        l1loss = 0
        for scale in range(4):
            pred_norm = (outputs[('disp', scale)] - 0.5) * 2
            pred_htheta = F.interpolate(pred_norm, [self.opt.height, self.opt.width], mode='bilinear', align_corners=True)
            l1loss = l1loss + torch.mean(torch.abs(pred_htheta - inputs['normgt']))

        l1loss = l1loss / 4
        losses['norm_l1loss'] = l1loss

        return losses

    def log_time(self, batch_idx, duration, loss_tot):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}\nloss_tot: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss_tot, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def record_img(self, inputs, outputs):
        vind = 0

        figrgb = tensor2rgb(inputs['color'], ind=vind)

        fignormc1gt = tensor2disp((inputs['normgt'][:, 0:1, :, :] + 1) / 2, vmax=1, ind=0)
        fignormc1pred = tensor2disp(outputs[('disp', 0)][:, 0:1, :, :], vmax=1, ind=0)

        fignormc2gt = tensor2disp((inputs['normgt'][:, 1:2, :, :] + 1) / 2, vmax=1, ind=0)
        fignormc2pred = tensor2disp(outputs[('disp', 0)][:, 1:2, :, :], vmax=1, ind=0)

        fignormc3gt = tensor2disp((inputs['normgt'][:, 2:3, :, :] + 1) / 2, vmax=1, ind=0)
        fignormc3pred = tensor2disp(outputs[('disp', 0)][:, 2:3, :, :], vmax=1, ind=0)

        fignormgt = tensor2rgb((inputs['normgt'] + 1) / 2, ind=0)
        prednorm = (outputs[('disp', 0)] - 0.5) * 2
        prednorm = prednorm / torch.norm(prednorm, dim=1, keepdim=True).expand([-1, 3, -1, -1])
        fignormpred = tensor2rgb((prednorm + 1) / 2, ind=0)

        fignormc1view = np.concatenate([np.array(figrgb), np.array(fignormc1gt), np.array(fignormc1pred)], axis=0)
        fignormc2view = np.concatenate([np.array(figrgb), np.array(fignormc2gt), np.array(fignormc2pred)], axis=0)
        fignormc3view = np.concatenate([np.array(figrgb), np.array(fignormc3gt), np.array(fignormc3pred)], axis=0)
        fignormview = np.concatenate([np.array(figrgb), np.array(fignormgt), np.array(fignormpred)], axis=0)

        self.writers['train'].add_image('normc1view', (torch.from_numpy(fignormc1view).float() / 255).permute([2, 0, 1]), self.step)
        self.writers['train'].add_image('normc2view', (torch.from_numpy(fignormc2view).float() / 255).permute([2, 0, 1]), self.step)
        self.writers['train'].add_image('normc3view', (torch.from_numpy(fignormc3view).float() / 255).permute([2, 0, 1]), self.step)
        self.writers['train'].add_image('normview', (torch.from_numpy(fignormview).float() / 255).permute([2, 0, 1]), self.step)


    def log(self, mode, inputs, outputs, losses, writeImage=False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
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
        # save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.step))
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

        for n in self.opt.models_to_load:
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
