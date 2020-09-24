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
parser.add_argument("--as_lidar",               action="store_true")

# OPTIMIZATION options
parser.add_argument("--batch_size",             type=int,   default=12,                 help="batch size")
parser.add_argument("--learning_rate",          type=float, default=1e-4,               help="learning rate")
parser.add_argument("--num_epochs",             type=int,   default=20,                 help="number of epochs")
parser.add_argument("--scheduler_step_size",    type=int,   default=15,                 help="step size of the scheduler")
parser.add_argument("--load_weights_folder",    type=str,   default=None,               help="name of models to load")
parser.add_argument("--num_workers",            type=int,   default=6,                  help="number of dataloader workers")
parser.add_argument("--isgt",                   type=float, default=1e-4)
parser.add_argument("--gradconstrainw",         type=float, default=1e-4)
parser.add_argument("--useselfdepth",           action="store_true")


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
        self.models["depth"] = DepthDecoder(self.models["encoder"].num_ch_enc)
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

        self.best_absrel = 1e10
        self.best_a1 = -1e10

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
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=False)

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

            if self.step % 1 == 0:
                self.log("train", inputs, outputs, losses, writeImage=False)

            if self.step % 2000 == 0 and self.step > 1999:
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if not key == 'tag':
                inputs[key] = ipt.to(self.device)

        inputs['anggt'] = self.sfnormOptimizer.depth2ang(inputs['depthgt'], inputs['K'])

        depthtoopt = torch.clone(inputs['depthgt'])
        depthtoopt[:, :, ::2] = 0
        depthtoopt.requires_grad = True
        optimizer = optim.Adam([depthtoopt], lr=1e-3)
        for i in range(100):
            a = 1

        outputs = dict()
        losses = dict()

        outputs.update(self.models['depth'](self.models['encoder'](inputs['color_aug'])))

        # Depth Branch
        losses.update(self.depth_compute_losses(inputs, outputs))

        # Constrain Branch
        losses.update(self.constrain_compute_losses(inputs, outputs))

        losses['totLoss'] = losses['depth_l1loss'] * 1e-4 + losses['l1constrain'] * self.opt.gradconstrainw
        losses['totLoss'] = losses['l1constrain']
        return outputs, losses

    def constrain_compute_losses(self, inputs, outputs):
        losses = dict()
        l1constrain = 0

        for scale in range(4):
            _, pred_depth = disp_to_depth(outputs[('disp', scale)], min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
            pred_depth = F.interpolate(pred_depth, [self.opt.height, self.opt.width], mode='bilinear', align_corners=True)
            if self.opt.useselfdepth:
                depthMap_gradx_est, depthMap_grady_est = self.sfnormOptimizer.ang2grad(inputs['anggt'], inputs['K'], pred_depth)
            else:
                depthMap_gradx_est, depthMap_grady_est = self.sfnormOptimizer.ang2grad(inputs['anggt'], inputs['K'], inputs['depthgt'])
            depthMap_gradx_num, depthMap_grady_num = self.sfnormOptimizer.get_depth_numgrad(pred_depth, issharp=False)
            l1constrain = l1constrain + (torch.mean(torch.abs(depthMap_gradx_est - depthMap_gradx_num)) + torch.mean(torch.abs(depthMap_grady_est - depthMap_grady_num))) / 2
            if scale == 0:
                outputs['depthMap_gradx_est'] = depthMap_gradx_est
                outputs['depthMap_grady_est'] = depthMap_grady_est
                outputs['depthMap_gradx_num'] = depthMap_gradx_num
                outputs['depthMap_grady_num'] = depthMap_grady_num
        l1constrain = l1constrain / 4
        losses['l1constrain'] = l1constrain
        return losses

    def depth_compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        l1loss = 0
        selector_depth = (inputs['depthgt'] > 0).float()
        for scale in range(4):
            _, pred_depth = disp_to_depth(outputs[('disp', scale)], min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
            pred_depth = F.interpolate(pred_depth, [self.opt.height, self.opt.width], mode='bilinear', align_corners=True)
            l1loss = l1loss + torch.sum(torch.abs(pred_depth - inputs['depthgt']) * selector_depth) / (torch.sum(selector_depth) + 1)
            if scale == 0:
                outputs['pred_depth'] = pred_depth
        l1loss = l1loss / 4
        losses['depth_l1loss'] = l1loss

        return losses

    def val(self):
        """Validate the model on a single minibatch
        """
        count = 0
        self.set_eval()
        errors = list()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                input_color = inputs["color"].cuda()
                outputs = self.models['depth'](self.models['encoder'](input_color))
                _, pred_depth = disp_to_depth(outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                for i in range(input_color.shape[0]):
                    gt_depth = inputs["depthgt"][i, 0, :, :].cpu().numpy()
                    cur_pred_depth = pred_depth[i:i+1, :, :, :]
                    cur_pred_depth = cur_pred_depth[0, 0, :, :].cpu().numpy()

                    mask = np.logical_and(gt_depth > self.MIN_DEPTH, gt_depth < self.MAX_DEPTH)

                    cur_pred_depth = cur_pred_depth[mask]
                    gt_depth = gt_depth[mask]

                    cur_pred_depth[cur_pred_depth < self.MIN_DEPTH] = self.MIN_DEPTH
                    cur_pred_depth[cur_pred_depth > self.MAX_DEPTH] = self.MAX_DEPTH

                    errors.append(compute_errors(gt_depth, cur_pred_depth))
                    count = count + 1
            del inputs, outputs
        mean_errors = np.array(errors).mean(0)

        if mean_errors[0] < self.best_absrel:
            self.best_absrel_perform = mean_errors
            self.best_absrel = mean_errors[0]
            self.save_model("best_absrel_models")
        if mean_errors[4] > self.best_a1:
            self.best_a1_perform = mean_errors
            self.best_a1 = mean_errors[4]
            self.save_model("best_a1_models")

        print("\nCurrent Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

        print("\nBest Absolute Relative Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*self.best_absrel_perform.tolist()) + "\\\\")

        print("\nBest A1 Relative Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*self.best_a1_perform.tolist()) + "\\\\")

        self.set_train()

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
        figdisp = tensor2disp(1 / outputs['pred_depth'], vmax=0.35, ind=0)

        depthgt = 1 / inputs['depthgt']
        depthgt[inputs['depthgt'] == 0] = 0
        figgt = tensor2disp(depthgt, vmax=0.35, ind=0)
        figcombined = np.concatenate([np.array(figrgb), np.array(figdisp), np.array(figgt)], axis=0)

        gradx_gt, grady_gt = self.sfnormOptimizer.get_depth_numgrad(inputs['depthgt'], issharp=True)

        figgradx_gt = tensor2grad(gradx_gt, pos_bar=0.08, neg_bar=-0.08)
        figgradx_num = tensor2grad(outputs['depthMap_gradx_num'], pos_bar=0.08, neg_bar=-0.08)
        figgradx_est = tensor2grad(outputs['depthMap_gradx_est'], pos_bar=0.08, neg_bar=-0.08)
        figgradx = np.concatenate([np.array(figgradx_gt), np.array(figgradx_num), np.array(figgradx_est)], axis=0)

        figgrady_gt = tensor2grad(grady_gt, pos_bar=0.08, neg_bar=-0.08)
        figgrady_num = tensor2grad(outputs['depthMap_grady_num'], pos_bar=0.08, neg_bar=-0.08)
        figgrady_est = tensor2grad(outputs['depthMap_grady_est'], pos_bar=0.08, neg_bar=-0.08)
        figgrady = np.concatenate([np.array(figgrady_gt), np.array(figgrady_num), np.array(figgrady_est)], axis=0)

        figoview = np.concatenate([np.array(figgradx), np.array(figgrady), np.array(figcombined)], axis=1)
        self.writers['train'].add_image('overview', (torch.from_numpy(figoview).float() / 255).permute([2, 0, 1]), self.step)

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
