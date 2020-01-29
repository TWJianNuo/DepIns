from __future__ import absolute_import, division, print_function

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from options import MonodepthOptions
from datasets import create_dataset
from networks import create_model
from tensorboardX import SummaryWriter
import torch
import time
import os
from utils import *
from layers import *
from datasets import SFNormDataset
import copy



# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import os
import argparse
import pathlib


file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class SFNormOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--dataA_path",
                                 type=str,)
        self.parser.add_argument("--dataB_path",
                                 type=str,)
        self.parser.add_argument("--predDepth_path",
                                 type=str,)
        self.parser.add_argument("--gtDepth_path",
                                 type=str,)
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(str(pathlib.Path().absolute()), "tmp"))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 default="eigen_zhou")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)

        self.parser.add_argument("--serial_batches",
                                 action="store_true")
        self.parser.add_argument("--noAug",
                                 action="store_true")




        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        # SYSTEM options
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")

        # Cycle GAN additional
        self.parser.add_argument('--max_dataset_size',
                                 type=int,
                                 default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--direction',
                                 type=str,
                                 default='AtoB',
                                 help='AtoB or BtoA')
        self.parser.add_argument("--crop_height",
                                 type=int,
                                 help="cropped input image height",
                                 default=256)
        self.parser.add_argument("--crop_width",
                                 type=int,
                                 help="cropped input image width",
                                 default=256)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options


options = SFNormOptions()
opts = options.parse()

def init_opts(opts):
    opts.phase = 'train'
    opts.lr_policy = 'linear'
    opts.norm = 'instance'
    opts.init_type = 'normal'
    opts.netG = 'resnet_9blocks'
    opts.netD = 'basic'
    opts.gan_mode = 'lsgan'
    opts.load_epoch = 'latest' # which epoch to load? set to latest to use latest cached model

    opts.gpu_ids = [0]

    opts.lambda_identity = 0.5
    opts.lambda_A = 10.0
    opts.lambda_B = 10.0
    opts.input_nc = 1
    opts.output_nc = 1
    opts.ngf = 64
    opts.ndf = 64
    opts.init_gain = 0.02
    opts.n_layers_D = 3
    opts.pool_size = 50
    opts.beta1 = 0.5
    opts.load_iter = 0
    opts.epoch_count = 1
    opts.n_epochs = 100 # number of epochs with the initial learning rate
    opts.n_epochs_decay = 100 # number of epochs to linearly decay learning rate to zero
    opts.print_freq = 10
    opts.display_freq = 400
    opts.save_epoch_freq = 5
    opts.save_latest_freq = 5000

    opts.isTrain = True
    opts.verbose = False
    opts.no_dropout = True
    opts.continue_train = False
    opts.save_by_iter = False
    opts.mode_sfnorm = True
    return opts

def visualize_check(data, i):
    visualization_root = '/media/shengjie/other/Depins/Depins/visualization/sfNorm_input_validation'

    figARgb = tensor2rgb(data['A_rgb'], ind=0)
    figBRgb = tensor2rgb(data['B_rgb'], ind=0)
    figADep = tensor2disp(1 - data['A_depth'], vmax=1, ind=0)
    figBDep = tensor2disp(1 - data['B_depth'], vmax=1, ind=0)

    figup = np.concatenate([np.array(figARgb), np.array(figBRgb)], axis=1)
    figdown = np.concatenate([np.array(figADep), np.array(figBDep)], axis=1)
    figCombined = np.concatenate([figup, figdown], axis=0)

    if i < 100:
        pil.fromarray(figCombined).save(os.path.join(visualization_root, str(i).zfill(10) + '.png'))

def organize_visuals(to_visuals, rgb_A, rgb_B):
    to_visuals_organized = dict()

    comb1_sets = ['real_A', 'fake_A', 'rec_A']
    comb2_sets = ['real_B', 'fake_B', 'rec_B']
    comb3_sets = ['real_A_sfnorm', 'fake_A_sfnorm', 'rec_A_sfnorm']
    comb4_sets = ['real_B_sfnorm', 'fake_B_sfnorm', 'rec_B_sfnorm']
    comb5_sets = ['idt_A', 'idt_B']

    comb1 = list()
    for img_name in comb1_sets:
        comb1.append(1 - (to_visuals[img_name] + 1) / 2)
    comb1 = torch.cat(comb1, dim=2)
    comb1 = tensor2disp(comb1, vmax=1, ind=0)

    comb2 = list()
    for img_name in comb2_sets:
        comb2.append(1 - (to_visuals[img_name] + 1) / 2)
    comb2 = torch.cat(comb2, dim=2)
    comb2 = tensor2disp(comb2, vmax=1, ind=0)

    comb3 = list()
    for img_name in comb3_sets:
        comb3.append((to_visuals[img_name] + 1) / 2)
    comb3 = torch.cat(comb3, dim=2)
    comb3 = tensor2rgb(comb3, ind=0)

    comb4 = list()
    for img_name in comb4_sets:
        comb4.append((to_visuals[img_name] + 1) / 2)
    comb4 = torch.cat(comb4, dim=2)
    comb4 = tensor2rgb(comb4, ind=0)

    comb5 = list()
    for img_name in comb5_sets:
        comb5.append(1 - (to_visuals[img_name] + 1) / 2)
    comb5 = torch.cat(comb5, dim=2)
    comb5 = tensor2disp(comb5, vmax=1, ind=0)

    comb6 = torch.cat([rgb_A, rgb_B], dim=2)
    comb6 = tensor2rgb(comb6, ind=0)

    to_visuals_organized['AB_depth'] = torch.from_numpy(np.concatenate([np.array(comb1), np.array(comb2)], axis=1)).permute([2,0,1]).float()/255
    to_visuals_organized['AB_sfnorm'] = torch.from_numpy(np.concatenate([np.array(comb3), np.array(comb4)], axis=1)).permute([2,0,1]).float()/255
    to_visuals_organized['idt'] = torch.from_numpy(np.array(comb5)).permute([2,0,1]).float()/255
    to_visuals_organized['rgb'] = torch.from_numpy(np.array(comb6)).permute([2,0,1]).float()/255

    return to_visuals_organized

def compute_depth_losses(depth_pred, depth_gt):
    """Compute depth metrics, to allow monitoring during training

    This isn't particularly accurate as it averages over the entire batch,
    so is only used to give an indication of validation performance
    """
    if depth_gt.shape[2] != 375 or depth_gt.shape[3] != 1242:
        print("Wrong shape : (%d, %d)" % (depth_gt.shape[2], depth_gt.shape[3]))

    losses = dict()
    depth_pred = (depth_pred + 1) / 2 * (80 - 1e-3) + 1e-3
    depth_pred = torch.clamp(F.interpolate(
        depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_pred = depth_pred.detach()

    mask = depth_gt > 0

    # garg/eigen crop
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]
    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

    depth_errors = compute_depth_errors(depth_gt, depth_pred)
    metric_name = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
    for i, metric in enumerate(depth_errors):
        losses[metric_name[i]] = np.array(depth_errors[i].cpu())

    return losses

if __name__ == "__main__":

    opts = init_opts(opts)
    dataset = SFNormDataset(opts)
    dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=not opts.serial_batches,
        num_workers=int(opts.num_workers))

    # opts_test = copy.deepcopy(opts)
    # opts_test.phase = 'test'
    # dataset_test = SFNormDataset(opts_test)
    # dataset_test = torch.utils.data.DataLoader(
    #     dataset_test,
    #     batch_size=opts.batch_size,
    #     shuffle=not opts.serial_batches,
    #     num_workers=int(opts.num_workers))

    model = create_model(opts)  # create a model given opt.model and other options
    model.setup(opts)  # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    sum_writers = SummaryWriter(os.path.join(opts.log_dir, opts.model_name, 'train'))
    # sum_writers_val = SummaryWriter(os.path.join(opts.log_dir, opts.model_name, 'val'))

    os.makedirs(os.path.join(opts.log_dir, opts.model_name, 'model'), exist_ok=True)

    train_start_time = time.time()

    for epoch in range(opts.epoch_count, opts.n_epochs + opts.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += opts.batch_size
            epoch_iter += opts.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opts.print_freq == 0:
                losses = model.get_current_losses()
                for l, v in losses.items():
                    sum_writers.add_scalar(l, v, total_iters)
                to_visuals = model.get_current_visuals()

                if torch.sum(torch.sum(data['gt_depth'], [0,2,3]) == 0) == 0:

                    eval_metrics = compute_depth_losses(depth_pred = to_visuals['fake_B'], depth_gt = data['gt_depth'].cuda())
                    eval_metrics_bs = compute_depth_losses(depth_pred = to_visuals['real_A'], depth_gt = data['gt_depth'].cuda())

                    for l, v in eval_metrics.items():
                        sum_writers.add_scalar('train/' + l, v, total_iters)

                    for l, v in eval_metrics_bs.items():
                        sum_writers.add_scalar('train_bs/' + l, v, total_iters)

            if total_iters % opts.display_freq == 0:   # display images on visdom and save images to a HTML file
                to_visuals = model.get_current_visuals()
                to_visuals = organize_visuals(to_visuals, rgb_A=data['A_rgb'], rgb_B=data['B_rgb'])
                for l, v in to_visuals.items():
                    sum_writers.add_image(l, v, total_iters)
                train_time = time.time() - train_start_time
                print("Epoch %d, time left %f hours" % (epoch, train_time / total_iters * dataset.__len__() * (opts.n_epochs + opts.n_epochs_decay) / 60 / 60))

            if total_iters % opts.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opts.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        if epoch % opts.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()                     # update learning rates at the end of every epoch.