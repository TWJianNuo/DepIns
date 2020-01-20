# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from options import MonodepthOptions

class InsTrainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales, predins = self.opt.predins)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True and not self.opt.noAug, img_ext=img_ext, load_detect=self.opt.predins,
            detect_path=self.opt.detect_path, load_seman = self.opt.loadSeman, load_pose=self.opt.loadPose,
            loadPredDepth = self.opt.loadPredDepth, predDepthPath = self.opt.predDepthPath)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size,  shuffle = not self.opt.noshuffle,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, load_detect=self.opt.predins, detect_path=self.opt.detect_path, load_seman = self.opt.loadSeman)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle = not self.opt.noshuffle,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        # Additional initialization
        if self.opt.predins:
            self.featureSize = 28
            self.boxLargeRat = 1.2

            self.models['insDecoder'] = networks.DynamicDecoder(resnet= self.models["encoder"], batch_size=self.opt.batch_size, imageHeight = self.opt.height).cuda()

            self.models['movDecoder'] = networks.MovementDecoder(batch_size=self.opt.batch_size, imageHeight=self.opt.height).cuda()

            self.layers = {}
            self.layers['rgbPooler'] = Pooler(self.opt.batch_size, shrinkScale = 1, imageHeight = self.opt.height, featureSize = self.featureSize)
        self.save_opts()

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
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time


            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                # if "depth_gt" in inputs:
                #     self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                # self.val()

            self.step += 1


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        outputs = {}
        losses = {}

        for key, ipt in inputs.items():
            if key not in ['entry_tag']:
                inputs[key] = ipt.to(self.device)

        features = self.models["encoder"](inputs["color_aug", 0, 0])

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs))

        if self.opt.predins and torch.sum(inputs['detect_label'] > 0) > 0:
            # Do the instance mask probability prediction
            extractInd = 3
            outputs.update(self.models['insDecoder'](features[extractInd], inputs['detect_label']))

            # Predict object-movement
            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    axisangle, translation = self.models['movDecoder'](outputs[('seq_features', 1)][0][extractInd], inputs['detect_label'], outputs['insFeature'])

                    outputs[("axisangle_obj", 0, f_i)] = axisangle
                    outputs[("translation_obj", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("obj_mov", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            # self.check_selfMov(inputs, outputs)
            # self.check_depthMap(inputs, outputs)
            losses.update(self.computeInsLoss(inputs, outputs))
        return outputs, losses

    def computeInsLoss(self, inputs, outputs):
        insDepth, _ = self.layers['rgbPooler'](inputs['predDepth'], inputs['detect_label'])
        insRgb, _ = self.layers['rgbPooler'](inputs[('color_aug', 0, 0)], inputs['detect_label'])
        gridx, gridy, valRois, valBInd = self.layers['rgbPooler'].get_grids(inputs['detect_label'])
        pixelLocs = torch.stack([gridx, gridy], dim=1)
        invcamK = inputs['invcamK'][valBInd, :, :]
        pts3d = backProjTo3d(pixelLocs, insDepth, invcamK)

        # Sample use self-movment
        cur_ind = 0
        mapping = {-1 : 1, 1 : 2}

        losses = {}
        totLoss = 0
        for f_i in self.opt.frame_ids[1:]:
            if f_i != "s":
                # Use self movement
                convertM = torch.inverse(inputs['bundledPoseM'][valBInd, mapping[f_i], :, :]) @ inputs['bundledPoseM'][valBInd, cur_ind, :, :]
                camKs = inputs['camK'][valBInd, :, :] @ convertM
                projected2d, projecDepth, selector = project_3dptsTo2dpts(pts3d=pts3d, camKs=camKs)
                sampledColor = sampleImgs(inputs[('color_aug', f_i, 0)][valBInd, :, :, :], projected2d)
                # tensor2flatrgb(sampledColor).show()
                # tensor2flatrgb(insRgb).show()

                # Use predicted Object movement
                camKs_obj = inputs['camK'][valBInd, :, :] @ convertM @ outputs[("obj_mov", 0, f_i)]
                projected2d_obj, projecDepth_obj, selector = project_3dptsTo2dpts(pts3d=pts3d, camKs=camKs_obj)
                sampledColor_obj = sampleImgs(inputs[('color_aug', f_i, 0)][valBInd, :, :, :], projected2d_obj)
                # tensor2flatrgb(sampledColor).show()
                # tensor2flatrgb(insRgb).show()

                ssimSelf = self.compute_reprojection_loss(sampledColor, insRgb)
                ssimObj = self.compute_reprojection_loss(sampledColor_obj, insRgb)
                maskLoss = torch.mean(ssimSelf * (1 - outputs['insProb']) + ssimObj * outputs['insProb'])

                totLoss = totLoss + maskLoss
                losses[('maskLoss', f_i)] = maskLoss
        totLoss = totLoss / 2
        losses['loss'] = totLoss
        # # Debug
        # pts3d = pts3d.view(pts3d.shape[0], 4, -1).permute([0,2,1])
        # projected2d, projecDepth, selector = project_3dptsTo2dpts(pts3d=pts3d, camKs=inputs['camK'][valBInd, :, :])
        # # Draw
        # drawInd = 0
        # selector = valBInd == drawInd
        # rgbim = np.array(tensor2rgb(inputs[('color_aug', 0, 0)], ind = drawInd))
        # pixel2dnp = projected2d[selector, :, :].cpu().numpy()
        #
        # plt.figure()
        # plt.imshow(rgbim)
        # for i in range(pixel2dnp.shape[0]):
        #     plt.scatter(pixel2dnp[i, 0, :], pixel2dnp[i, 1, :], s = 0.5, c = 'r')
        return losses


    def check_depthMap(self, inputs, outputs):
        xx, yy = np.meshgrid(range(self.opt.width), range(self.opt.height), indexing='xy')
        xx = torch.from_numpy(xx).unsqueeze(0).unsqueeze(0).expand([self.opt.batch_size, 1, -1, -1]).cuda().float()
        yy = torch.from_numpy(yy).unsqueeze(0).unsqueeze(0).expand([self.opt.batch_size, 1, -1, -1]).cuda().float()
        pixelLocs = torch.cat([xx, yy], dim=1)
        predDepth = inputs['predDepth']
        pts3d = backProjTo3d(pixelLocs, predDepth, inputs['invcamK'])


        # prepare to draw
        drawIndex = 0
        projected2d, projecDepth, selector = project_3dptsTo2dpts(pts3d=inputs['velo'], camKs=inputs['camK'])
        sampledColor = sampleImgs(inputs[('color', 0, 0)], projected2d)
        drawVelo = inputs['velo'][drawIndex, :, :].cpu().numpy()
        drawSelector = selector[drawIndex, 0, :].cpu().numpy() > 0
        drawX = drawVelo[drawSelector, 0]
        drawY = drawVelo[drawSelector, 1]
        drawZ = drawVelo[drawSelector, 2]
        drawColor = sampledColor[drawIndex, :, :].cpu().permute([1, 0]).numpy()[drawSelector, :]

        drawPts3dVal = pts3d[drawIndex, :, :, :].view(4, -1).cpu().permute([1, 0]).numpy()
        downsample_rat = 10

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=6., azim=-153)
        ax.dist = 1.7
        ax.scatter(drawX, drawY, drawZ, s = 1, c = drawColor)
        set_axes_equal(ax)
        ax.scatter(drawPts3dVal[::downsample_rat, 0], drawPts3dVal[::downsample_rat, 1],
                   drawPts3dVal[::downsample_rat, 2], s=0.7, c='r')
        plt.show()

    def check_selfMov(self, inputs, outputs):
        cur_ind = 0
        prev_ind = 1
        next_ind = 2
        projected2d, projecDepth, selector = project_3dptsTo2dpts(pts3d = inputs['velo'], camKs = inputs['camK'])
        sampledColor = sampleImgs(inputs[('color', 0, 0)], projected2d)
        sampledType = sampleImgs(inputs['semanLabel'].float(), projected2d, mode = 'nearest')

        # Mask out moving objects
        staticMask = sampledType < 10.9 # From semantic label 0 to 10 is staic categrory
        selector = selector * staticMask

        # Prepare to Draw
        velo_np = inputs['velo'].cpu().numpy()
        selector_np = selector.cpu().numpy() > 0
        sampledColor_np = sampledColor.cpu().numpy()

        # Draw current and prev Frame
        # convertM = torch.inverse(inputs['poseM'][cur_ind, :, :]) @ inputs['poseM'][prev_ind, :, :]
        # convertM = torch.inverse(convertM)
        convertM = torch.inverse(inputs['poseM'][prev_ind, :, :]) @ inputs['poseM'][cur_ind, :, :]
        curInPrev = convertM @ inputs['velo'][cur_ind, :, :].permute([1, 0])
        curInPrev = curInPrev.permute([1, 0])
        curInPrev_np = curInPrev.cpu().detach().numpy()

        draw_cur = velo_np[cur_ind, selector_np[cur_ind, 0, :], :]
        draw_curInPrev = curInPrev_np[selector_np[cur_ind, 0, :], :]
        sampledColor_curInPrev = sampledColor_np[cur_ind, :, selector_np[cur_ind, 0, :]]
        draw_prev = velo_np[prev_ind, selector_np[prev_ind, 0, :], :]
        sampledColor_prev = sampledColor_np[prev_ind, :, selector_np[prev_ind, 0, :]]
        downsample_rat = 1

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.view_init(elev=7., azim=-135)
        # ax.dist = 1.7
        # # ax.scatter(draw_prev[::downsample_rat, 0], draw_prev[::downsample_rat, 1], draw_prev[::downsample_rat, 2], s = 1, c = sampledColor_prev)
        # # ax.scatter(draw_curInPrev[::downsample_rat, 0], draw_curInPrev[::downsample_rat, 1], draw_curInPrev[::downsample_rat, 2], s=1, c=sampledColor_curInPrev)
        # ax.scatter(draw_prev[::downsample_rat, 0], draw_prev[::downsample_rat, 1], draw_prev[::downsample_rat, 2], s = 1, c = 'b')
        # ax.scatter(draw_curInPrev[::downsample_rat, 0], draw_curInPrev[::downsample_rat, 1], draw_curInPrev[::downsample_rat, 2], s=1, c='r')
        # ax.scatter(draw_cur[::downsample_rat, 0], draw_cur[::downsample_rat, 1], draw_cur[::downsample_rat, 2], s=1,
        #            c='g')
        # set_axes_equal(ax)
        # plt.show()
        #
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.view_init(elev=7., azim=-135)
        # ax.dist = 1.7
        # downsample_rat = 1
        # # ax.scatter(draw_prev[::downsample_rat, 0], draw_prev[::downsample_rat, 1], draw_prev[::downsample_rat, 2], s = 1, c = sampledColor_prev)
        # # ax.scatter(draw_curInPrev[::downsample_rat, 0], draw_curInPrev[::downsample_rat, 1], draw_curInPrev[::downsample_rat, 2], s=1, c=sampledColor_curInPrev)
        # ax.scatter(draw_prev[::downsample_rat, 0], draw_prev[::downsample_rat, 1], draw_prev[::downsample_rat, 2], s = 1, c = 'b')
        # ax.scatter(draw_cur[::downsample_rat, 0], draw_cur[::downsample_rat, 1], draw_cur[::downsample_rat, 2], s=1, c='r')
        # set_axes_equal(ax)
        # plt.show()

        # Draw in 2d
        fig = plt.figure()
        plt.scatter(draw_prev[::downsample_rat, 0], draw_prev[::downsample_rat, 1], s = 0.1, c = 'b')
        plt.scatter(draw_curInPrev[::downsample_rat, 0], draw_curInPrev[::downsample_rat, 1], s=0.1, c='r')
        plt.scatter(draw_cur[::downsample_rat, 0], draw_cur[::downsample_rat, 1], s=0.1, c='g')
        plt.savefig(os.path.join('/media/shengjie/other/Depins/Depins/visualization/gpsPoseVisualization', str(inputs['indicesRec'][cur_ind].cpu().numpy()) + '.png'), dpi = 200)
        plt.close(fig)

    def viewVelo(self, inputs, outputs):
        # For debug purpose

        projected2d, projecDepth, selector = project_3dptsTo2dpts(pts3d = inputs['velo'], camKs = inputs['camK'])
        sampledColor = sampleImgs(inputs[('color', 0, 0)], projected2d)

        # Visualization
        draw_index = 0
        velo_np = inputs['velo'].cpu().numpy()
        selector_np = selector.cpu().numpy() > 0
        sampledColor_np = sampledColor.cpu().numpy()
        projected2d_np = projected2d.cpu().numpy()
        projecDepth_np = projecDepth.cpu().numpy()


        drawx = projected2d_np[draw_index, 0, selector_np[draw_index, 0, :]]
        drawy = projected2d_np[draw_index, 1, selector_np[draw_index, 0, :]]
        drawDepth = projecDepth_np[draw_index, 0, selector_np[draw_index, 0, :]]

        dup_selector = filter_duplicated_depth([self.opt.height, self.opt.width], drawx, drawy, drawDepth)
        cm = plt.get_cmap('plasma')
        pts2dColor = cm(drawDepth[dup_selector] / 20)

        plt.figure()
        figrgb = tensor2rgb(inputs[('color', 0, 0)], ind = draw_index)
        plt.imshow(np.array(figrgb))
        plt.scatter(drawx[dup_selector], drawy[dup_selector], s = 0.5, c = pts2dColor)


        drawPts3d = velo_np[draw_index, selector_np[draw_index, 0, :], :]
        drawColors = sampledColor_np[draw_index, :, selector_np[draw_index, 0, :]]
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=6., azim=-153)
        ax.dist = 1.7
        downsample_rat = 1
        ax.scatter(drawPts3d[::downsample_rat, 0], drawPts3d[::downsample_rat, 1], drawPts3d[::downsample_rat, 2], s = 1, c = drawColors)
        set_axes_equal(ax)
        plt.show()


    def sample_depthAndBckprob(self, inputs, outputs):
        localoutputs = {}
        insBckProb, _ = self.layers['rgbPooler'](outputs['staticProb'], inputs['detect_label'])
        insDisp, _ = self.layers['rgbPooler'](outputs[('disp', 0)], inputs['detect_label'])

        # Debug purpose
        # insRgb, _ = self.layers['rgbPooler'](inputs[('color_aug', 0, 0)], inputs['detect_label'])
        # batchnum, chnum, l, l = insRgb.shape
        # insRgb = insRgb.permute(1,0,2,3).view(chnum, batchnum * l, l).unsqueeze(0)
        # tensor2rgb(insRgb, ind=0).show()

        localoutputs['insBckProb'] = insBckProb
        localoutputs['insDisp'] = insDisp
        return localoutputs


    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input

            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]

                    if self.opt.predins:
                        outputs[('seq_features', f_i)] = pose_inputs

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
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

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        # for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
        #     for s in self.opt.scales:
        #         for frame_id in self.opt.frame_ids:
        #             writer.add_image(
        #                 "color_{}_{}/{}".format(frame_id, s, j),
        #                 inputs[("color", frame_id, s)][j].data, self.step)
        #             if s == 0 and frame_id != 0:
        #                 writer.add_image(
        #                     "color_pred_{}_{}/{}".format(frame_id, s, j),
        #                     outputs[("color", frame_id, s)][j].data, self.step)
        #
        #         writer.add_image(
        #             "disp_{}/{}".format(s, j),
        #             normalize_image(outputs[("disp", s)][j]), self.step)
        #
        #         if self.opt.predictive_mask:
        #             for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
        #                 writer.add_image(
        #                     "predictive_mask_{}_{}/{}".format(frame_id, s, j),
        #                     outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
        #                     self.step)
        #
        #         elif not self.opt.disable_automasking:
        #             writer.add_image(
        #                 "automask_{}/{}".format(s, j),
        #                 outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        self.epoch = 1
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

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


options = MonodepthOptions()
opts = options.parse()
if __name__ == "__main__":
    trainer = InsTrainer(opts)
    trainer.train()
