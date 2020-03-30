from __future__ import absolute_import, division, print_function

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

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

import datasets_sfgan
import networks
from IPython import embed

import torchvision.transforms
from pointNet_network.pointNet_model import PointNetCls

from pointNet_network.ptn_discriminator import PtnD
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
def additional_opts_init(opts):
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

class Trainer_GAN:
    def __init__(self, options):
        self.opt = options
        self.opt = additional_opts_init(self.opt)
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
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # self.models['sfnD'] = PointNetCls(k = 1, feature_transform = False)
        self.models['sfnD'] = PtnD(opt=self.opt, k = 1, feature_transform=False)
        self.models['sfnD'].to(self.device)

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
        self.dataset = datasets_sfgan.SFGAN_Base_Dataset

        fpath = os.path.join(os.path.dirname(__file__), "..", "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        syn_train_filenames = readlines(fpath.format("syn_train"))
        syn_val_filenames = readlines(fpath.format("syn_val"))

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, syn_train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, opts = opts, is_train=True, load_seman=True)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle= not self.opt.noShuffle,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, syn_val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, opts = opts, is_train=False, load_seman=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        # for mode in ["train", "val"]:
        for mode in ["train"]:
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

        self.distillPtClound = DistillPtCloud(height=self.opt.height, width=self.opt.width, batch_size=self.opt.batch_size)
        self.distillPtClound.to(self.device)
        self.STEREO_SCALE_FACTOR = 5.4

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

        self.toTensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        self.sfnCom = ComputeSurfaceNormal(height=self.opt.height, width=self.opt.width, batch_size=self.opt.batch_size, minDepth=self.opt.min_depth, maxDepth=self.opt.max_depth).cuda()
        if self.opt.doVisualization:
            import matlab
            import matlab.engine
            self.eng = matlab.engine.start_matlab()

        # For visualization
        self.cap = 1000
        self.itcount = 0
        self.dp_real = np.zeros([self.cap, 1])
        self.dp_fake = np.zeros([self.cap, 1])
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

        with torch.no_grad():
            self.run_epoch()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Evaluate and visualize")
        self.set_eval()
        self.models['sfnD'].train()
        for batch_idx, inputs in enumerate(self.train_loader):
            outputs, losses = self.process_batch(inputs)

            for i in range(self.opt.batch_size):
                curind = np.mod(self.itcount, self.cap)
                self.dp_real[curind] = outputs['pred_real'][i].cpu().detach().numpy()
                self.dp_fake[curind] = outputs['pred_fake'][i].cpu().detach().numpy()
                self.itcount = self.itcount + 1

            self.log_img(mode="train", inputs=inputs, outputs=outputs)
            self.step += 1
            print(self.step)

        n_bins = 30
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        axs[0].hist(self.dp_real, bins=n_bins)
        axs[0].set_title('pred_real')
        axs[1].hist(self.dp_fake, bins=n_bins)
        axs[1].set_title('pred_fake')
        plt.savefig('/media/shengjie/other/Depins/Depins/visualization/pole_discriminator_selection/predspan.png')
        plt.close(fig)

    def process_batch(self, inputs, istrain = True):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if key not in ['entry_tag', 'syn_tag']:
                inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs, istrain = istrain)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

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
            outputs, losses = self.process_batch(inputs, istrain=False)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log_data(mode="val", losses=losses)
            self.log_img(mode = "val", inputs=inputs, outputs=outputs)
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
                # Save org Depth
                _, orgScale_depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
                outputs[("orgScale_depth", 0, scale)] = orgScale_depth

                depth = F.interpolate(
                    orgScale_depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            outputs[("depth", 0, scale)] = depth

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

    def check_depthMap(self, inputs, outputs, drawIndex):
        STEREO_SCALE_FACTOR = 5.4
        import matlab
        downsample_rat = 10

        xx, yy = np.meshgrid(range(self.opt.width), range(self.opt.height), indexing='xy')
        xx = torch.from_numpy(xx).unsqueeze(0).unsqueeze(0).expand([self.opt.batch_size, 1, -1, -1]).cuda().float()
        yy = torch.from_numpy(yy).unsqueeze(0).unsqueeze(0).expand([self.opt.batch_size, 1, -1, -1]).cuda().float()
        pixelLocs = torch.cat([xx, yy], dim=1)


        pts3d_rec = list()
        mono_sampledColor_rec = list()

        # Sync
        predDepth = inputs[('syn_depth', 0)]
        pts3d = backProjTo3d(pixelLocs, predDepth, inputs[('invcamK', 0)])
        mono_projected2d, _, _ = project_3dptsTo2dpts(pts3d=pts3d, camKs=inputs[('camK', 0)])
        mono_sampledColor = sampleImgs(inputs[('syn_rgb', 0)], mono_projected2d)
        pts3d_rec.append(pts3d)
        mono_sampledColor_rec.append(mono_sampledColor)

        # Real
        predDepth = outputs[("depth", 0, 0)] * STEREO_SCALE_FACTOR
        pts3d = backProjTo3d(pixelLocs, predDepth, inputs[('invcamK', 0)])
        mono_projected2d, _, _ = project_3dptsTo2dpts(pts3d=pts3d, camKs=inputs[('camK', 0)])
        mono_sampledColor = sampleImgs(inputs[('color', 0, 0)], mono_projected2d)
        pts3d_rec.append(pts3d)
        mono_sampledColor_rec.append(mono_sampledColor)

        # Velo
        projected2d, projecDepth, selector = project_3dptsTo2dpts(pts3d=inputs['velo'], camKs=inputs[('camK', 0)])
        sampledColor = sampleImgs(inputs[('color', 0, 0)], projected2d)
        drawVelo = inputs['velo'][drawIndex, :, :].cpu().numpy()
        drawSelector = selector[drawIndex, 0, :].cpu().numpy() > 0
        drawX_velo = drawVelo[drawSelector, 0]
        drawY_velo = drawVelo[drawSelector, 1]
        drawZ_velo = drawVelo[drawSelector, 2]
        drawColor_velo = sampledColor[drawIndex, :, :].cpu().permute([1, 0]).numpy()[drawSelector, :]

        drawX_velo = matlab.double(drawX_velo.tolist())
        drawY_velo = matlab.double(drawY_velo.tolist())
        drawZ_velo = matlab.double(drawZ_velo.tolist())
        drawColor_velo = matlab.double(drawColor_velo.tolist())

        # self.eng.eval('figure(\'visible\', \'off\')', nargout=0)
        self.eng.eval('figure()', nargout=0)
        for i in range(2):
            # In order of Sync-Real
            pts3d = pts3d_rec[i]
            mono_sampledColor = mono_sampledColor_rec[i]

            draw_mono_sampledColor = mono_sampledColor[drawIndex, :, :, :].detach().cpu().view(3, -1).permute([1,0]).numpy()[::downsample_rat, :]
            drawX_mono = pts3d[drawIndex, 0, :, :].detach().cpu().numpy().flatten()[::downsample_rat]
            drawY_mono = pts3d[drawIndex, 1, :, :].detach().cpu().numpy().flatten()[::downsample_rat]
            drawZ_mono = pts3d[drawIndex, 2, :, :].detach().cpu().numpy().flatten()[::downsample_rat]
            draw_mono_sampledColor = matlab.double(draw_mono_sampledColor.tolist())
            drawX_mono = matlab.double(drawX_mono.tolist())
            drawY_mono = matlab.double(drawY_mono.tolist())
            drawZ_mono = matlab.double(drawZ_mono.tolist())
            if i == 0:
                h = self.eng.scatter3(drawX_mono, drawY_mono, drawZ_mono, 5, 'r', 'filled', nargout=0)
            else:
                h = self.eng.scatter3(drawX_mono, drawY_mono, drawZ_mono, 5, draw_mono_sampledColor, 'filled', nargout = 0)
            self.eng.eval('hold on', nargout=0)

        # Draw velo
        h = self.eng.scatter3(drawX_velo, drawY_velo, drawZ_velo, 5, 'b', 'filled', nargout=0)

        self.eng.eval('axis equal', nargout = 0)
        xlim = matlab.double([0, 50])
        ylim = matlab.double([-10, 10])
        zlim = matlab.double([-5, 5])
        self.eng.xlim(xlim, nargout=0)
        self.eng.ylim(ylim, nargout=0)
        self.eng.zlim(zlim, nargout=0)
        self.eng.eval('view([-79 17])', nargout=0)
        self.eng.eval('camzoom(1.2)', nargout=0)
        self.eng.eval('grid off', nargout=0)
        # eng.eval('set(gca,\'YTickLabel\',[]);', nargout=0)
        # eng.eval('set(gca,\'XTickLabel\',[]);', nargout=0)
        # eng.eval('set(gca,\'ZTickLabel\',[]);', nargout=0)
        self.eng.eval('set(gca, \'XColor\', \'none\', \'YColor\', \'none\', \'ZColor\', \'none\')', nargout=0)

        folder, frame_index, _, _, _, _ = inputs['entry_tag'][drawIndex].split('\n')
        folder = folder.split(' ')[1].split('/')[1]
        frame_index = frame_index.split(' ')[1]
        sv_folder = os.path.join('/media/shengjie/other/Depins/Depins/visualization/vrKitti_unsuperDep_scale_visualizaiton', folder)
        os.makedirs(sv_folder, exist_ok = True)
        sv_path = os.path.join(sv_folder, frame_index + '.png')
        command = 'saveas(gcf, \'{}\')'.format(sv_path)
        self.eng.eval(command, nargout=0)
        self.eng.eval('close all', nargout=0)


    def compute_losses(self, inputs, outputs, istrain = True):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        ptCloud_syn, val_syn = self.distillPtClound(predDepth = inputs[("syn_depth", 0)], invcamK = inputs[('invcamK', 0)], semanticLabel = inputs['syn_semanLabel'], is_shrink = True)
        ptCloud_pred, val_pred = self.distillPtClound(predDepth = outputs[("depth", 0, 0)] * self.STEREO_SCALE_FACTOR, invcamK =inputs[('invcamK', 0)], semanticLabel = inputs['real_semanLabel'], is_shrink = False)
        # ptCloud_pred = ptCloud_pred + torch.Tensor([25, 0, 0]).unsqueeze(0).unsqueeze(2).expand([self.opt.batch_size,-1,10000]).cuda()

        outputs['pts_real'] = ptCloud_pred
        outputs['pts_realv'] = val_pred
        outputs['pts_syn'] = ptCloud_syn
        outputs['pts_synv'] = val_syn



        # Compute GAN Loss
        self.models['sfnD'].set_input(real=ptCloud_pred, realv=val_pred, syn=ptCloud_syn, synv=val_syn)
        pred_real, pred_fake = self.models['sfnD'].classification()
        losses["loss"] = torch.mean((pred_real + pred_fake))
        outputs['pred_real'] = pred_real
        outputs['pred_fake'] = pred_fake
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

    def log_data(self, mode, losses):
        writer = self.writers[mode]
        for l, v in losses.items():
            if v < 1e-3:
                continue
            else:
                writer.add_scalar("{}".format(l), v, self.step)

    def log_img(self, mode, inputs, outputs):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        # if self.eng is None:
        #     self.eng = matlab.engine.start_matlab()
        vis_root = '/media/shengjie/other/Depins/Depins/visualization/pole_discriminator_selection'
        real_root = os.path.join(vis_root, 'real')
        syn_root = os.path.join(vis_root, 'syn')
        os.makedirs(real_root, exist_ok=True)
        os.makedirs(syn_root, exist_ok=True)
        for j in range(min(2, self.opt.batch_size)):  # write a maxmimum of four images
            dns_rate = 10
            # fig = plt.figure('visible', 'off')
            fig, _ = plt.subplots()
            ax = Axes3D(fig)
            ax.view_init(elev=7., azim=-135)
            ax.dist = 1.7
            ax.scatter(outputs['pts_real'][j, 0, ::dns_rate].detach().cpu().numpy(),
                       outputs['pts_real'][j, 1, ::dns_rate].detach().cpu().numpy(),
                       outputs['pts_real'][j, 2, ::dns_rate].detach().cpu().numpy(), s=1, c='b')
            set_axes_equal(ax)
            ax.set_xlim(left=0, right=50)
            ax.set_ylim(bottom=-10, top=10)
            ax.set_zlim(bottom=-3, top=5)
            ax.view_init(elev=17, azim=-79)
            ax.dist = 10

            fig_name = inputs['entry_tag'][j].split('\n')[0].split('/')[1] + '_' + inputs['entry_tag'][j].split('\n')[1].split(' ')[1].zfill(10) + '.png'
            if outputs['pred_real'][j] > 0.5:
                plt.savefig(os.path.join(real_root, fig_name))
            else:
                plt.savefig(os.path.join(syn_root, fig_name))
            plt.close(fig)


            fig, _ = plt.subplots()
            ax = Axes3D(fig)
            ax.view_init(elev=7., azim=-135)
            ax.dist = 1.7
            ax.scatter(outputs['pts_syn'][j, 0, ::dns_rate].detach().cpu().numpy(),
                       outputs['pts_syn'][j, 1, ::dns_rate].detach().cpu().numpy(),
                       outputs['pts_syn'][j, 2, ::dns_rate].detach().cpu().numpy(), s=1, c='r')
            ax.set_xlim(left=0, right=50)
            ax.set_ylim(bottom=-10, top=10)
            ax.set_zlim(bottom=-3, top=5)
            ax.view_init(elev=17, azim=-79)
            ax.dist = 10
            fig_name = inputs['syn_tag'][j].split('\n')[0].split(' ')[1] + '_' + inputs['syn_tag'][j].split('\n')[1].split(' ')[1].zfill(10) + '.png'
            if outputs['pred_fake'][j] > 0.5:
                plt.savefig(os.path.join(real_root, fig_name))
            else:
                plt.savefig(os.path.join(syn_root, fig_name))
            plt.close(fig)



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
        save_folder = os.path.join(self.log_path, "models", "weights_step{}".format(self.step))
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

        # Save depth adam params
        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

        # Save SfnD adam params
        save_path = os.path.join(save_folder, "{}.pth".format("ptn_adam"))
        torch.save(self.models['sfnD'].optimizer_D.state_dict(), save_path)
        print("Model %s saved" % self.opt.model_name)
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


            optimizer_load_path = os.path.join(self.opt.load_weights_folder, "sfn_adam.pth")
            if os.path.isfile(optimizer_load_path):
                optimizer_dict = torch.load(optimizer_load_path)
                self.models['sfnD'].optimizer_D.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")


from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    script_name = os.path.basename(__file__)
    assert opts.script_name == script_name, print("Please Specify correct script name\n")
    trainer = Trainer_GAN(opts)
    trainer.train()