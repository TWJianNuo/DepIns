from __future__ import absolute_import, division, print_function

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from options import MonodepthOptions
import warnings

import torch.optim as optim
from torch.utils.data import DataLoader

import torch
version_num = torch.__version__
version_num = ''.join(i for i in version_num if i.isdigit())
version_num = int(version_num.ljust(10, '0'))
if version_num > 1100000000:
    from torch.utils.tensorboard import SummaryWriter
else:
    from tensorboardX import SummaryWriter

from layers import *

import datasets
import networks
from Exp_jointModel_localgeomAndDepth import ResnetEncoderArbChannels

import time
import json


warnings.filterwarnings("ignore")
options = MonodepthOptions()
opts = options.parse()

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(0)

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
        self.STEREO_SCALE_FACTOR = 5.4

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)


        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        self.models["encoder"] = ResnetEncoderArbChannels(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_channels=5)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales, num_output_channels=3)
        self.models["depth"].to(self.device)

        self.parameters_to_train += list(self.models["depth"].parameters())
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.set_dataset()
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.set_layers()
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            self.train_num, self.val_num))

        if self.opt.load_weights_folder is not None:
            self.load_model()
        self.save_opts()

        self.unitFK = 0.58
        self.kittiw = 1242
        self.kittih = 375
        intrinsicKitti = np.array([
            [0.58 * self.kittiw, 0, 0.5 * self.kittiw],
            [0, 1.92 * self.kittih, 0.5 * self.kittih],
            [0, 0, 1]], dtype=np.float32)
        self.localthetadespKitti = LocalThetaDesp(height=self.kittih, width=self.kittiw,batch_size=self.opt.batch_size, intrinsic=intrinsicKitti).cuda()


        intrinsicKitti_scaled = np.array([
            [0.58 * self.opt.width, 0, 0.5 * self.opt.width],
            [0, 1.92 * self.opt.height, 0.5 * self.opt.height],
            [0, 0, 1]], dtype=np.float32)
        self.localthetadespKitti_scaled = LocalThetaDesp(height=self.opt.height, width=self.opt.width,batch_size=self.opt.batch_size, intrinsic=intrinsicKitti_scaled).cuda()

        self.thetalossmap = torch.zeros([1, 1, self.opt.height, self.opt.width]).expand([self.opt.batch_size, -1, -1, -1]).cuda()
        self.thetalossmap[:,:,110::,:] = 1

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80
        self.minabsrel = 1e10
        self.maxa1 = -1e10
    def set_layers(self):
        """properly handle layer initialization under multiple dataset situation
        """
        self.backproject_depth = {}
        self.project_3d = {}
        self.selfOccluMask = SelfOccluMask().cuda()

        height = self.opt.height
        width = self.opt.width
        for n, scale in enumerate(self.opt.scales):
            h = height // (2 ** scale)
            w = width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "{}_files.txt")
        test_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "eigen", "test_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(test_fpath)

        gt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", "eigen",  "gt_depths.npz")
        self.gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

        train_dataset = datasets.KITTIRAWDataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=not self.opt.no_aug, load_seman = False,
            kitti_gt_path = self.opt.kitti_gt_path, theta_gt_path=self.opt.theta_gt_path, predDepth_path = self.opt.predDepth_path
        )

        val_dataset = datasets.KITTIRAWDataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, load_seman = False, theta_gt_path=self.opt.theta_gt_path, predDepth_path = self.opt.predDepth_path
        )

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=not self.opt.no_shuffle,
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

            if self.step % 500 == 0:
                self.record_img(inputs, outputs)

            if self.step % 100 == 0:
                self.log_time(batch_idx, duration, losses["totLoss"])

            if self.step % 2 == 0:
                self.log("train", inputs, outputs, losses, writeImage=False)

            if self.step % 2000 == 0 and self.step > 1999:
                self.val()

            self.step += 1

    def depth2activation(self, depth):
        min_disp = 1 / self.opt.max_depth
        max_disp = 1 / self.opt.min_depth
        activation = depth / self.STEREO_SCALE_FACTOR
        activation = 1 / activation
        activation = (activation - min_disp) / (max_disp - min_disp)
        return activation

    def process_batch(self, inputs, isval = False):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if not (key == 'entry_tag' or key == 'syn_tag'):
                inputs[key] = ipt.to(self.device)

        outputs = dict()
        losses = dict()

        normedThetah = inputs['htheta'] / 2 / np.pi
        normedThetav = inputs['vtheta'] / 2 / np.pi
        cattedInput = torch.cat([inputs[('color_aug', 0, 0)], normedThetah, normedThetav], dim=1).cuda()
        outputs.update(self.models['depth'](self.models['encoder'](cattedInput)))

        # Depth Branch
        self.generate_images_pred(inputs, outputs)
        losses.update(self.depth_compute_losses(inputs, outputs))

        # Constrain Branch
        losses.update(self.constrain_compute_losses(inputs, outputs))

        losses['totLoss'] = losses['l1loss'] * self.opt.l1lossScale + \
                            losses['pholoss'] * self.opt.pholossScale + \
                            losses['l1constrain'] * self.opt.l1constrainScale

        return outputs, losses

    def constrain_compute_losses(self, inputs, outputs):
        losses = dict()
        l1constrain = 0
        htheta_pred_detached = inputs['htheta'].detach()
        vtheta_pred_detached = inputs['vtheta'].detach()

        for i in range(len(self.opt.scales)):
            scaledDepth = F.interpolate(outputs[('depth', 0, i)] * self.STEREO_SCALE_FACTOR, [self.opt.height, self.opt.width], mode='bilinear', align_corners=True)
            curconstrain, derivx, num_grad, rgb_gradw = self.localthetadespKitti_scaled.depth_localgeom_consistency(scaledDepth, htheta_pred_detached, vtheta_pred_detached, rgb=inputs[('color', 0, 0)], isdebias=self.opt.isdebias)
            l1constrain = l1constrain + curconstrain

            if i == 0:
                outputs['derivx'] = derivx
                outputs['num_grad'] = num_grad
                outputs['gradweights'] = rgb_gradw

        l1constrain = l1constrain / len(self.opt.scales)
        losses['l1constrain'] = l1constrain
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

            hloss, vloss, scl = self.localthetadespKitti.cleaned_path_loss(depthmap=inputs['depth_gt'], htheta=htheta_pred, vtheta=vtheta_pred)
            if i == 0:
                outputs['htheta_pred'] = outputs[('disp', i)][:, 0:1, :, :] * float(np.pi) * 2
                outputs['vtheta_pred'] = outputs[('disp', i)][:, 1:2, :, :] * float(np.pi) * 2
            ltheta = ltheta + (hloss * 10 + vloss) / 2
            sclLoss = sclLoss + scl
        ltheta = ltheta / 4
        sclLoss = sclLoss / 4

        losses['ltheta'] = ltheta
        losses['sclLoss'] = sclLoss
        return losses

    def val(self):
        """Validate the model on a single minibatch
        """
        count = 0
        self.set_eval()
        errors = list()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                normedThetah = inputs['htheta'] / 2 / np.pi
                normedThetav = inputs['vtheta'] / 2 / np.pi
                cattedInput = torch.cat([inputs[('color', 0, 0)], normedThetah, normedThetav], dim=1).cuda()

                outputs = self.models['depth'](self.models['encoder'](cattedInput))
                _, pred_depth = disp_to_depth(outputs[("disp", 0)][:,2:3,:,:], self.opt.min_depth, self.opt.max_depth)
                pred_depth = pred_depth * self.STEREO_SCALE_FACTOR
                for i in range(cattedInput.shape[0]):
                    gt_depth = self.gt_depths[count]
                    gt_height, gt_width = gt_depth.shape
                    cur_pred_depth = pred_depth[i:i+1,:,:,:]
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
                    # tensor2disp(outputs[("disp", 0)][:,2:3,:,:], ind=0, vmax=0.1).show()
                    count = count + 1
            del inputs, outputs
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

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        height = self.opt.height
        width = self.opt.width
        source_scale = 0
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)][:,2:3,:,:]
            disp = F.interpolate(disp, [height, width], mode="bilinear", align_corners=False)
            scaledDisp, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            frame_id = "s"
            T = inputs["stereo_T"]
            cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
            pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

            outputs[("depth", 0, scale)] = depth
            outputs[("sample", frame_id, scale)] = pix_coords
            outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)], outputs[("sample", frame_id, scale)], padding_mode="border")

            if scale == 0:
                outputs[("real_scale_disp", scale)] = scaledDisp * (torch.abs(inputs[("K", source_scale)][:, 0, 0] * T[:, 0, 3]).view(self.opt.batch_size, 1, 1,1).expand_as(scaledDisp))

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """

        l1_loss = torch.abs(target - pred).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def depth_compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        source_scale = 0
        target = inputs[("color", 0, source_scale)]
        outputs['selfOccMask'] = self.selfOccluMask(outputs[('real_scale_disp', source_scale)], inputs['stereo_T'][:, 0, 3])

        pholoss = 0
        l1loss = 0
        selector_depth = (inputs['depth_gt'] > 0).float()
        for scale in self.opt.scales:
            reprojection_loss = self.compute_reprojection_loss(outputs[("color", 's', scale)], target)
            identity_reprojection_loss = self.compute_reprojection_loss(inputs[("color", 's', source_scale)], target) + torch.randn(reprojection_loss.shape).cuda() * 0.00001
            combined = torch.cat((reprojection_loss, identity_reprojection_loss), dim=1)
            to_optimise, idxs = torch.min(combined, dim=1, keepdim=True)
            reprojection_loss_mask = (idxs != 1).float() * (1 - outputs['selfOccMask'])
            pholoss = pholoss + (reprojection_loss * reprojection_loss_mask).sum() / (reprojection_loss_mask.sum() + 1)

            pred_depth = outputs[('depth', 0, scale)] * self.STEREO_SCALE_FACTOR
            pred_depth = F.interpolate(pred_depth, [self.kittih, self.kittiw], mode='bilinear', align_corners=True)
            l1loss = l1loss + torch.sum(torch.abs(pred_depth - inputs['depth_gt']) * selector_depth) / (torch.sum(selector_depth) + 1)

        pholoss = pholoss / len(self.opt.scales)
        l1loss = l1loss / len(self.opt.scales)
        losses['pholoss'] = pholoss
        losses['l1loss'] = l1loss

        return losses

    @staticmethod
    def compute_proxy_supervised_loss(pred, target, valid_pixels, loss_mask):
        """ Compute proxy supervised loss (depth hint loss) for prediction.

            - valid_pixels is a mask of valid depth hint pixels (i.e. non-zero depth values).
            - loss_mask is a mask of where to apply the proxy supervision (i.e. the depth hint gave
            the smallest reprojection error)"""

        # first compute proxy supervised loss for all valid pixels
        depth_hint_loss = torch.log(torch.abs(target - pred) + 1) * valid_pixels

        # only keep pixels where depth hints reprojection loss is smallest
        depth_hint_loss = depth_hint_loss * loss_mask

        return depth_hint_loss

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
        return depth_errors

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

        figrgb = tensor2rgb(inputs[('color', 0, 0)], ind=vind)

        figdisp = tensor2disp(outputs[('disp', 0)][:,2:3,:,:], vmax=0.1, ind=0)
        _, predDepth = disp_to_depth(outputs[('disp', 0)][:,2:3,:,:], min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
        predDepth = predDepth * self.STEREO_SCALE_FACTOR
        predDepth = F.interpolate(predDepth, [self.kittih, self.kittiw], mode='bilinear', align_corners=True)
        predD2htheta, predD2vtheta = self.localthetadespKitti.get_theta(predDepth)
        fighpred_fromD = tensor2disp(predD2htheta - 1, vmax=4, ind=vind).resize(figdisp.size, pil.BILINEAR)
        figvpred_fromD = tensor2disp(predD2vtheta - 1, vmax=4, ind=vind).resize(figdisp.size, pil.BILINEAR)
        figcombined = np.concatenate([np.array(figrgb), np.array(figdisp), np.array(fighpred_fromD), np.array(figvpred_fromD)], axis=0)
        self.writers['train'].add_image('overview', (torch.from_numpy(figcombined).float() / 255).permute([2, 0, 1]), self.step)

        fig_thetagrad = tensor2grad(outputs['derivx'], percentile=80, viewind=0)
        fig_depthgrad = tensor2grad(outputs['num_grad'], percentile=80, viewind=0)
        fig_mask = tensor2disp(outputs['gradweights'], vmax=1, ind=0)
        fig_grad = np.concatenate([np.array(figrgb), np.array(fig_thetagrad), np.array(fig_depthgrad), np.array(fig_mask)], axis=0)
        self.writers['train'].add_image('fig_grad', (torch.from_numpy(fig_grad).float() / 255).permute([2, 0, 1]), self.step)

        figrgb_stereo = tensor2rgb(inputs[('color', 's', 0)], ind=vind)
        figrgb2 = tensor2rgb(inputs[('color', 0, 0)], ind=vind)
        mask = 1 - outputs['selfOccMask'][vind, 0, :, :].detach().cpu().numpy()
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, axis=2, repeats=3)
        figrgb2 = pil.fromarray((np.array(figrgb2).astype(np.float) * mask).astype(np.uint8))
        occmask = tensor2disp(outputs['selfOccMask'], vmax = 1, ind=vind)
        color_recon = tensor2rgb(outputs[('color', 's', 0)], ind=vind)
        combined2 = np.concatenate([figrgb2, figrgb_stereo, color_recon, occmask])
        self.writers['train'].add_image('rgb', (torch.from_numpy(combined2).float() / 255).permute([2, 0, 1]), self.step)

        fig_inputDepth = tensor2disp(self.depth2activation(inputs['predDepth']), vmax=0.1, ind=0)
        fig_htheta = tensor2disp(inputs['htheta'] - 1, vmax=4, ind=0)
        fig_vtheta = tensor2disp(inputs['vtheta'] - 1, vmax=4, ind=0)
        fig_input = np.concatenate([np.array(fig_inputDepth), np.array(fig_htheta), np.array(fig_vtheta)])
        self.writers['train'].add_image('input', (torch.from_numpy(fig_input).float() / 255).permute([2, 0, 1]), self.step)

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
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        # save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        # torch.save(self.model_optimizer.state_dict(), save_path)
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
    trainer = Trainer(opts)
    trainer.train()
