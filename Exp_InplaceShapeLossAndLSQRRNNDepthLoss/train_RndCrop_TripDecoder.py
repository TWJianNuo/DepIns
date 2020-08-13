from __future__ import absolute_import, division, print_function

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from options import MonodepthOptions

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

from Exp_InplaceShapeLossAndLSQRRNNDepthLoss import MonoDatasetRndCrop
import networks

import time
import json

warnings.filterwarnings("ignore")
options = MonodepthOptions()
opts = options.parse()

from collections import OrderedDict
from layers import *

import inplaceShapeLoss_cuda


class DepthDecoder_earlysplit(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder_earlysplit, self).__init__()
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        self.output_cat = ['htheta', 'vtheta']
        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            for cat in self.output_cat:
                # upconv_0
                num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 0, cat)] = ConvBlock(num_ch_in, num_ch_out)

                # upconv_1
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 1, cat)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            for c in self.output_cat:
                self.convs[("conv", s, c)] = Conv3x3(self.num_ch_dec[s], 1)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        for c in self.output_cat:
            x = input_features[-1]
            for i in range(4, -1, -1):
                x = self.convs[("upconv", i, 0, c)](x)
                x = [upsample(x)]
                if self.use_skips and i > 0:
                    x += [input_features[i - 1]]
                x = torch.cat(x, 1)
                x = self.convs[("upconv", i, 1, c)](x)

                if i in self.scales:
                    self.outputs[(c, i)] = self.sigmoid(self.convs[("conv", i, c)](x))
        return self.outputs


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
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["decoder"] = DepthDecoder_earlysplit(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["decoder"].to(self.device)

        self.parameters_to_train += list(self.models["decoder"].parameters())
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
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.set_layers()
        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            self.train_num, self.val_num))

        if self.opt.load_weights_folder is not None:
            self.load_model()

        self.save_opts()


        self.kittiw = 1220
        self.kittih = 365
        Kkitti = np.array([
            [0.58 * self.kittiw, 0, 0.5 * self.kittiw],
            [0, 1.92 * self.kittih, 0.5 * self.kittih],
            [0, 0, 1]], dtype=np.float32)
        self.localthetadespKitti = LocalThetaDesp(height=self.kittih, width=self.kittiw,batch_size=self.opt.batch_size, intrinsic=Kkitti).cuda()

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


        train_dataset = MonoDatasetRndCrop(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            is_train=not self.opt.no_aug, kitti_gt_path = self.opt.kitti_gt_path
        )

        val_dataset = MonoDatasetRndCrop(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            is_train=False, kitti_gt_path = self.opt.kitti_gt_path
        )

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=not self.opt.no_shuffle,
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

        outputs = dict()
        losses = dict()


        outputs.update(self.models['decoder'](self.models['encoder'](inputs[('color_aug', 0, 0)])))

        # Theta Branch
        losses.update(self.theta_compute_losses(inputs, outputs))

        # Depth Branch
        # self.generate_images_pred(inputs, outputs)
        # losses.update(self.depth_compute_losses(inputs, outputs))

        # Constrain Branch
        # losses.update(self.constrain_compute_losses(inputs, outputs))

        # losses['totLoss'] = losses['ltheta'] * self.opt.lthetaScale + \
        #                     losses['sclLoss'] * self.opt.sclLossScale + \
        #                     losses['l1loss'] * self.opt.l1lossScale + \
        #                     losses['pholoss'] * self.opt.pholossScale + \
        #                     losses['l1constrain'] * self.opt.l1constrainScale

        losses['totLoss'] = losses['ltheta'] * self.opt.lthetaScale + \
                            losses['sclLoss'] * self.opt.sclLossScale
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
        losses['l1constrain'] = l1constrain
        return losses

    def theta_compute_losses(self, inputs, outputs):
        losses = dict()
        ltheta = 0
        sclLoss = 0
        for i in range(len(self.opt.scales)):
            htheta_pred = outputs[('htheta', i)] * float(np.pi) * 2
            vtheta_pred = outputs[('vtheta', i)] * float(np.pi) * 2

            htheta_pred = F.interpolate(htheta_pred, [self.kittih, self.kittiw], mode='bilinear', align_corners=True)
            vtheta_pred = F.interpolate(vtheta_pred, [self.kittih, self.kittiw], mode='bilinear', align_corners=True)

            inbl, outbl, scl = self.localthetadespKitti.inplacePath_loss(depthmap=inputs['depthgt'], htheta=htheta_pred, vtheta=vtheta_pred, balancew = self.opt.balancew)

            if i == 0:
                outputs['htheta_pred'] = outputs[('htheta', i)] * float(np.pi) * 2
                outputs['vtheta_pred'] = outputs[('vtheta', i)] * float(np.pi) * 2

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
        count = 0
        self.set_eval()
        errors = list()

        self.kittiw = 1220
        self.kittih = 365

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                input_color = inputs[("color", 0, 0)].cuda()
                depthgt = inputs['depthgt'].cuda()
                outputs = self.models['decoder'](self.models['encoder'](input_color))
                htheta = outputs[('htheta', 0)] * 2 * np.pi
                vtheta = outputs[('vtheta', 0)] * 2 * np.pi

                htheta = F.interpolate(htheta, [self.kittih, self.kittiw], mode='bilinear', align_corners=True)
                vtheta = F.interpolate(vtheta, [self.kittih, self.kittiw], mode='bilinear', align_corners=True)

                ratioh, ratiohl, ratiov, ratiovl = self.localthetadespKitti.get_ratio(htheta=htheta, vtheta=vtheta)

                logdepthd = torch.log(depthgt)
                valindic = depthgt > 0

                lossrec = torch.zeros_like(logdepthd)
                inplaceShapeLoss_cuda.inplaceShapeLoss_integration(logdepthd, ratiohl, ratiovl, valindic.int(), lossrec, 30, 30)

                err = torch.sum(torch.abs(lossrec)) / torch.sum(lossrec > 0)
                errors.append(err.cpu().numpy())
                count = count + 1
                del inputs, outputs
        mean_err = np.array(errors).mean(0)

        if mean_err < self.minabsrel:
            self.minabsrel = mean_err
            self.save_model("best_aggregated_model")

        print("\nCurrent Performance: %f" % mean_err)

        print("\nBest Performance: %f" % self.minabsrel)

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
            # if scale == 0:
            #     outputs['reprojection_loss'] = reprojection_loss

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

    def compute_depth_losses(self, depth, depth_gt):
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
        fighpred = tensor2disp(outputs['htheta_pred'] - 1, vmax = 4, ind=vind)
        figvpred = tensor2disp(outputs['vtheta_pred'] - 1, vmax = 4, ind=vind)

        figcombined = np.concatenate([np.array(figrgb), np.array(fighpred), np.array(figvpred)], axis=0)
        self.writers['train'].add_image('overview', (torch.from_numpy(figcombined).float() / 255).permute([2, 0, 1]), self.step)
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
