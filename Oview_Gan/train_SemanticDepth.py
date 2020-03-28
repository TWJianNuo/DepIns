from __future__ import absolute_import, division, print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from options import MonodepthOptions
import warnings

import torch.optim as optim
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

from layers import *

import datasets
import networks

import time
import json

warnings.filterwarnings("ignore")
options = MonodepthOptions()
opts = options.parse()
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
        self.sfx = nn.Softmax()
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.models_D = {}
        self.models_D["discriminator"] = networks.ResnetDiscriminator(
            18, self.opt.weights_init == "pretrained")
        self.models_D["discriminator"].to(self.device)

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
        if not self.opt.no_ssim:
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

        self.bcewl = nn.BCEWithLogitsLoss()
        self.sig = nn.Sigmoid()
        self.DOptimizer = optim.Adam(self.models_D["discriminator"].parameters(), self.opt.lrD)
        self.accRec = list() # Record recent 40 results

    def set_layers(self):
        """properly handle layer initialization under multiple dataset situation
        """
        self.backproject_depth = {}
        self.project_3d = {}

        if self.opt.selfocclu:
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

        if self.opt.bnMorphLoss:
            from bnmorph.bnmorph import BNMorph
            self.tool = grad_computation_tools(batch_size=self.opt.batch_size, height=self.opt.height,
                                               width=self.opt.width).cuda()

            self.bnmorph = BNMorph(height=self.opt.height, width=self.opt.width, senseRange=20).cuda()
            self.foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17,
                                   18]  # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
            self.textureMeasure = TextureIndicatorM().cuda()

        if self.opt.Dloss:
            from eppl_render.eppl_render import EpplRender
            self.epplrender = EpplRender(height=self.opt.height, width=self.opt.width, batch_size=self.opt.batch_size, sampleNum=self.opt.eppsm).cuda()

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.syn_split, "{}_files.txt")
        syn_train_filenames = readlines(fpath.format("train"))
        syn_val_filenames = readlines(fpath.format("val"))

        train_dataset = datasets.KITTIRAWDataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, load_seman = True, load_hints = self.opt.load_hints, hints_path = self.opt.hints_path,
            load_syn=self.opt.load_syn, syn_filenames=syn_train_filenames, syn_root=self.opt.syn_path
        )

        val_dataset = datasets.KITTIRAWDataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, load_seman = True, load_hints = self.opt.load_hints, hints_path = self.opt.hints_path,
            load_syn=self.opt.load_syn, syn_filenames=syn_val_filenames, syn_root=self.opt.syn_path
        )

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.train_num = train_dataset.__len__()
        self.val_num = val_dataset.__len__()
        self.num_total_steps = self.train_num // self.opt.batch_size * self.opt.num_epochs


    def set_train_D(self):
        """Convert all models to training mode
        """
        for m in self.models_D.values():
            m.train()

    def set_eval_D(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models_D.values():
            m.eval()

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

    def supervised_with_morph(self, inputs):
        outputs, losses = self.process_batch(inputs)

        stable_disp = outputs['disp', 0].detach()
        disparity_grad_bin = self.tool.get_disparityEdge(outputs['disp', 0])
        semantics_grad_bin = self.tool.get_semanticsEdge(inputs['semanLabel'])
        # tensor2disp(semantics_grad_bin, ind=0, vmax=1).show()

        morphedx, morphedy, ocoeff = self.bnmorph.bnmorph(disparity_grad_bin, semantics_grad_bin)
        morphedx = (morphedx / (self.opt.width - 1) - 0.5) * 2
        morphedy = (morphedy / (self.opt.height - 1) - 0.5) * 2
        grid = torch.cat([morphedx, morphedy], dim=1).permute(0, 2, 3, 1)
        dispMaps_morphed = F.grid_sample(stable_disp, grid, padding_mode="border")
        outputs['dispMaps_morphed'] = dispMaps_morphed

        with torch.no_grad():
            th = 1.05
            ssim_val_predict = self.compute_reprojection_loss(outputs[('color', 's', 0)], inputs[('color', 0, 0)])
            scaledDisp, depth = disp_to_depth(dispMaps_morphed, self.opt.min_depth, self.opt.max_depth)
            frame_id = "s"
            T = inputs["stereo_T"]
            cam_points = self.backproject_depth[0](depth, inputs[("inv_K", 0)])
            pix_coords = self.project_3d[0](cam_points, inputs[("K", 0)], T)
            morphed_rgb = F.grid_sample(inputs[("color", frame_id, 0)], pix_coords, padding_mode="border")
            ssim_val_morph = self.compute_reprojection_loss(morphed_rgb, inputs[('color', 0, 0)])
            selector_mask = (ssim_val_predict - th * ssim_val_morph > 0).float() * outputs['grad_proj_msak']
            texture_measure = torch.mean(self.textureMeasure(inputs[('color', 0, 0)]), dim=1, keepdim=True)

        losses["similarity_loss"] = 100 * torch.sum(torch.log(1 + torch.abs(dispMaps_morphed - outputs['disp', 0]) * texture_measure) * selector_mask) / (torch.sum(selector_mask) + 1)
        losses['totLoss'] = losses["similarity_loss"] * self.opt.bnMorphLoss_w + losses['totLoss']

        camIndex = [3]
        rendered_syn, _, _ = self.epplrender.forward(depthmap=inputs[('syn_depth', 0)],
                                                     semanticmap=inputs['syn_semanLabel'],
                                                     intrinsic=inputs['realIn'],
                                                     extrinsic=inputs['realEx'],
                                                     camIndex=camIndex)

        rendered_real, _, _ = self.epplrender.forward(depthmap=outputs[('depth', 0, 0)] * self.STEREO_SCALE_FACTOR,
                                                    semanticmap=inputs['semanLabel'],
                                                    intrinsic=inputs['realIn'],
                                                    extrinsic=inputs['realEx'],
                                                    camIndex=camIndex)

        outputs['rendered_syn'] = rendered_syn
        outputs['rendered_real'] = rendered_real
        self.set_eval_D()
        pred_real = self.models_D['discriminator'](rendered_real)
        loss_G = self.bcewl(pred_real,torch.ones_like(pred_real))
        losses['loss_G'] = loss_G
        if np.sum(np.array(self.accRec)) > 0.8:
            losses['totLoss'] = losses['totLoss'] + loss_G * self.opt.weightD
            losses['isD'] = 1
        else:
            losses['isD'] = 0
        self.model_optimizer.zero_grad()
        losses['totLoss'].backward()
        self.model_optimizer.step()

        # Train Discriminator
        self.set_eval()
        self.set_train_D()
        pred_syn = self.models_D['discriminator'](rendered_syn)
        pred_real = self.models_D['discriminator'](rendered_real.detach())
        loss_D = (self.bcewl(pred_syn, torch.ones_like(pred_syn)) + self.bcewl(pred_real, torch.zeros_like(pred_real))) / 2
        losses['loss_D'] = loss_D
        self.DOptimizer.zero_grad()
        loss_D.backward()
        self.DOptimizer.step()

        acc_rate = (torch.sum(self.sig(pred_syn) > 0.5) + torch.sum(self.sig(pred_real) <= 0.5)) / (2 * torch.sum(torch.ones_like(pred_syn)))
        losses['D_acc_rate'] = acc_rate
        if len(self.accRec) < 100:
            self.accRec.append(acc_rate.detach().cpu().numpy())
        else:
            self.accRec[self.step % 100] = acc_rate.detach().cpu().numpy()
        return outputs, losses

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.supervised_with_morph(inputs)

            duration = time.time() - before_op_time


            if self.step % 100 == 0:
                self.log_time(batch_idx, duration, losses['loss_depth/0'], losses["totLoss"])

            if self.step % 50 == 0:
                self.record_img(inputs, outputs)

            if self.step % 2 == 0:
                self.log("train", inputs, outputs, losses, writeImage=False)

            if self.step % 15 == 0:
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if not (key == 'entry_tag' or key == 'syn_tag'):
                inputs[key] = ipt.to(self.device)

        features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs = dict()
        outputs.update(self.models["depth"](features))
        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

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
            self.compute_depth_losses(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses, False)
            del inputs, outputs, losses
        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        height = self.opt.height
        width = self.opt.width
        source_scale = 0
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [height, width], mode="bilinear", align_corners=False)
            scaledDisp, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            frame_id = "s"
            T = inputs["stereo_T"]
            cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
            pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

            outputs[("disp", scale)] = disp
            outputs[("depth", 0, scale)] = depth
            outputs[("sample", frame_id, scale)] = pix_coords
            outputs[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)], outputs[("sample", frame_id, scale)], padding_mode="border")

            if scale == 0:
                cam_points = self.backproject_depth[source_scale](inputs['depth_hint'], inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
                outputs[("color_depth_hint", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)], pix_coords, padding_mode="border")

                outputs['grad_proj_msak'] = ((pix_coords[:, :, :, 0] > -1) * (pix_coords[:, :, :, 1] > -1) * (pix_coords[:, :, :, 0] < 1) * (pix_coords[:, :, :, 1] < 1)).unsqueeze(1).float()
                outputs[("real_scale_disp", scale)] = scaledDisp * (torch.abs(inputs[("K", source_scale)][:, 0, 0] * T[:, 0, 3]).view(self.opt.batch_size, 1, 1,1).expand_as(scaledDisp))

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """

        l1_loss = torch.abs(target - pred).mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        losses["totLoss"] = 0

        source_scale = 0
        target = inputs[("color", 0, source_scale)]
        if self.opt.selfocclu:
            sourceSSIMMask = self.selfOccluMask(outputs[('real_scale_disp', source_scale)], inputs['stereo_T'][:, 0, 3])
        else:
            sourceSSIMMask = torch.zeros_like(outputs[('real_scale_disp', source_scale)])
        outputs['ssimMask'] = sourceSSIMMask

        # compute depth hint reprojection loss
        if self.opt.load_hints:
            pred = outputs[("color_depth_hint", 's', 0)]
            depth_hint_reproj_loss = self.compute_reprojection_loss(pred, inputs[("color", 0, 0)])
            depth_hint_reproj_loss += 1000 * (1 - inputs['depth_hint_mask'])
        else:
            depth_hint_reproj_loss = None

        for scale in self.opt.scales:
            reprojection_loss = self.compute_reprojection_loss(outputs[("color", 's', scale)], target)
            identity_reprojection_loss = self.compute_reprojection_loss(inputs[("color", 's', source_scale)], target) + torch.randn(reprojection_loss.shape).cuda() * 0.00001
            combined = torch.cat((reprojection_loss, identity_reprojection_loss, depth_hint_reproj_loss), dim=1)
            to_optimise, idxs = torch.min(combined, dim=1, keepdim=True)

            reprojection_loss_mask = (idxs != 1).float() * (1 - outputs['ssimMask'])
            depth_hint_loss_mask = (idxs == 2).float()


            losses["loss_depth/{}".format(scale)] = (reprojection_loss * reprojection_loss_mask).sum() / (reprojection_loss_mask.sum() +1e-7)
            losses["totLoss"] += losses["loss_depth/{}".format(scale)] / self.num_scales
            # proxy supervision loss
            if self.opt.load_hints:
                valid_pixels = inputs['depth_hint_mask']

                depth_hint_loss = self.compute_proxy_supervised_loss(outputs[('depth', 0, scale)], inputs['depth_hint'], valid_pixels,
                                                                     depth_hint_loss_mask)
                depth_hint_loss = depth_hint_loss.sum() / (depth_hint_loss_mask.sum() + 1e-7)
                losses['depth_hint_loss/{}'.format(scale)] = depth_hint_loss
                losses["totLoss"] += depth_hint_loss / self.num_scales * self.opt.depth_hint_param

            if self.opt.disparity_smoothness > 0:
                mult_disp = outputs[('disp', scale)]
                mean_disp = mult_disp.mean(2, True).mean(3, True)
                norm_disp = mult_disp / (mean_disp + 1e-7)
                losses["loss_smooth"] = get_smooth_loss(norm_disp, target) / (2 ** scale)
                losses["totLoss"] += self.opt.disparity_smoothness * losses["loss_smooth"] / self.num_scales

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

    def log_time(self, batch_idx, duration, loss_depth, loss_tot):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
                                     self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}\nloss_depth: {:.5f} | loss_tot: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss_depth, loss_tot, sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def record_img(self, inputs, outputs):
        viewIndex = 0
        fig_seman_syn = tensor2semantic(inputs['syn_semanLabel'], ind=viewIndex)
        fig_render_syn = tensor2disp(outputs['rendered_syn'], ind = viewIndex, vmax=0.1)

        fig_seman_real = tensor2semantic(inputs['semanLabel'], ind=viewIndex)
        fig_disp = tensor2disp(outputs[('disp', 0)], ind=viewIndex, vmax=0.1)
        fig_rgb = tensor2rgb(inputs[('color', 0, 0)], ind=viewIndex)
        fig_render_real = tensor2disp(outputs['rendered_real'], ind=viewIndex, vmax=0.1)

        combined3 = np.concatenate([np.array(fig_render_syn), np.array(fig_seman_syn)], axis=1)
        combined1 = np.concatenate([np.array(fig_disp), np.array(fig_rgb)], axis=1)
        combined2 = np.concatenate([np.array(fig_render_real), np.array(fig_seman_real)], axis=1)

        combined = np.concatenate([combined3, combined1, combined2], axis=0)
        self.writers['train'].add_image('rendered', torch.from_numpy(combined).float() / 255, dataformats = 'HWC', global_step = self.step)
    def log(self, mode, inputs, outputs, losses, writeImage=False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            if l != 'totLoss':
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

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
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

        for model_name, model in self.models_D.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        save_path = os.path.join(save_folder, "{}.pth".format("adam_D"))
        torch.save(self.DOptimizer.state_dict(), save_path)
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
