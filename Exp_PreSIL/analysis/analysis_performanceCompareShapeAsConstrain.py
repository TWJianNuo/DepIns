from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from torch.utils.data import DataLoader

# Resolve Tensorbard Confliction across pytorch version
import torch
import warnings

from Exp_PreSIL.dataloader_kitti import KittiDataset

import networks


from layers import *
from networks import *

import argparse

default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",                  type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                    type=str,                               help="path to kitti gt file")
parser.add_argument("--semanticspred_path",         type=str,   default='None',             help="path to kitti semantics prediction file")
parser.add_argument("--instancepred_path",          type=str,   default='None',             help="path to kitti instance prediction file")
parser.add_argument("--split",                      type=str,                               help="train/val split to use")
parser.add_argument("--num_layers",                 type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                     type=int,   default=320,                help="input image height")
parser.add_argument("--width",                      type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                       type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                       type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",                  type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",                  type=float, default=100.0,              help="maximum depth")
parser.add_argument("--vlsfold",                    type=str)

# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=1,                 help="batch size")
parser.add_argument("--load_weights_folder_depth",  type=str,   default=None,               help="name of models to load")
parser.add_argument("--load_weights_folder_depthbs",   type=str,   default=None,               help="name of models to load")

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

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.models_bs = {}
        self.parameters_to_train = []

        self.device = "cuda"

        self.models["encoder_depth"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder_depth"].to(self.device)
        self.models["depth"] = DepthDecoder(self.models["encoder_depth"].num_ch_enc, num_output_channels=1)
        self.models["depth"].to(self.device)

        self.models_bs["encoder_depth"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models_bs["encoder_depth"].to(self.device)
        self.models_bs["depth"] = DepthDecoder(self.models_bs["encoder_depth"].num_ch_enc, num_output_channels=1)
        self.models_bs["depth"].to(self.device)

        self.set_dataset()

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.load_model()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.minabsrel = 1e10
        self.maxa1 = -1e10

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.getcwd(), "splits", self.opt.split, "{}_files.txt")
        val_filenames = readlines(fpath.format("test"))
        # val_filenames = readlines('/home/shengjie/Documents/Project_SemanticDepth/splits/eigen/test_files.txt')

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, instancepred_path=self.opt.instancepred_path
        )

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False)

        self.val_num = val_dataset.__len__()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
        for m in self.models_bs.values():
            m.eval()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        xx, yy = np.meshgrid(range(self.opt.crpw), range(self.opt.crph), indexing='xy')
        cm = plt.cm.get_cmap('seismic')
        semidenseroot = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt'

        deptherrbs = list()
        deptherrpred = list()

        a1bssemi = list()
        a1predsemi = list()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                outputsbs = self.models_bs['depth'](self.models_bs['encoder_depth'](inputs['color']))
                outputs = self.models['depth'](self.models['encoder_depth'](inputs['color']))

                pred_depthbs = F.interpolate(outputsbs['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=False)
                _, pred_depthbs = disp_to_depth(pred_depthbs, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
                pred_depthbs = pred_depthbs * self.STEREO_SCALE_FACTOR

                pred_depth = F.interpolate(outputs['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=False)
                _, pred_depth = disp_to_depth(pred_depth, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
                pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

                mask = (inputs['depthgt'] > self.MIN_DEPTH).int() * (inputs['depthgt'] < self.MAX_DEPTH).int() * (inputs['instancepred'] > 0).int()
                cropmask = torch.zeros_like(mask)
                cropmask[0, 0, int(0.40810811 * self.opt.crph): int(0.99189189 * self.opt.crph), int(0.03594771 * self.opt.crpw): int(0.96405229 * self.opt.crpw)] = 1
                mask[cropmask == 0] = 0
                mask = mask == 1

                pred_depth = torch.clamp(pred_depth, min=self.MIN_DEPTH, max=self.MAX_DEPTH)
                pred_depthbs = torch.clamp(pred_depthbs, min=self.MIN_DEPTH, max=self.MAX_DEPTH)

                if torch.sum(mask) == 0:
                    continue

                gtnp = inputs['depthgt'][mask].cpu().numpy()
                bsnp = pred_depthbs[mask].cpu().numpy()
                prednp = pred_depth[mask].cpu().numpy()

                deptherrbs.append(compute_errors(gtnp, bsnp))
                deptherrpred.append(compute_errors(gtnp, prednp))

                # figname = inputs['tag'][0].split(' ')[0].split('/')[1] + '_' + inputs['tag'][0].split(' ')[1]
                #
                # fig1 = tensor2rgb(F.interpolate(inputs['color'], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=False), ind=0)
                # fig2 = tensor2disp(1 / pred_depthbs, vmax=0.15, ind=0)
                # fig3 = tensor2disp(1 / pred_depth, vmax=0.15, ind=0)
                #
                # figcombined = np.concatenate([np.array(fig1), np.array(fig2), np.array(fig3)], axis=0)
                # pil.fromarray(figcombined).save(os.path.join(self.opt.vlsfold, "{}.png".format(figname)))

                a1bs = np.maximum((gtnp / bsnp), (bsnp / gtnp))
                a1pred = np.maximum((gtnp / prednp), (prednp / gtnp))
                contrast = (a1pred - 1) - (a1bs - 1)
                contrast = contrast / 0.15 + 0.5

                masknp = mask[0,0,:,:].cpu().numpy()
                xxvls = xx[masknp]
                yyvls = yy[masknp]
                colors = cm(contrast)

                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 9))
                ax1.imshow(tensor2rgb(F.interpolate(inputs['color'], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=False), ind=0))
                ax1.scatter(xxvls, yyvls, s=1, c=colors[:, 0:3])
                ax1.title.set_text("Bs: %f, Pred: %f" % (np.mean(a1bs), np.mean(a1pred)))

                semidensegt = pil.open(os.path.join(semidenseroot, inputs['tag'][0].split(' ')[0], 'image_02', "{}.png".format(inputs['tag'][0].split(' ')[1].zfill(10))))
                w, h = semidensegt.size
                left = int((w - self.opt.crpw) / 2)
                top = int((h - self.opt.crph) / 2)
                semidensegt = semidensegt.crop((left, top, left + self.opt.crpw, top + self.opt.crph))
                semidensegt = np.array(semidensegt).astype(np.float) / 256.0
                semidensegt = torch.from_numpy(semidensegt).unsqueeze(0).unsqueeze(0).cuda()

                mask_semi = (semidensegt > self.MIN_DEPTH).int() * (semidensegt < self.MAX_DEPTH).int() * (inputs['instancepred'] > 0).int()
                cropmask_semi = torch.zeros_like(mask_semi)
                cropmask_semi[0, 0, int(0.40810811 * self.opt.crph): int(0.99189189 * self.opt.crph), int(0.03594771 * self.opt.crpw): int(0.96405229 * self.opt.crpw)] = 1
                mask_semi[cropmask_semi == 0] = 0
                mask_semi = mask_semi == 1

                gtnp_semi = semidensegt[mask_semi].cpu().numpy()
                bsnp_semi = pred_depthbs[mask_semi].cpu().numpy()
                prednp_semi = pred_depth[mask_semi].cpu().numpy()

                a1bs_semi = np.maximum((gtnp_semi / bsnp_semi), (bsnp_semi / gtnp_semi))
                a1pred_semi = np.maximum((gtnp_semi / prednp_semi), (prednp_semi / gtnp_semi))
                contrast_semi = (a1pred_semi - 1) - (a1bs_semi - 1)
                contrast_semi = contrast_semi / 0.15 + 0.5

                a1bssemi.append((a1bs_semi).mean())
                a1predsemi.append((a1pred_semi).mean())

                masknp_semi = mask_semi[0,0,:,:].cpu().numpy()
                xxvls_semi = xx[masknp_semi]
                yyvls_semi = yy[masknp_semi]
                colors_semi = cm(contrast_semi)
                ax2.imshow(tensor2rgb(F.interpolate(inputs['color'], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=False), ind=0))
                ax2.scatter(xxvls_semi, yyvls_semi, s=1, c=colors_semi[:, 0:3])
                ax2.title.set_text("Bs: %f, Pred: %f" % (np.mean(a1bs_semi), np.mean(a1pred_semi)))

                ax3.imshow(tensor2disp(1 / pred_depthbs, vmax=0.15, ind=0))
                ax4.imshow(tensor2disp(1 / pred_depth, vmax=0.15, ind=0))

                figname = inputs['tag'][0].split(' ')[0].split('/')[1] + '_' + inputs['tag'][0].split(' ')[1]
                plt.savefig(os.path.join(self.opt.vlsfold, "{}.png".format(figname)), bbox_inches='tight')
                plt.close()

                print("batch %d finished" % batch_idx)

        err1 = np.array(deptherrbs)
        err1 = np.mean(err1, axis=0)
        err2 = np.array(deptherrpred)
        err2 = np.mean(err2, axis=0)
        print("\nBaseline Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*err1.tolist()) + "\\\\")

        print("\nPrediction Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*err2.tolist()) + "\\\\")

        a1bssemi = np.array(a1bssemi).mean()
        a1predsemi = np.array(a1predsemi).mean()
        print("\nBaseline A1 Performance: %f" % a1bssemi)
        print("\nPrediction A1 Performance: %f" % a1predsemi)

    def load_model(self):
        """Load model(s) from disk
        """
        load_depth_folder = os.path.expanduser(self.opt.load_weights_folder_depth)
        load_depthbs_folder = os.path.expanduser(self.opt.load_weights_folder_depthbs)

        assert os.path.isdir(load_depth_folder), "Cannot find folder {}".format(self.opt.load_weights_folder_depth)
        assert os.path.isdir(load_depthbs_folder), "Cannot find folder {}".format(self.opt.load_weights_folder_depthbs)

        models_to_load = ['encoder_depth', 'depth']
        pthfilemapping = {'encoder_depth': 'encoder', 'depth': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_depth_folder, "{}.pth".format(pthfilemapping[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_depthbs_folder, "{}.pth".format(pthfilemapping[n]))
            if n in self.models_bs:
                model_dict = self.models_bs[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models_bs[n].load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
