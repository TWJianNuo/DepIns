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
parser.add_argument("--split",                      type=str,                               help="train/val split to use")
parser.add_argument("--num_layers",                 type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                     type=int,   default=320,                help="input image height")
parser.add_argument("--width",                      type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                       type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                       type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",                  type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",                  type=float, default=100.0,              help="maximum depth")


# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=1,                 help="batch size")
parser.add_argument("--load_weights_folder_depth",  type=str,   default=None,               help="name of models to load")
parser.add_argument("--load_weights_folder_norm",   type=str,   default=None,               help="name of models to load")

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
        self.parameters_to_train = []

        self.device = "cuda"

        self.models["encoder_depth"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder_depth"].to(self.device)
        self.models["depth"] = DepthDecoder(self.models["encoder_depth"].num_ch_enc, num_output_channels=1)
        self.models["depth"].to(self.device)

        self.models["encoder_norm"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder_norm"].to(self.device)
        self.models["norm"] = DepthDecoder(self.models["encoder_norm"].num_ch_enc, num_output_channels=3)
        self.models["norm"].to(self.device)

        self.set_dataset()

        self.load_model()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.minabsrel = 1e10
        self.maxa1 = -1e10

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()

        self.crph = 365
        self.crpw = 1220

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "{}_files.txt")
        self.val_filenames = readlines(fpath.format("train"))

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def getpredmask(self, rgbnp):
        gth, gtw, _ = rgbnp.shape

        rgbs = np.stack([rgbnp[:self.crph, :self.crpw, :], rgbnp[gth-self.crph:, gtw-self.crpw:, :], rgbnp[:self.crph, gtw-self.crpw:, :], rgbnp[gth-self.crph:, :self.crpw, :]], axis=0)

        weights = np.zeros([gth, gtw])
        weights[:self.crph, :self.crpw] = weights[:self.crph, :self.crpw] + 1
        weights[:self.crph, gtw-self.crpw:] = weights[:self.crph, gtw-self.crpw:] + 1
        weights[gth-self.crph:, gtw-self.crpw:] = weights[gth-self.crph:, gtw-self.crpw:] + 1
        weights[gth-self.crph:, :self.crpw] = weights[gth-self.crph:, :self.crpw] + 1

        # rgbredo = np.zeros_like(rgbnp, dtype=np.float32)
        # rgbredo[:crph, :crpw, :] += rgbnp[:crph, :crpw, :]
        # rgbredo[:crph, gtw-crpw:, :] += rgbnp[:crph, gtw-crpw:, :]
        # rgbredo[gth-crph:, gtw-crpw:, :] += rgbnp[gth-crph:, gtw-crpw:, :]
        # rgbredo[gth-crph:, :crpw, :] += rgbnp[gth-crph:, :crpw, :]
        # rgbredo = rgbredo / np.stack([weights, weights, weights], axis=2)
        #
        # assert np.abs(rgbredo - rgbnp).max() < 1e-3
        return rgbs, weights

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        minang = -np.pi/3*2
        maxang = 2*np.pi - np.pi/3*2
        vind = 0

        semidense_depthfolder = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt'
        rawdata_root = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
        semantics_root = '/home/shengjie/Documents/Data/Kitti/kitti_raw/kitti_data'
        dirmapping = {'l': 'image_02', 'r': 'image_03'}

        weightsx = torch.Tensor([[0., 0., 0],
                                [-2., 0., 2.],
                                [0., 0., 0]]).unsqueeze(0).unsqueeze(0)
        weightsx = weightsx / 4 / 2

        weightsy = torch.Tensor([[0., -2., 0.],
                                 [0., 0., 0.],
                                 [0., 2., 0.]]).unsqueeze(0).unsqueeze(0)
        weightsy = weightsy / 4 / 2
        diffx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        diffy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        diffx.weight = nn.Parameter(weightsx, requires_grad=False)
        diffy.weight = nn.Parameter(weightsy, requires_grad=False)

        diffx = diffx.cuda()
        diffy = diffy.cuda()

        vlsfold = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/vls_colinearity'
        with torch.no_grad():
            for entry in self.val_filenames:
                seq, frame, dir = entry.split(' ')

                rgbpath = os.path.join(rawdata_root, seq, dirmapping[dir], "data", "{}.png".format(frame.zfill(10)))
                depthpath = os.path.join(semidense_depthfolder, seq, dirmapping[dir], "{}.png".format(frame.zfill(10)))
                semanticspath = os.path.join(semantics_root, seq, 'semantic_prediction', dirmapping[dir], "{}.png".format(frame.zfill(10)))

                rgb = pil.open(rgbpath)
                rgbnp = np.array(rgb)
                # depth = np.array(pil.open(depthpath)).astype(np.float32) / 256.0
                # semantics = np.array(pil.open(semanticspath))

                from kitti_utils import read_calib_file
                cam2cam = read_calib_file(os.path.join(rawdata_root, seq.split('/')[0], 'calib_cam_to_cam.txt'))
                K = np.eye(4)
                K[0:3, :] = cam2cam['P_rect_0{}'.format(dirmapping[dir][-1])].reshape(3, 4)
                K = torch.from_numpy(K).float().unsqueeze(0).cuda()

                gtw, gth = rgb.size

                sfoptimizer = SurfaceNormalOptimizer(height=gth, width=gtw, batch_size=1).cuda()

                rgbs, weights = self.getpredmask(rgbnp)
                rgbstorch = torch.from_numpy(rgbs.astype(np.float32)).permute([0, 3, 1, 2]).cuda() / 255.0
                rgbstorch = F.interpolate(rgbstorch, [self.opt.height, self.opt.width], mode='bilinear', align_corners=True)
                weightstorch = torch.from_numpy(weights).float().unsqueeze(0).expand([3, -1, -1]).cuda()

                outputs_norm = self.models['norm'](self.models['encoder_norm'](rgbstorch))
                pred_ang_netsize = (outputs_norm[("disp", 0)] - 0.5) * 2 * np.pi
                pred_ang_cropsize = F.interpolate(pred_ang_netsize, [self.crph, self.crpw], mode='bilinear', align_corners=True)
                pred_ang = torch.zeros([3, gth, gtw], device='cuda')

                pred_ang[:, :self.crph, :self.crpw] += pred_ang_cropsize[0]
                pred_ang[:, gth - self.crph:, gtw - self.crpw:] += pred_ang_cropsize[1]
                pred_ang[:, :self.crph, gtw - self.crpw:] += pred_ang_cropsize[2]
                pred_ang[:, gth - self.crph:, :self.crpw] += pred_ang_cropsize[3]
                pred_ang = (pred_ang / weightstorch).unsqueeze(0)

                prednorm = sfoptimizer.ang2normal(pred_ang, K)

                gradanghx = diffx(pred_ang[:, 0:1, :, :])
                gradanghy = diffy(pred_ang[:, 1:2, :, :])

                outputs_depth = self.models['depth'](self.models['encoder_depth'](rgbstorch))
                _, pred_depth_netsize = disp_to_depth(outputs_depth[("disp", 0)], min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
                pred_depth_netsize = pred_depth_netsize * self.STEREO_SCALE_FACTOR
                pred_depth_netsize = F.interpolate(pred_depth_netsize, [self.crph, self.crpw], mode='bilinear', align_corners=True)
                pred_depth = torch.zeros([1, gth, gtw], device='cuda')

                pred_depth[:, :self.crph, :self.crpw] += pred_depth_netsize[0]
                pred_depth[:, gth - self.crph:, gtw - self.crpw:] += pred_depth_netsize[1]
                pred_depth[:, :self.crph, gtw - self.crpw:] += pred_depth_netsize[2]
                pred_depth[:, gth - self.crph:, :self.crpw] += pred_depth_netsize[3]
                pred_depth = (pred_depth / weightstorch[0:1]).unsqueeze(0)

                depth_ang = sfoptimizer.depth2ang(pred_depth, K)
                gradanghx_depth = diffx(depth_ang[:, 0:1, :, :])
                gradanghy_depth = diffy(depth_ang[:, 1:2, :, :])

                # rgb.show()
                # tensor2disp(torch.abs(gradanghx), vmax=0.2, ind=0).show()
                # tensor2disp(torch.abs(gradanghy), vmax=0.2, ind=0).show()
                # tensor2disp(pred_ang[:, 0:1, :, :] - minang, vmax=maxang, ind=vind).show()
                # tensor2disp(pred_ang[:, 1:2, :, :] - minang, vmax=maxang, ind=vind).show()
                # tensor2rgb((prednorm + 1) / 2, ind=0).show()

                fig1 = rgb
                fig2 = tensor2disp(torch.abs(gradanghx), vmax=0.2, ind=0)
                fig3 = tensor2disp(torch.abs(gradanghy), vmax=0.2, ind=0)
                fig4 = tensor2disp(pred_ang[:, 0:1, :, :] - minang, vmax=maxang, ind=vind)
                fig5 = tensor2disp(pred_ang[:, 1:2, :, :] - minang, vmax=maxang, ind=vind)

                fig6 = tensor2disp(1/pred_depth, vmax=0.2, ind=0)
                fig7 = tensor2disp(torch.abs(gradanghx_depth), vmax=0.2, ind=0)
                fig8 = tensor2disp(torch.abs(gradanghy_depth), vmax=0.2, ind=0)
                fig9 = tensor2disp(depth_ang[:, 0:1, :, :] - minang, vmax=maxang, ind=vind)
                fig10 = tensor2disp(depth_ang[:, 1:2, :, :] - minang, vmax=maxang, ind=vind)
                figcombinedl = np.concatenate([np.array(fig1), np.array(fig2), np.array(fig3), np.array(fig4), np.array(fig5)], axis=0)
                figcombinedr = np.concatenate([np.array(fig6), np.array(fig7), np.array(fig8), np.array(fig9), np.array(fig10)], axis=0)

                figcombined = pil.fromarray(np.concatenate([figcombinedl, figcombinedr], axis=1))

                figcombined.save(os.path.join(vlsfold, "{}_{}_{}.png".format(seq.split('/')[0], frame, dir)))

                print("{} finished".format(entry))

    def load_model(self):
        """Load model(s) from disk
        """
        load_depth_folder = os.path.expanduser(self.opt.load_weights_folder_depth)
        load_norm_folder = os.path.expanduser(self.opt.load_weights_folder_norm)

        assert os.path.isdir(load_depth_folder), "Cannot find folder {}".format(self.opt.load_weights_folder_depth)
        assert os.path.isdir(load_norm_folder), "Cannot find folder {}".format(self.opt.load_weights_folder_norm)

        models_to_load = ['encoder_depth', 'depth']
        pthfilemapping = {'encoder_depth': 'encoder', 'depth': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder_depth, "{}.pth".format(pthfilemapping[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

        models_to_load = ['encoder_norm', 'norm']
        pthfilemapping = {'encoder_norm': 'encoder', 'norm': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder_norm, "{}.pth".format(pthfilemapping[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
