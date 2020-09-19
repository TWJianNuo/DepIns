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

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "{}_files.txt")
        val_filenames = readlines(fpath.format("train"))

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, semanticspred_path=self.opt.semanticspred_path
        )

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True)

        self.val_num = val_dataset.__len__()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def val(self):
        """Validate the model on a single minibatch
        """
        import matlab
        import matlab.engine
        eng = matlab.engine.start_matlab()

        self.set_eval()
        errors = list()

        minang = - np.pi / 3 * 2
        maxang = 2 * np.pi - np.pi / 3 * 2
        vind = 0
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                _, _, gt_height, gt_width = inputs['depthgt'].shape

                outputs_depth = self.models['depth'](self.models['encoder_depth'](inputs['color']))
                _, pred_depth = disp_to_depth(outputs_depth[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

                outputs_norm = self.models['norm'](self.models['encoder_norm'](inputs['color']))
                pred_ang = (outputs_norm[("disp", 0)] - 0.5) * 2 * np.pi

                pred_depth = F.interpolate(pred_depth, [gt_height, gt_width], mode='bilinear', align_corners=True)
                pred_ang = F.interpolate(pred_ang, [gt_height, gt_width], mode='bilinear', align_corners=True)
                color_gtsize = F.interpolate(inputs['color'], [gt_height, gt_width], mode='bilinear', align_corners=True)

                pred_depth_numgradx, pred_depth_numgrady = self.sfnormOptimizer.get_depth_numgrad(pred_depth, issharp=False)
                pred_depth_anggradx, pred_depth_anggrady = self.sfnormOptimizer.ang2grad(pred_ang, inputs['K'], pred_depth)
                prednormx, prednormy = self.sfnormOptimizer.ang2dirs(pred_ang, inputs['K'])
                pred_norm = self.sfnormOptimizer.ang2normal(pred_ang, inputs['K'])

                tensor2rgb(color_gtsize, ind=vind).show()
                tensor2rgb((pred_norm + 1) / 2, ind=0).show()
                tensor2semantic(inputs['semanticspred'], ind=0).show()
                tensor2grad(pred_depth_numgradx, percentile=90, viewind=vind).show()
                tensor2grad(pred_depth_anggradx, percentile=90, viewind=vind).show()
                tensor2grad(pred_depth_numgrady, percentile=95, viewind=vind).show()
                tensor2grad(pred_depth_anggrady, percentile=95, viewind=vind).show()


                tensor2disp(pred_ang[:, 0:1, :, :] - minang, vmax=maxang, ind=vind).show()
                tensor2disp(pred_ang[:, 1:2, :, :] - minang, vmax=maxang, ind=vind).show()
                tensor2disp(1 / pred_depth, vmax=0.3, ind=vind).show()




                depthmapnp = inputs['depthgt'][0, 0].cpu().numpy()
                vallidarmask = depthmapnp > 0
                xx, yy = np.meshgrid(range(gt_width), range(gt_height), indexing='xy')
                k = inputs['K'][0].cpu().numpy()

                gtpts3d = np.stack([xx[vallidarmask] * depthmapnp[vallidarmask], yy[vallidarmask] * depthmapnp[vallidarmask], depthmapnp[vallidarmask], np.ones(np.sum(vallidarmask))], axis=0)
                gtpts3d = (np.linalg.inv(k) @ gtpts3d).T

                cm = plt.get_cmap('magma')
                colors = cm(1 / depthmapnp[vallidarmask] * 5)
                plt.figure()
                plt.imshow(tensor2rgb(color_gtsize, ind=0))
                plt.scatter(xx[vallidarmask], yy[vallidarmask], s=0.1, c=colors[:, 0:3])

                fx = k[0, 0]
                bx = k[0, 2]
                fy = k[1, 1]
                by = k[1, 2]
                obsx = 315
                obsy = 263
                obsdepth = depthmapnp[obsy, obsx]
                obspts3d = np.stack([(obsx - bx) /fx * obsdepth, (obsy - by) / fy * obsdepth, obsdepth], axis=0)

                xrange = 1
                yrange = 1
                zrange = 1
                xlim = matlab.double([obspts3d[0] - xrange, obspts3d[0] + xrange])
                ylim = matlab.double([obspts3d[1] - yrange, obspts3d[1] + yrange])
                zlim = matlab.double([obspts3d[2] - zrange, obspts3d[2] + zrange])

                gtpts3d_vlsmx = matlab.double(gtpts3d[:, 0].tolist())
                gtpts3d_vlsmy = matlab.double(gtpts3d[:, 1].tolist())
                gtpts3d_vlsmz = matlab.double(gtpts3d[:, 2].tolist())

                eng.eval('figure()', nargout=0)
                eng.scatter3(gtpts3d_vlsmx, gtpts3d_vlsmy, gtpts3d_vlsmz, 5, 'g', 'filled', nargout=0)
                eng.eval('hold on', nargout=0)
                eng.scatter3(matlab.double(obspts3d[0:1].tolist()), matlab.double(obspts3d[1:2].tolist()), matlab.double(obspts3d[2:3].tolist()), 10, 'r', 'filled', nargout=0)
                eng.eval('axis equal', nargout=0)
                eng.xlim(xlim, nargout=0)
                eng.ylim(ylim, nargout=0)
                eng.zlim(zlim, nargout=0)



        mean_errors = np.array(errors).mean(0)
        print("\nCurrent Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

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
