from __future__ import absolute_import, division, print_function

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
parser.add_argument("--vlsfold",                    type=str)


# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=1,                 help="batch size")
parser.add_argument("--load_weights_folder",  type=str,   default=None,              help="name of models to load")

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.models = {}
        self.parameters_to_train = []

        self.device = "cuda"
        self.set_dataset()

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

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

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, predang_path='/home/shengjie/Documents/Data/Kitti/kitti_angpred'
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

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()

        modelname = 'kitti_depth_l1_baseline_semidense'

        vlsroot = '/media/shengjie/disk1/visualization/shapeintegrationpred/kitti_depth_l1_baseline_semidense'
        os.makedirs(vlsroot, exist_ok=True)

        predroot = '/media/shengjie/disk1/visualization/shapeintegrationpred/kitti_depth_l1_baseline_semidense_pred'
        os.makedirs(predroot, exist_ok=True)

        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder"].to(self.device)
        self.models["depth"] = DepthDecoder(self.models["encoder"].num_ch_enc, num_output_channels=1)
        self.models["depth"].to(self.device)

        models_to_load = ['encoder', 'depth']
        for n in models_to_load:
            path = os.path.join(self.opt.load_weights_folder, modelname, 'models', 'best_a1_models', "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                _, _, gt_height, gt_width = inputs['depthgt'].shape

                outputs_depth = self.models['depth'](self.models['encoder'](inputs['color']))
                _, pred_depth = disp_to_depth(outputs_depth[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_depth = pred_depth * self.STEREO_SCALE_FACTOR
                pred_depth = F.interpolate(pred_depth, [gt_height, gt_width], mode='bilinear', align_corners=True)
                pred_norm = self.sfnormOptimizer.depth2norm(intrinsic=inputs['K'], depthMap=pred_depth)

                figd1 = tensor2disp(1 / pred_depth, vmax=0.15, ind=0)
                fign1 = tensor2rgb((1 + pred_norm) / 2, ind=0)
                fig = np.concatenate([figd1, fign1], axis=1)

                figname = "{}_{}.png".format(inputs['tag'][0].split(' ')[0].split('/')[1], inputs['tag'][0].split(' ')[1])
                pil.fromarray(fig).save(os.path.join(vlsroot, figname))

                pred_depthnp = (pred_depth[0, 0].cpu().numpy() * 256.0).astype(np.uint16)
                pil.fromarray(pred_depthnp).save(os.path.join(predroot, figname))

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
