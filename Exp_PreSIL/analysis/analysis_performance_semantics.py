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

        self.models["encoder_norm"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder_norm"].to(self.device)
        self.models["norm"] = DepthDecoder(self.models["encoder_norm"].num_ch_enc, num_output_channels=3)
        self.models["norm"].to(self.device)

        self.set_dataset()

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.minabsrel = 1e10
        self.maxa1 = -1e10

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()
        self.semancat = [5, 6, 7] # poles

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.getcwd(), "splits", self.opt.split, "{}_files.txt")
        val_filenames = readlines(fpath.format("test"))

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, semanticspred_path=self.opt.semanticspred_path
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

        modelnames = ['intconstrainWallPoleBs', 'intconstrainWallPole', 'intconstrainWall', 'intconstrainPole', 'intconstrainPole2', 'intconstrainPole3', 'intconstrainPole4', 'intconstrainPole5']

        for k in range(len(modelnames)):
            vlsroot = os.path.join(self.opt.vlsfold, modelnames[k])
            os.makedirs(vlsroot, exist_ok=True)

            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
            self.models["encoder"].to(self.device)
            self.models["depth"] = DepthDecoder(self.models["encoder"].num_ch_enc, num_output_channels=1)
            self.models["depth"].to(self.device)

            models_to_load = ['encoder', 'depth']
            for n in models_to_load:
                path = os.path.join(self.opt.load_weights_folder, modelnames[k],'models', 'best_a1_models', "{}.pth".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

            errors = list()
            count = 0
            with torch.no_grad():
                for batch_idx, inputs in enumerate(self.val_loader):
                    for key, ipt in inputs.items():
                        if not key == 'tag':
                            inputs[key] = ipt.to(self.device)

                    _, _, gt_height, gt_width = inputs['depthgt'].shape

                    semanmask = torch.zeros_like(inputs['semanticspred'])

                    for kk in torch.unique(inputs['semanticspred']):
                        if kk in self.semancat:
                            semanmask[inputs['semanticspred'] == kk] = 1

                    outputs_depth = self.models['depth'](self.models['encoder'](inputs['color']))
                    _, pred_depth = disp_to_depth(outputs_depth[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                    pred_depth = pred_depth * self.STEREO_SCALE_FACTOR
                    pred_depth = F.interpolate(pred_depth, [gt_height, gt_width], mode='bilinear', align_corners=True)

                    gt_depth = inputs['depthgt'][0, 0, :, :].cpu().numpy()
                    gt_height, gt_width = gt_depth.shape
                    cur_pred_depth = pred_depth[0,0,:,:].cpu().numpy()

                    semanmasknp = semanmask[0,0,:,:].cpu().numpy()
                    mask = np.logical_and(gt_depth > self.MIN_DEPTH, gt_depth < self.MAX_DEPTH)
                    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                     0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
                    crop_mask = np.zeros(mask.shape)
                    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)
                    mask = np.logical_and(mask, semanmasknp == 1)

                    if np.sum(mask) == 0:
                        continue

                    cur_pred_depth = cur_pred_depth[mask]
                    gt_depth = gt_depth[mask]

                    cur_pred_depth[cur_pred_depth < self.MIN_DEPTH] = self.MIN_DEPTH
                    cur_pred_depth[cur_pred_depth > self.MAX_DEPTH] = self.MAX_DEPTH

                    errors.append(compute_errors(gt_depth, cur_pred_depth))
                    count = count + 1

            mean_errors = np.array(errors).mean(0)

            print("\nPerformance of %s:" % modelnames[k])
            print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
            print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")


if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
