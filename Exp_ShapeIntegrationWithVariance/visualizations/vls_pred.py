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
parser.add_argument("--data_path",              type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                type=str,                               help="path to kitti gt file")
parser.add_argument("--predang_path",           type=str,                               help="path to kitti gt file")
parser.add_argument("--predsemantics_path",     type=str,   default='None',             help="path to kitti semantics prediction file")
parser.add_argument("--val_gt_path",            type=str,                               help="path to validation gt file")
parser.add_argument("--model_name",             type=str,                               help="name of the model")
parser.add_argument("--split",                  type=str,                               help="train/val split to use")
parser.add_argument("--log_dir",                type=str,   default="default_logpath",    help="path to log file")
parser.add_argument("--num_layers",             type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                 type=int,   default=320,                help="input image height")
parser.add_argument("--width",                  type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                   type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                   type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",              type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",              type=float, default=100.0,              help="maximum depth")
parser.add_argument("--print_freq",             type=int,   default=50)
parser.add_argument("--val_frequency",          type=int,   default=10)
parser.add_argument("--banshuffle",             action="store_true")

# Integration options
parser.add_argument("--lam",                    type=float,   default=0.05)
parser.add_argument("--inttimes",               type=int,     default=1)
parser.add_argument("--clipvariance",           type=float,   default=5)
parser.add_argument("--maxrange",               type=float,   default=100)
parser.add_argument("--startepochs",            type=int,     default=1)
parser.add_argument("--bansemantics",           action="store_true")
parser.add_argument("--banground",              action="store_true")

# OPTIMIZATION options
parser.add_argument("--batch_size",             type=int,   default=1,                 help="batch size")
parser.add_argument("--learning_rate",          type=float, default=1e-4,               help="learning rate")
parser.add_argument("--num_epochs",             type=int,   default=20,                 help="number of epochs")
parser.add_argument("--scheduler_step_size",    type=int,   default=15,                 help="step size of the scheduler")
parser.add_argument("--load_weights_folder",    type=str,   default=None,               help="name of models to load")
parser.add_argument("--num_workers",            type=int,   default=6,                  help="number of dataloader workers")

# LOGGING options
parser.add_argument("--log_frequency",          type=int,   default=250,                help="number of batches between each tensorboard log")
parser.add_argument("--save_frequency",         type=int,   default=1,                  help="number of epochs between each save")
parser.add_argument("--vlsfold",                type=str,   default=250,                help="number of batches between each tensorboard log")

class Trainer:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

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
        from integrationModule import CRFIntegrationModule
        self.crfintmodule = CRFIntegrationModule(clipvariance=self.opt.clipvariance, maxrange=self.opt.maxrange, lam=self.opt.lam)

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

        modelnames = ['crfintwvariance_bansemantics_banground_1',
                      'crfintwvariance_bansemantics_banground_5']

        predroot = '/media/shengjie/disk1/visualization/shapeintegrationpred'

        for k in range(len(modelnames)):
            vlsroot = os.path.join(self.opt.vlsfold, modelnames[k])
            os.makedirs(vlsroot, exist_ok=True)

            orgpredroot = os.path.join(predroot, modelnames[k] + '_pred', 'org')
            intpredroot = os.path.join(predroot, modelnames[k] + '_pred', 'int')
            os.makedirs(orgpredroot, exist_ok=True)
            os.makedirs(intpredroot, exist_ok=True)

            self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
            self.models["encoder"].to(self.device)
            self.models["depth"] = MultiModalityDecoder(self.models["encoder"].num_ch_enc, modalities=['depth', 'variance'], nchannelout=[1, 1], additionalblocks=1)
            self.models["depth"].to(self.device)

            models_to_load = ['encoder', 'depth']
            for n in models_to_load:
                path = os.path.join(self.opt.load_weights_folder, modelnames[k], 'models', 'best_a1_models', "{}.pth".format(n))
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

            self.set_eval()

            with torch.no_grad():
                for batch_idx, inputs in enumerate(self.val_loader):
                    for key, ipt in inputs.items():
                        if not key == 'tag':
                            inputs[key] = ipt.to(self.device)

                    _, _, gt_height, gt_width = inputs['depthgt'].shape

                    outputs = dict()
                    encoder_feature = self.models['encoder'](inputs['color_aug'])
                    outputs.update(self.models['depth'](encoder_feature))

                    scaled_disp, pred_depth = disp_to_depth(outputs[('depth', 0)], min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
                    pred_depth = F.interpolate(pred_depth, [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True)
                    pred_depth = pred_depth * self.STEREO_SCALE_FACTOR
                    pred_variance = F.interpolate(outputs[('variance', 0)], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True) * (self.opt.clipvariance + 1)

                    pred_ang = torch.cat([inputs['angh'], inputs['angv']], dim=1).contiguous()
                    pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang).contiguous()
                    normfromang = self.sfnormOptimizer.ang2normal(ang=pred_ang, intrinsic=inputs['K'])
                    singularnorm = self.sfnormOptimizer.ang2edge(ang=pred_ang, intrinsic=inputs['K'])

                    semanticspred = torch.ones_like(inputs['depthgt']).int().contiguous()
                    semanedgemask = torch.ones_like(semanticspred)
                    # exclude the top area
                    semanedgemask[:, :, 0:int(0.40810811 * self.opt.crph), :] = 0
                    # exclude distant places
                    semanedgemask = semanedgemask * (pred_depth < 30).int()
                    semanedgemask = semanedgemask * (normfromang[:, 1, :, :].unsqueeze(1) < 0.8).int()
                    # exclup singular value of the normal
                    semanedgemask = semanedgemask * (1 - singularnorm).int()

                    pred_depth_integrated = self.crfintmodule(pred_log, semanticspred, semanedgemask, pred_variance, pred_depth, times=1)

                    orgnorm = self.sfnormOptimizer.depth2norm(depthMap=pred_depth, intrinsic=inputs['K'])
                    integratednorm = self.sfnormOptimizer.depth2norm(depthMap=pred_depth_integrated, intrinsic=inputs['K'])

                    figd1 = tensor2disp(1 / pred_depth, vmax=0.15, ind=0)
                    figd2 = tensor2disp(1 / pred_depth_integrated, vmax=0.15, ind=0)
                    fign1 = tensor2rgb((1 + orgnorm) / 2, ind=0)
                    fign2 = tensor2rgb((1 + integratednorm) / 2, ind=0)

                    figu = np.concatenate([np.array(figd1), np.array(fign1)], axis=1)
                    figd = np.concatenate([np.array(figd2), np.array(fign2)], axis=1)
                    fig = np.concatenate([figu, figd], axis=0)

                    figname = "{}_{}.png".format(inputs['tag'][0].split(' ')[0].split('/')[1], inputs['tag'][0].split(' ')[1])
                    pil.fromarray(fig).save(os.path.join(vlsroot, figname))

                    pred_depthnp = (pred_depth[0,0].cpu().numpy() * 256.0).astype(np.uint16)
                    pil.fromarray(pred_depthnp).save(os.path.join(orgpredroot, figname))
                    pred_depth_integratednp = (pred_depth_integrated[0,0].cpu().numpy() * 256.0).astype(np.uint16)
                    pil.fromarray(pred_depth_integratednp).save(os.path.join(intpredroot, figname))
if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
