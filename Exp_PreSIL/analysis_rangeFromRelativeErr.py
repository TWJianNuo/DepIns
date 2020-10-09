from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

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

from Exp_PreSIL.dataloader_kitti import KittiDataset

import networks

import time
import json

from layers import *
from networks import *

import argparse

default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",                  type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                    type=str,                               help="path to kitti gt file")
parser.add_argument("--predang_path",               type=str,                               help="path to kitti gt file")
parser.add_argument("--semanticspred_path",         type=str,   default='None',             help="path to kitti semantics prediction file")
parser.add_argument("--val_gt_path",                type=str,                               help="path to validation gt file")
parser.add_argument("--split",                      type=str,                               help="train/val split to use")
parser.add_argument("--num_layers",                 type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                     type=int,   default=320,                help="input image height")
parser.add_argument("--width",                      type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                       type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                       type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",                  type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",                  type=float, default=100.0,              help="maximum depth")
parser.add_argument("--load_angweights_folder",     type=str,                               help="path to kitti gt file")
parser.add_argument("--load_depthweights_folder",   type=str,                               help="path to kitti gt file")
parser.add_argument("--load_angErr_folder",     type=str,                               help="path to kitti gt file")
parser.add_argument("--load_depthErr_folder",   type=str,                               help="path to kitti gt file")

# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=12,                 help="batch size")
parser.add_argument("--num_workers",                type=int,   default=6,                  help="number of dataloader workers")

@numba.jit(nopython=True, parallel=True)
def dynamicReceptiveField(height, width, err_shapenph, err_shapenpv, err_depthnph, err_depthnpv, sw, sh):
    hrecord = np.zeros((height, width))
    vrecord = np.zeros((height, width))

    for m in range(height):
        for n in range(width):
            hgood = 0
            vgood = 0

            intl = 0
            for lx in range(sw):
                ckx = n - (lx + 1)
                if ckx >= 0:
                    intl = intl + err_shapenph[m, ckx]
                    refl = err_depthnph[m, n] + err_depthnph[m, n - (lx + 1)]
                    if intl <= refl:
                        hgood = hgood + 1
                    else:
                        break

            intr = 0
            for rx in range(sw):
                ckx = n + (rx + 1)
                if ckx < width:
                    intr = intr + err_shapenph[m, ckx - 1]
                    refr = err_depthnph[m, n] + err_depthnph[m, n + (rx + 1)]
                    if intr <= refr:
                        hgood = hgood + 1
                    else:
                        break

            intu = 0
            for ru in range(sh):
                cky = m - (ru + 1)
                if cky >= 0:
                    intu = intu + err_shapenpv[cky, n]
                    refu = err_depthnpv[m, n] + err_depthnpv[m - (ru + 1), n]
                    if intu <= refu:
                        vgood = vgood + 1
                    else:
                        break

            intd = 0
            for rd in range(sh):
                cky = m + (rd + 1)
                if cky < height:
                    intd = intd + err_shapenpv[cky - 1, n]
                    refd = err_depthnpv[m, n] + err_depthnpv[m + (rd + 1), n]
                    if intd <= refd:
                        vgood = vgood + 1
                    else:
                        break

            hrecord[m,n] = hgood
            vrecord[m,n] = vgood
    return hrecord, vrecord

class Trainer:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.device = "cuda"

        self.depthmodels = {}
        self.depthmodels["depthencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=False, num_input_channels=3)
        self.depthmodels["depthdecoder"] = DepthDecoder(self.depthmodels["depthencoder"].num_ch_enc, num_output_channels=1)
        self.depthmodels["depthencoder"].to(self.device)
        self.depthmodels["depthdecoder"].to(self.device)
        self.load_model(weightFolder=self.opt.load_depthweights_folder, encoderName='depthencoder',
                        decoderName='depthdecoder', encoder=self.depthmodels["depthencoder"],
                        decoder=self.depthmodels["depthdecoder"])
        for m in self.depthmodels.values():
            m.eval()

        self.angmodels = {}
        self.angmodels["angencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=False)
        self.angmodels["angdecoder"] = DepthDecoder(self.angmodels["angencoder"].num_ch_enc, num_output_channels=2)
        self.angmodels["angencoder"].to(self.device)
        self.angmodels["angdecoder"].to(self.device)
        self.load_model(weightFolder=self.opt.load_angweights_folder, encoderName='angencoder',
                        decoderName='angdecoder', encoder=self.angmodels["angencoder"],
                        decoder=self.angmodels["angdecoder"])
        for m in self.angmodels.values():
            m.eval()

        self.depthErrmodels = {}
        self.depthErrmodels["dErrencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=False, num_input_channels=4)
        self.depthErrmodels["dErrdecoder"] = DepthDecoder(self.depthErrmodels["dErrencoder"].num_ch_enc, num_output_channels=2)
        self.depthErrmodels["dErrencoder"].to(self.device)
        self.depthErrmodels["dErrdecoder"].to(self.device)
        self.load_model(weightFolder=self.opt.load_depthErr_folder, encoderName='dErrencoder',
                        decoderName='dErrdecoder', encoder=self.depthErrmodels["dErrencoder"],
                        decoder=self.depthErrmodels["dErrdecoder"])
        for m in self.depthErrmodels.values():
            m.eval()

        self.angErrmodels = {}
        self.angErrmodels["aErrencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=False, num_input_channels=5)
        self.angErrmodels["aErrdecoder"] = DepthDecoder(self.angErrmodels["aErrencoder"].num_ch_enc, num_output_channels=2)
        self.angErrmodels["aErrencoder"].to(self.device)
        self.angErrmodels["aErrdecoder"].to(self.device)
        self.load_model(weightFolder=self.opt.load_angErr_folder, encoderName='aErrencoder',
                        decoderName='aErrdecoder', encoder=self.angErrmodels["aErrencoder"],
                        decoder=self.angErrmodels["aErrdecoder"])
        for m in self.angErrmodels.values():
            m.eval()

        print("Training is using:\t", self.device)

        self.set_dataset()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "{}_files.txt")
        test_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "test_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(test_fpath)

        train_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, train_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=True, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path
        )

        val_dataset = KittiDataset(
            self.opt.data_path, self.opt.val_gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path
        )

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                outputs_depth = self.depthmodels['depthdecoder'](self.depthmodels['depthencoder'](inputs['color']))
                outputs_ang = self.angmodels['angdecoder'](self.angmodels['angencoder'](inputs['color']))
                outputs_dErr = self.depthErrmodels["dErrdecoder"](self.depthErrmodels["dErrencoder"](torch.cat([inputs['color'], outputs_depth['disp', 0]], dim=1)))
                outputs_aErr = self.angErrmodels["aErrdecoder"](self.angErrmodels["aErrencoder"](torch.cat([inputs['color'], outputs_ang['disp', 0]], dim=1)))

                pred_depth = F.interpolate(outputs_depth['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                _, pred_depth = disp_to_depth(pred_depth, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
                pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

                err_depth_act = F.interpolate(outputs_dErr['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                err_shape_act = F.interpolate(outputs_aErr['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                rgb_resized = F.interpolate(inputs['color'], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)

                pred_ang = F.interpolate(outputs_ang['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                pred_ang = (pred_ang - 0.5) * 2 * np.pi
                pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang)
                minang = - np.pi / 3 * 2
                maxang = 2 * np.pi - np.pi / 3 * 2
                # tensor2disp(pred_ang[:, 0:1, :, :] - minang, vmax=maxang, ind=0).show()
                # tensor2disp(outputs_depth['disp', 0], vmax=0.1, ind=0).show()

                err_depth = self.sfnormOptimizer.act2err(errpred=err_depth_act, intrinsic=inputs['K'])
                err_shape = self.sfnormOptimizer.act2err(errpred=err_shape_act, intrinsic=inputs['K'])

                # tensor2disp(err_depth[:, 0:1, :, :], vmax=np.pi / 2, ind=0).show()
                # tensor2disp(err_shape[:, 0:1, :, :], vmax=np.pi / 2, ind=0).show()

                err_depthnph = err_depth[0, 0, :, :].cpu().numpy()
                err_depthnpv = err_depth[0, 1, :, :].cpu().numpy()
                err_shapenph = err_shape[0, 0, :, :].cpu().numpy()
                err_shapenpv = err_shape[0, 1, :, :].cpu().numpy()

                hgood, vgood = dynamicReceptiveField(height=self.opt.height, width=self.opt.width, err_shapenph=err_shapenph, err_shapenpv=err_shapenpv, err_depthnph=err_depthnph, err_depthnpv=err_depthnpv, sw=self.opt.width, sh=self.opt.height)
                plt.figure()
                plt.imshow(tensor2disp(torch.from_numpy(hgood).unsqueeze(0).unsqueeze(0), vmax=50, ind=0))
                plt.colorbar()
                plt.title('Horizontal Direction Receptive field Range')
                plt.figure()
                plt.imshow(tensor2disp(torch.from_numpy(vgood).unsqueeze(0).unsqueeze(0), vmax=50, ind=0))
                plt.colorbar()
                plt.title('Vertical Direction Receptive field Range')


                sw = 100
                sh = 50

                logh = pred_log[0, 0, :, :].detach().cpu().numpy()
                logv = pred_log[0, 1, :, :].detach().cpu().numpy()
                pred_depthnp = pred_depth[0, 0, :, :].detach().cpu().numpy()

                depthgtnp = inputs['depthgt'][0, 0, :, :].detach().cpu().numpy()
                nn, mm = np.meshgrid(range(self.opt.crpw), range(self.opt.crph), indexing='xy')
                mask = np.zeros([self.opt.crph, self.opt.crpw])
                mask[int(0.40810811 * self.opt.crph):int(0.99189189 * self.opt.crph), int(0.03594771 * self.opt.crpw):int(0.96405229 * self.opt.crpw)] = 1
                mask = mask == 1
                mask = mask * (depthgtnp > 0)
                nn = nn[mask]
                mm = mm[mask]
                goodpts = list()
                badpts = list()
                goodptsl = list()
                badptsl = list()
                for kk in range(6):
                    rndindex = np.random.randint(low=0, high=nn.shape[0])
                    n = nn[rndindex]
                    m = mm[rndindex]

                    intl = 0
                    for lx in range(sw):
                        ckx = n - (lx + 1)
                        if ckx >= 0:
                            intl = intl + err_shapenph[m, ckx]
                            refl = err_depthnph[m, n] + err_depthnph[m, n - (lx + 1)]
                            if intl <= refl:
                                goodpts.append(np.array([ckx, m]))
                            else:
                                badpts.append(np.array([ckx, m]))

                    intr = 0
                    for rx in range(sw):
                        ckx = n + (rx + 1)
                        if ckx < self.opt.crpw:
                            intr = intr + err_shapenph[m, ckx - 1]
                            refr = err_depthnph[m, n] + err_depthnph[m, n + (rx + 1)]
                            if intr <= refr:
                                goodpts.append(np.array([ckx, m]))
                            else:
                                badpts.append(np.array([ckx, m]))

                    intu = 0
                    for ru in range(sh):
                        cky = m - (ru + 1)
                        if cky >= 0:
                            intu = intu + err_shapenpv[cky, n]
                            refu = err_depthnpv[m, n] + err_depthnpv[m - (ru + 1), n]
                            if intu <= refu:
                                goodpts.append(np.array([n, cky]))
                            else:
                                badpts.append(np.array([n, cky]))

                    intd = 0
                    for rd in range(sh):
                        cky = m + (rd + 1)
                        if cky < self.opt.crph:
                            intd = intd + err_shapenpv[cky - 1, n]
                            refd = err_depthnpv[m, n] + err_depthnpv[m + (rd + 1), n]
                            if intd <= refd:
                                goodpts.append(np.array([n, cky]))
                            else:
                                badpts.append(np.array([n, cky]))

                    ## ================================================= ##
                    intl = 0
                    for lx in range(sw):
                        ckx = n - (lx + 1)
                        if ckx >= 0:
                            intl = intl + logh[m, ckx]
                            refl = np.log(pred_depthnp[m, n]) - np.log(pred_depthnp[m, n - (lx + 1)])
                            if mask[m, n - (lx + 1)]:
                                gtl = np.log(depthgtnp[m, n]) - np.log(depthgtnp[m, n - (lx + 1)])
                                if np.abs(intl - gtl) < np.abs(refl - gtl):
                                    goodptsl.append(np.array([ckx, m]))
                                else:
                                    badptsl.append(np.array([ckx, m]))

                    intr = 0
                    for rx in range(sw):
                        ckx = n + (rx + 1)
                        if ckx < self.opt.crpw:
                            intr = intr - logh[m, ckx - 1]
                            refr = np.log(pred_depthnp[m, n]) - np.log(pred_depthnp[m, n + (rx + 1)])
                            if mask[m, n + (rx + 1)]:
                                gtr = np.log(depthgtnp[m, n]) - np.log(depthgtnp[m, n + (rx + 1)])
                                if np.abs(intr - gtr) < np.abs(refr - gtr):
                                    goodptsl.append(np.array([ckx, m]))
                                else:
                                    badptsl.append(np.array([ckx, m]))

                    intu = 0
                    for ru in range(sh):
                        cky = m - (ru + 1)
                        if cky >= 0:
                            intu = intu + logv[cky, n]
                            refu = np.log(pred_depthnp[m, n]) - np.log(pred_depthnp[m - (ru + 1), n])
                            if mask[m - (ru + 1), n]:
                                gtu = np.log(depthgtnp[m, n]) - np.log(depthgtnp[m - (ru + 1), n])
                                if np.abs(intu - gtu) < np.abs(refu - gtu):
                                    goodptsl.append(np.array([n, cky]))
                                else:
                                    badptsl.append(np.array([n, cky]))

                    intd = 0
                    for rd in range(sh):
                        cky = m + (rd + 1)
                        if cky < self.opt.crph:
                            intd = intd - logv[cky - 1, n]
                            refd = np.log(pred_depthnp[m, n]) - np.log(pred_depthnp[m + (rd + 1), n])
                            if mask[m + (rd + 1), n]:
                                gtd = np.log(depthgtnp[m, n]) - np.log(depthgtnp[m + (rd + 1), n])
                                if np.abs(intd - gtd) < np.abs(refd - gtd):
                                    goodptsl.append(np.array([n, cky]))
                                else:
                                    badptsl.append(np.array([n, cky]))


                goodpts = np.array(goodpts)
                badpts = np.array(badpts)
                goodptsl = np.array(goodptsl)
                badptsl = np.array(badptsl)

                fig, axs = plt.subplots(2, figsize=(16,8))
                axs[0].imshow(tensor2rgb(rgb_resized, ind=0))
                axs[0].scatter(goodpts[:,0], goodpts[:,1], s=0.1, c='r')
                axs[0].scatter(badpts[:,0], badpts[:,1], s=0.1, c='b')
                axs[1].imshow(tensor2rgb(rgb_resized, ind=0))
                axs[1].scatter(goodpts[:,0], goodpts[:,1], s=0.1, c='c')
                axs[1].scatter(badpts[:,0], badpts[:,1], s=0.1, c='c')
                axs[1].scatter(badptsl[:,0], badptsl[:,1], s=0.7, c='b')
                axs[1].scatter(goodptsl[:,0], goodptsl[:,1], s=0.7, c='r')
                plt.savefig(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/analysis_dynamicReceptiveField', '{}.png'.format(batch_idx)))
                plt.close()

    def load_model(self, weightFolder, encoderName, decoderName, encoder, decoder):
        """Load model(s) from disk
        """
        assert os.path.isdir(weightFolder), "Cannot find folder {}".format(weightFolder)
        print("loading model from folder {}".format(weightFolder))

        path = os.path.join(weightFolder, "encoder.pth")
        print("Loading {} weights...".format(encoderName))
        model_dict = encoder.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        encoder.load_state_dict(model_dict)

        path = os.path.join(weightFolder, "depth.pth")
        print("Loading {} weights...".format(decoderName))
        model_dict = decoder.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        decoder.load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
