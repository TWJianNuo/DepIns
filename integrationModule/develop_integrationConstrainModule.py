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
parser.add_argument("--load_angweights_folder", type=str,   default=None,               help="name of models to load")

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

        self.models["angencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["angencoder"].to(self.device)

        self.models["angdecoder"] = DepthDecoder(self.models["encoder_depth"].num_ch_enc, num_output_channels=2)
        self.models["angdecoder"].to(self.device)
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
        val_filenames = readlines(fpath.format("train"))

        self.val_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, instancepred_path=self.opt.instancepred_path)

        self.val_loader = DataLoader(
            self.val_dataset, self.opt.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True)

        self.val_num = self.val_dataset.__len__()

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

                _, _, gt_height, gt_width = inputs['depthgt'].shape

                outputs_depth = self.models['depth'](self.models['encoder_depth'](inputs['color']))
                _, pred_depth = disp_to_depth(outputs_depth[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
                pred_depth = pred_depth * self.STEREO_SCALE_FACTOR
                pred_depth = F.interpolate(pred_depth, [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=False)

                outputs_ang = self.models['angdecoder'](self.models['angencoder'](inputs['color']))
                angfromang = (outputs_ang[("disp", 0)] - 0.5) * 2 * np.pi
                angfromang = F.interpolate(angfromang, [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=False)
                angfromdepth = self.sfnormOptimizer.depth2ang_log(depthMap=pred_depth, intrinsic=inputs['K'])

                logfromang = self.sfnormOptimizer.ang2log(intrinsic=inputs["K"], ang=angfromang)
                logfromdepth = self.sfnormOptimizer.ang2log(intrinsic=inputs["K"], ang=angfromdepth)

                mask = torch.zeros_like(pred_depth, dtype=torch.int)
                mask[:, :, int(0.40810811 * self.opt.crph): -1, :] = 1
                mask = mask.contiguous()
                mask = (mask * inputs['instancepred'] > 0).int()
                instancepred = inputs['instancepred'].int().contiguous()

                import shapeintegration_cuda
                constrainout = torch.zeros_like(pred_depth)
                constraingradin = torch.ones_like(pred_depth)
                constraingradout = torch.zeros_like(pred_depth)
                counts = torch.zeros_like(pred_depth)
                shapeintegration_cuda.shapeIntegration_crf_constrain_forward(logfromang, instancepred, mask, pred_depth, constrainout, counts, self.opt.crph, self.opt.crpw, 1)
                shapeintegration_cuda.shapeIntegration_crf_constrain_backward(logfromang, instancepred, mask, pred_depth, constraingradin, counts, constraingradout, self.opt.crph, self.opt.crpw, 1)

                plt.figure()
                plt.imshow(constrainout.detach().cpu().numpy()[0,0,:,:])

                plt.figure()
                plt.imshow(counts.detach().cpu().numpy()[0,0,:,:])

                plt.figure()
                plt.imshow(torch.abs(constraingradout).detach().cpu().numpy()[0,0,:,:])

                minang = - np.pi / 3 * 2
                maxang = 2 * np.pi - np.pi / 3 * 2
                vind = 0
                tensor2disp(angfromang[:, 0:1, :, :] - minang, vmax=maxang, ind=vind).show()
                tensor2disp(angfromdepth[:, 0:1, :, :] - minang, vmax=maxang, ind=vind).show()
                tensor2disp(angfromang[:, 1:2, :, :] - minang, vmax=maxang, ind=vind).show()
                tensor2disp(angfromdepth[:, 1:2, :, :] - minang, vmax=maxang, ind=vind).show()
                tensor2disp(mask, vmax=1, ind=vind).show()
                figname = inputs['tag'][0].split(' ')[0].split('/')[1] + '_' + inputs['tag'][0].split(' ')[1]
                tensor2disp(inputs['instancepred'] > 0, vmax=1, ind=0).show()
                tensor2rgb(inputs['color'], ind=0).show()

    def val_forward_backward_torch(self):
        self.set_eval()

        for batch_idx in range(self.val_dataset.__len__()):
            # batch_idx = self.val_dataset.filenames.index('2011_09_26/2011_09_26_drive_0052_sync 0000000026 l')
            batch_idx = 30
            inputs = self.val_dataset.__getitem__(batch_idx)

            for key, ipt in inputs.items():
                if not key == 'tag':
                    inputs[key] = ipt.to(self.device).unsqueeze(0)

            with torch.no_grad():
                outputs_depth = self.models['depth'](self.models['encoder_depth'](inputs['color']))
                outputs_ang = self.models['angdecoder'](self.models['angencoder'](inputs['color']))

            pred_depth = F.interpolate(outputs_depth['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=False)
            _, pred_depth = disp_to_depth(pred_depth, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
            pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

            pred_ang = F.interpolate(outputs_ang['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=False)
            pred_ang = (pred_ang - 0.5) * 2 * np.pi
            pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang)
            pred_log = pred_log.contiguous()

            mask = torch.zeros_like(pred_depth, dtype=torch.int)
            mask[:,:,int(0.40810811 * self.opt.crph):, :] = 1
            mask = mask.contiguous()
            instancepred = inputs['instancepred'].int().contiguous()

            from integrationModule import IntegrationConstrainFunction
            integrateConstrainFunction = IntegrationConstrainFunction.apply

            # xx, yy = np.meshgrid(range(self.opt.crpw), range(self.opt.crph), indexing='xy')
            # valxx = xx[instancepred.detach().cpu().numpy()[0,0,:,:] > 0]
            # valyy = yy[instancepred.detach().cpu().numpy()[0,0,:,:] > 0]
            #
            # rndind = np.random.randint(0, len(valxx))
            # c = 0
            # m = valyy[rndind]
            # n = valxx[rndind]
            #
            # import shapeintegration_cuda
            # constrainout = torch.zeros_like(pred_depth)
            # constraingradin = torch.zeros_like(pred_depth)
            # constraingradout = torch.zeros_like(pred_depth)
            # counts = torch.zeros_like(pred_depth)
            # shapeintegration_cuda.shapeIntegration_crf_constrain_forward(pred_log, instancepred, mask, pred_depth, constrainout, counts, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            # depthingrad, valcount = self.validate_forward(pred_log, instancepred, mask, pred_depth, self.opt.crph, self.opt.crpw, c, m, n, constrainout[c, 0, m, n], counts[c, 0, m, n])
            #
            # constraingradin[c, 0, m, n] = 1
            # shapeintegration_cuda.shapeIntegration_crf_constrain_backward(pred_log, instancepred, mask, pred_depth, counts, constraingradin, constraingradout, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            #
            # assert torch.abs(constraingradout - depthingrad).max() < 0.01

            # Numerical Test for gradient of depth
            # eps = 1e-3
            # pred_depth1 = pred_depth.clone()
            # pred_depth1[c, 0, m, n] = pred_depth1[c, 0, m, n] + eps
            # pred_depth2 = pred_depth.clone()
            # pred_depth2[c, 0, m, n] = pred_depth2[c, 0, m, n] - eps
            # opteddepth1 = integrateConstrainFunction(pred_log, instancepred, mask, pred_depth1, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            # opteddepth2 = integrateConstrainFunction(pred_log, instancepred, mask, pred_depth2, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            # numgraddepth = (torch.sum(opteddepth1 - opteddepth2)) / eps / 2
            #
            # pred_depth_ref = pred_depth.clone()
            # pred_depth_ref.requires_grad = True
            # optimi = torch.optim.SGD([pred_depth_ref], lr=1e-3)
            # optimi.zero_grad()
            # opteddepth = integrateConstrainFunction(pred_log, instancepred, mask, pred_depth_ref, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            # torch.sum(opteddepth).backward()
            #
            # ratio = numgraddepth / pred_depth_ref.grad[c, 0, m, n]

            mask = mask * (instancepred > 0).int()
            pred_depth.requires_grad = True
            optimi = torch.optim.SGD([pred_depth], lr=1e3)
            for k in range(100):
                opteddepth = integrateConstrainFunction(pred_log, instancepred, mask, pred_depth, self.opt.crph, self.opt.crpw, self.opt.batch_size)
                loss = torch.mean(opteddepth)
                optimi.zero_grad()
                loss.backward()
                optimi.step()
                print(loss)

                opteddepth = torch.clamp(opteddepth, max=1)
                plt.figure()
                plt.imshow(opteddepth.detach().cpu().numpy()[0,0,:,:])

                plt.figure()
                plt.imshow(tensor2disp(1/pred_depth, vmax=0.15, ind=0))


    def validate_forward(self, log, semantics, mask, depthin, height, width, c, m, n, integrateddiff, integratedcount):
        # depthin.requires_grad = True
        # adaptor = torch.optim.SGD([depthin], lr=1e-4)
        # adaptor.zero_grad()
        depthingrad = torch.zeros_like(depthin)
        semancat = semantics[c][0][m][n]
        depthh = 0
        depthv = 0

        if mask[c][0][m][n] == 1:
            breakpos = False
            breakneg = False
            countpos = 0
            countneg = 0
            intpos = 0
            intneg = 0

            for ii in range(width * 2):
                if breakpos and breakneg:
                    break
                if (ii & 1) == False:
                    if (breakpos): continue
                    sn = int(n + int(ii / 2) + 1)
                    if sn >= width:
                        breakpos=True
                        continue
                    elif semancat != semantics[c][0][m][sn]:
                        breakpos=True
                        continue
                    elif mask[c][0][m][sn] == 0:
                        breakpos=True
                        continue
                    else:
                        intpos += log[c][0][m][sn - 1]
                        countpos = countpos + 1
                else:
                    if breakneg:
                        continue
                    sn = int(n - int(ii / 2) - 1)
                    if sn < 0:
                        breakneg=True
                        continue
                    elif semancat != semantics[c][0][m][sn]:
                        breakneg=True
                        continue
                    elif mask[c][0][m][sn] == 0:
                        breakneg=True
                        continue
                    else:
                        intneg -= log[c][0][m][sn]
                        countneg = countneg + 1

            for ii in range(countpos, 0, -1):
                sn = n + ii
                depthh += torch.abs(torch.exp(-intpos) * depthin[c][0][m][sn] - depthin[c][0][m][n]) / (countpos + countneg)
                if torch.exp(-intpos) * depthin[c][0][m][sn] - depthin[c][0][m][n] >= 0:
                    depthingrad[c][0][m][sn] += torch.exp(-intpos) / integratedcount
                    depthingrad[c][0][m][n] += -1 / integratedcount
                else:
                    depthingrad[c][0][m][sn] += -torch.exp(-intpos) / integratedcount
                    depthingrad[c][0][m][n] += 1 / integratedcount
                intpos -= log[c][0][m][sn-1]

            for ii in range(countneg, 0, -1):
                sn = n - ii
                depthh += torch.abs(torch.exp(-intneg) * depthin[c][0][m][sn] - depthin[c][0][m][n]) / (countpos + countneg)
                if torch.exp(-intneg) * depthin[c][0][m][sn] - depthin[c][0][m][n] >= 0:
                    depthingrad[c][0][m][sn] += torch.exp(-intneg) / integratedcount
                    depthingrad[c][0][m][n] += -1 / integratedcount
                else:
                    depthingrad[c][0][m][sn] += -torch.exp(-intneg) / integratedcount
                    depthingrad[c][0][m][n] += 1 / integratedcount
                intneg += log[c][0][m][sn]

            counth = countneg + countpos

        if mask[c][0][m][n] == 1:
            breakpos = False
            breakneg = False
            countpos = 0
            countneg = 0
            intpos = 0
            intneg = 0

            for ii in range(height * 2):
                if breakpos and breakneg:
                    break
                if (ii & 1) == False:
                    if breakpos:
                        continue
                    sm = int(m + int(ii / 2) + 1)
                    if sm >= height:
                        breakpos=True
                        continue
                    elif semancat != semantics[c][0][sm][n]:
                        breakpos=True
                        continue
                    elif mask[c][0][sm][n] == 0:
                        breakpos=True
                        continue
                    else:
                        intpos += log[c][1][sm - 1][n]
                        countpos = countpos + 1
                else:
                    if breakneg:
                        continue
                    sm = int(m - int(ii / 2) - 1)
                    if sm < 0:
                        breakneg=True
                        continue
                    elif semancat != semantics[c][0][sm][n]:
                        breakneg=True
                        continue
                    elif mask[c][0][sm][n] == 0:
                        breakneg=True
                        continue
                    else:
                        intneg -= log[c][1][sm][n]
                        countneg = countneg + 1

            for ii in range(countpos, 0, -1):
                sm = m + ii
                depthv += torch.abs(torch.exp(-intpos) * depthin[c][0][sm][n] - depthin[c][0][m][n]) / (countpos + countneg)
                if torch.exp(-intpos) * depthin[c][0][sm][n] - depthin[c][0][m][n] >= 0:
                    depthingrad[c][0][sm][n] += torch.exp(-intpos) / integratedcount
                    depthingrad[c][0][m][n] += -1 / integratedcount
                else:
                    depthingrad[c][0][sm][n] += -torch.exp(-intpos) / integratedcount
                    depthingrad[c][0][m][n] += 1 / integratedcount
                intpos -= log[c][1][sm-1][n]

            for ii in range(countneg, 0, -1):
                sm = m - ii
                depthv += torch.abs(torch.exp(-intneg) * depthin[c][0][sm][n] - depthin[c][0][m][n]) / (countpos + countneg)
                if torch.exp(-intneg) * depthin[c][0][sm][n] - depthin[c][0][m][n] >= 0:
                    depthingrad[c][0][sm][n] += torch.exp(-intneg) / integratedcount
                    depthingrad[c][0][m][n] += -1 / integratedcount
                else:
                    depthingrad[c][0][sm][n] += -torch.exp(-intneg) / integratedcount
                    depthingrad[c][0][m][n] += 1 / integratedcount
                intneg += log[c][1][sm][n]

            countv = countneg + countpos

        refdiff = depthh * counth / (counth + countv) + depthv * countv / (counth + countv)
        # refdiff.backward()

        # grad on c, m , n
        # depthin.grad[c, 0, m, n]
        # depthingrad[c, 0, m, n]
        return depthingrad, (counth + countv)
    def load_model(self):
        """Load model(s) from disk
        """
        load_depth_folder = os.path.expanduser(self.opt.load_weights_folder_depth)
        load_weights_folder = os.path.expanduser(self.opt.load_angweights_folder)

        assert os.path.isdir(load_depth_folder), "Cannot find folder {}".format(self.opt.load_weights_folder_depth)
        assert os.path.isdir(load_weights_folder), "Cannot find folder {}".format(load_weights_folder)

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

        models_to_load = ['angencoder', 'angdecoder']
        modelnameMap = {'angencoder': 'encoder', 'angdecoder': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(load_weights_folder, "{}.pth".format(modelnameMap[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val_forward_backward_torch()
