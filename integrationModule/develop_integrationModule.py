from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from Exp_PreSIL.dataloader_kitti import KittiDataset
import networks
from layers import *
from networks import *
import argparse
import shapeintegration_cuda

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

# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=12,                 help="batch size")
parser.add_argument("--num_workers",                type=int,   default=6,                  help="number of dataloader workers")

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

        self.train_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, train_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=True, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path
        )

        self.val_dataset = KittiDataset(
            self.opt.data_path, self.opt.val_gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path
        )

        self.train_loader = DataLoader(
            self.train_dataset, self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            self.val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def val_forward_backward_torch(self):
        self.set_eval()
        from kitti_utils import translateTrainIdSemanticsTorch, variancebar
        variancebar = torch.from_numpy(variancebar).float().cuda()

        self.val_dataset.filenames.index('2011_09_26/2011_09_26_drive_0052_sync 0000000026 l')

        for batch_idx in range(self.val_dataset.__len__()):
            # batch_idx = self.val_dataset.filenames.index('2011_09_26/2011_09_26_drive_0052_sync 0000000026 l')
            inputs = self.val_dataset.__getitem__(batch_idx)

            for key, ipt in inputs.items():
                if not key == 'tag':
                    inputs[key] = ipt.to(self.device).unsqueeze(0)

            with torch.no_grad():
                outputs_depth = self.depthmodels['depthdecoder'](self.depthmodels['depthencoder'](inputs['color']))
                outputs_ang = self.angmodels['angdecoder'](self.angmodels['angencoder'](inputs['color']))

            pred_depth = F.interpolate(outputs_depth['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
            _, pred_depth = disp_to_depth(pred_depth, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
            pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

            pred_ang = F.interpolate(outputs_ang['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
            pred_ang = (pred_ang - 0.5) * 2 * np.pi
            pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang)
            pred_log = pred_log.contiguous()

            confidence = torch.ones_like(pred_depth).contiguous()

            edge = self.sfnormOptimizer.ang2edge(ang=pred_ang, intrinsic=inputs['K']).int()

            mask = torch.zeros_like(pred_depth, dtype=torch.int)
            mask[:,:,int(0.40810811 * self.opt.crph):int(0.99189189 * self.opt.crph), int(0.03594771 * self.opt.crpw):int(0.96405229 * self.opt.crpw)] = 1
            mask = mask * (1 - edge)
            mask = mask.contiguous()

            semantics = translateTrainIdSemanticsTorch(inputs['semanticspred']).int().contiguous()

            assert semantics.max() < variancebar.shape[0]

            from integrationModule import IntegrationFunction
            integrateFunction = IntegrationFunction.apply

            c = np.random.randint(0, self.opt.batch_size)
            m = np.random.randint(0, self.opt.crph)
            n = np.random.randint(0, self.opt.crpw)
            eps = 1e-2
            # Numerical Test for gradient of depth
            pred_depth1 = pred_depth.clone()
            pred_depth1[c, 0, m, n] = pred_depth1[c, 0, m, n] + eps
            pred_depth2 = pred_depth.clone()
            pred_depth2[c, 0, m, n] = pred_depth2[c, 0, m, n] - eps
            opteddepth1 = integrateFunction(pred_ang, pred_log, confidence, semantics, mask, pred_depth1, variancebar, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            opteddepth2 = integrateFunction(pred_ang, pred_log, confidence, semantics, mask, pred_depth2, variancebar, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            numgraddepth = torch.sum(torch.abs(opteddepth1 - opteddepth2)) / eps / 2

            pred_depth_ref = pred_depth.clone()
            pred_depth_ref.requires_grad = True
            opteddepth = integrateFunction(pred_ang, pred_log, confidence, semantics, mask, pred_depth_ref, variancebar, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            torch.sum(torch.abs(opteddepth)).backward()

            ratio = numgraddepth / pred_depth_ref.grad[c, 0, m, n]


            # Numerical Test for gradient of confidence
            while(True):
                c = np.random.randint(0, self.opt.batch_size)
                m = np.random.randint(0, self.opt.crph)
                n = np.random.randint(0, self.opt.crpw)
                if variancebar[semantics[c, 0, m, n], 0] > 0 and variancebar[semantics[c, 0, m, n], 1] > 0 and mask[c, 0, m, n] == 1:
                    break
            eps = 1e-1
            confidence1 = confidence.clone()
            confidence1[c, 0, m, n] = confidence1[c, 0, m, n] + eps
            confidence2 = confidence.clone()
            confidence2[c, 0, m, n] = confidence2[c, 0, m, n] - eps
            opteddepth1 = integrateFunction(pred_ang, pred_log, confidence1, semantics, mask, pred_depth, variancebar, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            opteddepth2 = integrateFunction(pred_ang, pred_log, confidence2, semantics, mask, pred_depth, variancebar, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            numgradconfidence = torch.sum(torch.abs(opteddepth1) - torch.abs(opteddepth2)) / eps / 2

            confidence_ref = confidence.clone()
            confidence_ref.requires_grad = True
            opteddepth = integrateFunction(pred_ang, pred_log, confidence_ref, semantics, mask, pred_depth, variancebar, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            torch.sum(torch.abs(opteddepth)).backward()

            ratio = numgradconfidence / confidence_ref.grad[c, 0, m, n]


    def val_forward_backward_cuda(self):
        self.set_eval()
        from kitti_utils import translateTrainIdSemanticsTorch, variancebar
        variancebar = torch.from_numpy(variancebar).float().cuda()

        self.val_dataset.filenames.index('2011_09_26/2011_09_26_drive_0052_sync 0000000026 l')

        for batch_idx in range(self.val_dataset.__len__()):
            # batch_idx = self.val_dataset.filenames.index('2011_09_26/2011_09_26_drive_0052_sync 0000000026 l')
            inputs = self.val_dataset.__getitem__(batch_idx)

            for key, ipt in inputs.items():
                if not key == 'tag':
                    inputs[key] = ipt.to(self.device).unsqueeze(0)

            with torch.no_grad():
                outputs_depth = self.depthmodels['depthdecoder'](self.depthmodels['depthencoder'](inputs['color']))
                outputs_ang = self.angmodels['angdecoder'](self.angmodels['angencoder'](inputs['color']))

            pred_depth = F.interpolate(outputs_depth['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
            _, pred_depth = disp_to_depth(pred_depth, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
            pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

            # pred_ang = self.sfnormOptimizer.depth2ang_log(pred_depth, intrinsic=inputs['K'])
            pred_ang = F.interpolate(outputs_ang['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
            pred_ang = (pred_ang - 0.5) * 2 * np.pi
            pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang)
            pred_log = pred_log.contiguous()

            confidence = torch.ones_like(pred_depth).contiguous()
            summedConfidence = torch.zeros_like(pred_depth).contiguous()

            edge = self.sfnormOptimizer.ang2edge(ang=pred_ang, intrinsic=inputs['K']).int()

            mask = torch.zeros_like(pred_depth, dtype=torch.int)
            mask[:,:,int(0.40810811 * self.opt.crph):int(0.99189189 * self.opt.crph), int(0.03594771 * self.opt.crpw):int(0.96405229 * self.opt.crpw)] = 1
            mask = mask * (1 - edge)
            mask = mask.contiguous()

            semantics = translateTrainIdSemanticsTorch(inputs['semanticspred']).int().contiguous()

            assert semantics.max() < variancebar.shape[0]

            integrated_depth = torch.zeros_like(pred_depth)
            shapeintegration_cuda.shapeIntegration_forward(pred_ang, pred_log, confidence, semantics, mask, pred_depth, integrated_depth, summedConfidence, variancebar, self.opt.crph, self.opt.crpw, self.opt.batch_size)

            while(True):
                c = np.random.randint(0, self.opt.batch_size)
                m = np.random.randint(0, self.opt.crph)
                n = np.random.randint(0, self.opt.crpw)
                if variancebar[semantics[c, 0, m, n], 0] > 0 and variancebar[semantics[c, 0, m, n], 1] > 0 and mask[c, 0, m, n] == 1:
                    break
            self.validate_forward(ang=pred_ang, log=pred_log, confidence=confidence, semantics=semantics, mask=mask, depthin=pred_depth, varbar=variancebar, height=self.opt.crph, width=self.opt.crpw, c=c, m=m, n=n, integrateddepth=integrated_depth[c,0,m,n])

            gradin = torch.zeros_like(pred_depth).contiguous()
            gradin[c, 0, m, n] = 1
            gradout_depth = torch.zeros_like(pred_depth).contiguous()
            gradout_confidence = torch.zeros_like(pred_depth).contiguous()
            shapeintegration_cuda.shapeIntegration_backward(pred_ang, pred_log, confidence, semantics, mask, pred_depth, integrated_depth, summedConfidence, gradin, gradout_depth, gradout_confidence, variancebar, self.opt.crph, self.opt.crpw, self.opt.batch_size)
            self.validate_backward(ang=pred_ang, log=pred_log, confidence=confidence, semantics=semantics, mask=mask, depthin=pred_depth, varbar=variancebar, height=self.opt.crph, width=self.opt.crpw, c=c, m=m, n=n, grad_depth=gradout_depth, grad_confidence=gradout_confidence, summedConfidence=summedConfidence)

            minang = - np.pi / 3 * 2
            maxang = 2 * np.pi - np.pi / 3 * 2
            vind = 0

            vlscolor = F.interpolate(inputs['color'], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True)
            figrgb = tensor2rgb(vlscolor, ind=vind)
            fig_depthgt = tensor2disp(inputs['depthgt'], vmax=40, ind=vind)

            viewind = 0
            fig1 = tensor2rgb(F.interpolate(inputs['color'], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True), ind=viewind)
            fig2 = tensor2disp(1/pred_depth, vmax=0.15, ind=viewind)
            fig3 = tensor2disp(1 / integrated_depth, vmax=0.15, ind=viewind)
            fig4 = tensor2semantic(inputs['semanticspred'], ind=viewind)
            fig5 = tensor2disp(pred_ang[:, 0:1, :, :] - minang, vmax=maxang, ind=viewind)
            fig6 = tensor2disp(pred_ang[:, 1:2, :, :] - minang, vmax=maxang, ind=viewind)
            fig7 = tensor2disp(mask, vmax=1, ind=viewind)
            fig8 = tensor2disp(mask, vmax=1, ind=viewind)

            figl = np.concatenate([np.array(fig1), np.array(fig5), np.array(fig6), np.array(fig7)], axis=0)
            figr = np.concatenate([np.array(fig4), np.array(fig2), np.array(fig3), np.array(fig8)], axis=0)
            fig = np.concatenate([np.array(figl), np.array(figr)], axis=1)

            figname = inputs['tag'].split(' ')[0].split('/')[1] + '_' + inputs['tag'].split(' ')[1]
            pil.fromarray(fig).save('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/analysis_semanticMetric/integratedDepth2/{}.png'.format(figname))

    def validate_backward(self, ang, log, confidence, semantics, mask, depthin, varbar, height, width, c, m, n, grad_depth, grad_confidence, summedConfidence):
        confidence = confidence.clone()
        confidence.requires_grad = True
        depthin = depthin.clone()
        depthin.requires_grad = True

        semancat = semantics[c][0][m][n]
        depthh = 0
        depthv = 0

        intconfidenceh = 0
        intconfidencev = 0
        intconfidenceh += confidence[c][0][m][n]
        intconfidencev += confidence[c][0][m][n]

        intconfidence = summedConfidence[c, 0, m, n]
        inputgrad = 1
        mimiced_gradConfidence = torch.zeros_like(grad_confidence)

        if varbar[semancat][0] > 0:
            breakpos = False
            breakneg = False
            countpos = 0
            countneg = 0
            intpos = 0
            intneg = 0

            sums = ang[c][0][m][n].clone()
            sumsquare = ang[c][0][m][n] * ang[c][0][m][n]

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
                    elif (sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][0]:
                        breakpos=True
                        continue
                    else:
                        intpos += log[c][0][m][sn - 1]
                        intconfidenceh += confidence[c][0][m][sn]
                        sums += ang[c][0][m][sn]
                        sumsquare += ang[c][0][m][sn] * ang[c][0][m][sn]
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
                    elif (sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][0]:
                        breakneg=True
                        continue
                    else:
                        intneg -= log[c][0][m][sn]
                        intconfidenceh += confidence[c][0][m][sn]
                        sums += ang[c][0][m][sn]
                        sumsquare += ang[c][0][m][sn] * ang[c][0][m][sn]
                        countneg = countneg + 1

            for ii in range(countpos, 0, -1):
                sn = n + ii
                depthh += confidence[c][0][m][sn] / intconfidenceh * torch.exp(-intpos) * depthin[c][0][m][sn]
                mimiced_gradConfidence[c][0][m][sn] += (1 / intconfidence - confidence[c][0][m][sn] / intconfidence / intconfidence) * torch.exp(-intpos) * depthin[c][0][m][sn] * inputgrad
                intpos -= log[c][0][m][sn-1]

            for ii in range(countneg, 0, -1):
                sn = n - ii
                depthh += confidence[c][0][m][sn] / intconfidenceh * torch.exp(-intneg) * depthin[c][0][m][sn]
                mimiced_gradConfidence[c][0][m][sn] += (1 / intconfidence - confidence[c][0][m][sn] / intconfidence / intconfidence) * torch.exp(-intneg) * depthin[c][0][m][sn] * inputgrad
                intneg += log[c][0][m][sn]

        if varbar[semancat][1] > 0:
            breakpos = False
            breakneg = False
            countpos = 0
            countneg = 0
            intpos = 0
            intneg = 0

            sums = ang[c][1][m][n].clone()
            sumsquare = ang[c][1][m][n] * ang[c][1][m][n]

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
                    elif (sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][1]:
                        breakpos=True
                        continue
                    else:
                        intpos += log[c][1][sm - 1][n]
                        intconfidencev += confidence[c][0][sm][n]
                        sums += ang[c][1][sm][n]
                        sumsquare += ang[c][1][sm][n] * ang[c][1][sm][n]
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
                    elif (sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][1]:
                        breakneg=True
                        continue
                    else:
                        intneg -= log[c][1][sm][n]
                        intconfidencev += confidence[c][0][sm][n]
                        sums += ang[c][1][sm][n]
                        sumsquare += ang[c][1][sm][n] * ang[c][1][sm][n]
                        countneg = countneg + 1

            for ii in range(countpos, 0, -1):
                sm = m + ii
                depthv += confidence[c][0][sm][n] / intconfidencev * torch.exp(-intpos) * depthin[c][0][sm][n]
                mimiced_gradConfidence[c][0][sm][n] += (1 / intconfidence - confidence[c][0][sm][n] / intconfidence / intconfidence) * torch.exp(-intpos) * depthin[c][0][sm][n] * inputgrad
                intpos -= log[c][1][sm-1][n]

            for ii in range(countneg, 0, -1):
                sm = m - ii
                depthv += confidence[c][0][sm][n] / intconfidencev * torch.exp(-intneg) * depthin[c][0][sm][n]
                mimiced_gradConfidence[c][0][sm][n] += (1 / intconfidence - confidence[c][0][sm][n] / intconfidence / intconfidence) * torch.exp(-intneg) * depthin[c][0][sm][n] * inputgrad
                intneg += log[c][1][sm][n]

        depthh += confidence[c][0][m][n] / intconfidenceh * depthin[c][0][m][n]
        depthv += confidence[c][0][m][n] / intconfidencev * depthin[c][0][m][n]
        depthout = depthh * intconfidenceh / (intconfidenceh + intconfidencev) + depthv * intconfidencev / (intconfidenceh + intconfidencev)
        mimiced_gradConfidence[c][0][m][n] += (2 / intconfidence - 4 * confidence[c][0][m][n] / intconfidence / intconfidence) * depthin[c][0][m][n] * inputgrad
        depthout.backward()

        a = 1
        assert torch.abs(confidence.grad - grad_confidence).max() < 1e-3
        assert torch.abs(depthin.grad - grad_depth).max() < 1e-3
        assert torch.abs((intconfidenceh + intconfidencev) - summedConfidence[c, 0, m, n]) < 1e-3

    def validate_forward(self, ang, log, confidence, semantics, mask, depthin, varbar, height, width, c, m, n, integrateddepth):
        semancat = semantics[c][0][m][n]
        intconfidenceh = confidence[c][0][m][n].clone()
        intconfidencev = confidence[c][0][m][n].clone()
        depthh = 0
        depthv = 0

        if varbar[semancat][0] > 0:
            breakpos = False
            breakneg = False
            countpos = 0
            countneg = 0
            intpos = 0
            intneg = 0

            sums = ang[c][0][m][n].clone()
            sumsquare = ang[c][0][m][n] * ang[c][0][m][n]

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
                    elif (sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][0]:
                        breakpos=True
                        continue
                    else:
                        intpos += log[c][0][m][sn - 1]
                        intconfidenceh += confidence[c][0][m][sn]
                        sums += ang[c][0][m][sn]
                        sumsquare += ang[c][0][m][sn] * ang[c][0][m][sn]
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
                    elif (sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][0]:
                        breakneg=True
                        continue
                    else:
                        intneg -= log[c][0][m][sn]
                        intconfidenceh += confidence[c][0][m][sn]
                        sums += ang[c][0][m][sn]
                        sumsquare += ang[c][0][m][sn] * ang[c][0][m][sn]
                        countneg = countneg + 1

            for ii in range(countpos, 0, -1):
                sn = n + ii
                depthh += confidence[c][0][m][sn] / intconfidenceh * torch.exp(-intpos) * depthin[c][0][m][sn]
                intpos -= log[c][0][m][sn-1]

            for ii in range(countneg, 0, -1):
                sn = n - ii
                depthh += confidence[c][0][m][sn] / intconfidenceh * torch.exp(-intneg) * depthin[c][0][m][sn]
                intneg += log[c][0][m][sn]

            varref = (sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1))
            assert torch.abs(torch.var(ang[c, 0, m, n - countneg: n + countpos + 1], unbiased=False) - varref) < 1e-3

        if varbar[semancat][1] > 0:
            breakpos = False
            breakneg = False
            countpos = 0
            countneg = 0
            intpos = 0
            intneg = 0

            sums = ang[c][1][m][n].clone()
            sumsquare = ang[c][1][m][n] * ang[c][1][m][n]

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
                    elif (sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][1]:
                        breakpos=True
                        continue
                    else:
                        intpos += log[c][1][sm - 1][n]
                        intconfidencev += confidence[c][0][sm][n]
                        sums += ang[c][1][sm][n]
                        sumsquare += ang[c][1][sm][n] * ang[c][1][sm][n]
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
                    elif (sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1)) > varbar[semancat][1]:
                        breakneg=True
                        continue
                    else:
                        intneg -= log[c][1][sm][n]
                        intconfidencev += confidence[c][0][sm][n]
                        sums += ang[c][1][sm][n]
                        sumsquare += ang[c][1][sm][n] * ang[c][1][sm][n]
                        countneg = countneg + 1

            for ii in range(countpos, 0, -1):
                sm = m + ii
                depthv += confidence[c][0][sm][n] / intconfidencev * torch.exp(-intpos) * depthin[c][0][sm][n]
                intpos -= log[c][1][sm-1][n]

            for ii in range(countneg, 0, -1):
                sm = m - ii
                depthv += confidence[c][0][sm][n] / intconfidencev * torch.exp(-intneg) * depthin[c][0][sm][n]
                intneg += log[c][1][sm][n]

        varref = (sumsquare / (countpos + countneg + 1) - sums * sums / (countpos + countneg + 1) / (countpos + countneg + 1))
        assert torch.abs(torch.var(ang[c, 1, m - countneg: m + countpos + 1, n], unbiased=False) - varref) < 1e-3

        depthh += confidence[c][0][m][n] / intconfidenceh * depthin[c][0][m][n]
        depthv += confidence[c][0][m][n] / intconfidencev * depthin[c][0][m][n]
        depthout = depthh * intconfidenceh / (intconfidenceh + intconfidencev) + depthv * intconfidencev / (intconfidenceh + intconfidencev)

        assert torch.abs(depthout - integrateddepth) < 1e-3

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
    trainer.val_forward_backward_torch()
