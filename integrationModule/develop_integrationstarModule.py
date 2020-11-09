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
            self.val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False)

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
                instancepred = inputs['instancepred'].int().contiguous()
                mask = (mask * instancepred > 0).int()

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

        self.shrinkw = 4
        weightsx = torch.ones([1, 1, 2 * self.shrinkw + 1, 2 * self.shrinkw + 1])

        shrink = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=int(2 * self.shrinkw + 1), padding=self.shrinkw, bias=False)
        shrink.weight = nn.Parameter(weightsx, requires_grad=False)
        shrink = shrink.cuda()


        err1 = list()
        err2 = list()
        err3 = list()
        for batch_idx in range(self.val_dataset.__len__()):
            # batch_idx = self.val_dataset.filenames.index('2011_09_26/2011_09_26_drive_0052_sync 0000000026 l')
            batch_idx = 58
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

            instancepredcp = inputs['instancepred'].int().contiguous()
            instancepred = torch.zeros_like(instancepredcp)
            for kk in torch.unique(instancepredcp):
                if kk > 0:
                    tmpmask = instancepredcp != kk
                    tmpmask = tmpmask.float()
                    tmpmask = shrink(tmpmask) > (self.shrinkw * self.shrinkw) * 0.4
                    instancepred[tmpmask==0] = kk

            mask = torch.zeros_like(pred_depth, dtype=torch.int)
            mask[:,:,int(0.40810811 * self.opt.crph):, :] = 1
            mask = mask * (instancepred > 0).int()
            mask = mask.contiguous()

            # import shapeintegration_cuda
            # lam = 0.05
            # depth_optedin = torch.clone(pred_depth)
            # for k in range(5):
            #     depth_optedout = torch.zeros_like(depth_optedin)
            #     shapeintegration_cuda.shapeIntegration_crf_star_forward(pred_log, instancepred, mask, pred_depth, depth_optedin, depth_optedout, self.opt.crph, self.opt.crpw, self.opt.batch_size, lam)
            #     depth_optedin = torch.clone(depth_optedout)

            # import shapeintegration_cuda
            # lam = 0.05
            # depth_optedin = torch.clone(pred_depth)
            # depth_optedout = torch.zeros_like(depth_optedin)
            # shapeintegration_cuda.shapeIntegration_crf_star_forward(pred_log, instancepred, mask, pred_depth, depth_optedin, depth_optedout, self.opt.crph, self.opt.crpw, self.opt.batch_size, lam)
            #
            # xx, yy = np.meshgrid(range(self.opt.crpw), range(self.opt.crph), indexing='xy')
            # valxx = xx[instancepred.detach().cpu().numpy()[0,0,:,:] > 0]
            # valyy = yy[instancepred.detach().cpu().numpy()[0,0,:,:] > 0]
            #
            # rndind = np.random.randint(0, len(valxx))
            # c = 0
            # m = valyy[rndind]
            # n = valxx[rndind]
            #
            # endptsrec = self.validate_forward(pred_log, instancepred, mask, depth_optedin, self.opt.crph, self.opt.crpw, c, m, n, lam, depth_optedout[c,0,m,n])
            #
            # figrgb = tensor2rgb(F.interpolate(inputs['color'], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=False), ind=0)
            # coloredimg = np.zeros_like(np.array(figrgb))
            # instanceprednp = instancepred[0,0,:,:].cpu().numpy()
            # ratio = 0.3
            # for k in np.unique(instanceprednp):
            #     if k > 0:
            #         selector = instanceprednp == k
            #         coloredimg[selector, :] = np.repeat((np.random.random([1, 3]) * 255).astype(np.uint8), np.sum(selector), axis=0)
            # combined = np.array(figrgb).astype(np.float) * (1 - ratio) + coloredimg.astype(np.float) * ratio
            # combined = combined.astype(np.uint8)
            #
            # plt.figure()
            # plt.imshow(pil.fromarray(combined))
            # plt.scatter([n], [m], 1, 'r')
            # plt.scatter(endptsrec[:, 0], endptsrec[:, 1], 1, 'g')
            #
            # sufrnorm = self.sfnormOptimizer.depth2norm(depthMap=depth_optedin, intrinsic=inputs['K'])
            # sufrnormref = self.sfnormOptimizer.ang2normal(ang=pred_ang, intrinsic=inputs['K'])
            # tensor2rgb((sufrnorm + 1) / 2, ind=0).show()
            # tensor2rgb((sufrnormref + 1) / 2, ind=0).show()
            # tensor2disp(1 / depth_optedin, vmax=0.15, ind=0).show()
            # tensor2disp(1 / pred_depth, vmax=0.15, ind=0).show()

            depthgt = inputs['depthgt'].contiguous().float()
            depth_maskin = (depthgt > 0).int()
            import shapeintegration_cuda
            lam = 0.05
            depth_optedin = torch.clone(inputs['depthgt'])
            for k in range(10):
                depth_optedout = torch.zeros_like(depth_optedin)
                shapeintegration_cuda.shapeIntegration_crf_star_mask_forward(pred_log, instancepred, mask, depthgt, depth_optedin, depth_maskin, depth_optedout, self.opt.crph, self.opt.crpw, self.opt.batch_size, lam)
                depth_optedin = depth_optedout.clone()


            # angfromdepth = self.sfnormOptimizer.depth2ang_log(depthMap=pred_depth, intrinsic=inputs['K'])
            # logfromdepth = self.sfnormOptimizer.ang2log(ang=angfromdepth, intrinsic=inputs['K'])
            # depth_optedin = torch.clone(inputs['depthgt'])
            # for k in range(10):
            #     depth_optedout_cp = torch.zeros_like(depth_optedin)
            #     shapeintegration_cuda.shapeIntegration_crf_star_mask_forward(logfromdepth, instancepred, mask, depthgt, depth_optedin, depth_maskin, depth_optedout_cp, self.opt.crph, self.opt.crpw, self.opt.batch_size, lam)
            #     depth_optedin = depth_optedout.clone()

            angfromdepth = self.sfnormOptimizer.depth2ang_log(depthMap=pred_depth, intrinsic=inputs['K'])
            logfromdepth = self.sfnormOptimizer.ang2log(ang=angfromdepth, intrinsic=inputs['K'])
            depth_optedin = torch.clone(pred_depth)
            depth_optedout_cp = torch.zeros_like(depth_optedin)
            shapeintegration_cuda.shapeIntegration_crf_star_forward(logfromdepth, instancepred, mask, depthgt, depth_optedin, depth_optedout_cp, self.opt.crph, self.opt.crpw, self.opt.batch_size, lam)

            semidenseroot = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt'
            semidensefilepath = os.path.join(semidenseroot, inputs['tag'].split(' ')[0], 'image_02', '{}.png'.format(inputs['tag'].split(' ')[1]))
            semidensegt = pil.open(semidensefilepath)
            w, h = semidensegt.size
            left = int((w - self.opt.crpw) / 2)
            top = int((h - self.opt.crph) / 2)
            semidensegt = semidensegt.crop((left, top, left + self.opt.crpw, top + self.opt.crph))
            semidensegt = np.array(semidensegt) / 256.0

            depth_optedoutnp = depth_optedout[0,0,:,:].cpu().numpy()
            depth_optedout_cpnp = depth_optedout_cp[0, 0, :, :].cpu().numpy()
            instancemasknp = (instancepred[0,0,:,:].cpu().numpy() > 0) * (semidensegt > 0) * (depth_optedoutnp > 0) * (depth_optedout_cpnp > 0)
            gtdepthl = semidensegt[instancemasknp]

            gtfromshape = depth_optedoutnp[instancemasknp]
            gtfromdepth = depth_optedout_cpnp[instancemasknp]
            gtfromrawdepth = pred_depth[0,0,:,:].cpu().numpy()[instancemasknp]

            if np.sum(instancemasknp) == 0:
                continue

            err1.append(compute_errors(gtdepthl, gtfromshape))
            err2.append(compute_errors(gtdepthl, gtfromdepth))
            err3.append(compute_errors(gtdepthl, gtfromrawdepth))
            print("%d batch finished" % batch_idx)

            depth_optedoutvls = depth_optedout.clone()
            depth_optedoutvls[depth_optedoutvls == 0] = np.inf
            fig1 = tensor2disp(1 / depth_optedoutvls, vmax=0.15, ind=0)

            depth_optedout_cpvls = depth_optedout_cp.clone()
            depth_optedout_cpvls[depth_optedout_cpvls == 0] = np.inf
            fig2 = tensor2disp(1 / depth_optedout_cpvls, vmax=0.15, ind=0)

            semidensegtvls = torch.from_numpy(np.copy(semidensegt)).unsqueeze(0).unsqueeze(0)
            semidensegtvls[semidensegtvls == 0] = np.inf
            fig3 = tensor2disp(1 / semidensegtvls, vmax=0.15, ind=0)

            figrgb = tensor2rgb(F.interpolate(inputs['color'], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=False), ind=0)
            coloredimg = np.zeros_like(np.array(figrgb))
            instanceprednp = instancepred[0,0,:,:].cpu().numpy()
            ratio = 0.3
            for k in np.unique(instanceprednp):
                if k > 0:
                    selector = instanceprednp == k
                    coloredimg[selector, :] = np.repeat((np.random.random([1, 3]) * 255).astype(np.uint8), np.sum(selector), axis=0)
            combined = np.array(figrgb).astype(np.float) * (1 - ratio) + coloredimg.astype(np.float) * ratio
            combined = combined.astype(np.uint8)

            figcombined = np.concatenate([np.array(fig1), np.array(fig2), np.array(fig3), np.array(combined)], axis=0)
            pil.fromarray(figcombined).save(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/compare_shapeanddepthInstance', "{}.png".format(str(batch_idx).zfill(2))))

        err1 = np.array(err1)
        err1 = np.mean(err1, axis=0)
        err2 = np.array(err2)
        err2 = np.mean(err2, axis=0)
        err3 = np.array(err3)
        err3 = np.mean(err3, axis=0)
        print("\nFromShape Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*err1.tolist()) + "\\\\")

        print("\nFromDepth Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*err2.tolist()) + "\\\\")

        print("\nFromDepthRaw Performance:")
        print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*err3.tolist()) + "\\\\")
    def validate_forward(self, log, semantics, mask, depthin, height, width, c, m, n, lam, refdepth):
        depthingrad = torch.zeros_like(depthin)
        semancat = semantics[c][0][m][n]

        totcounts = 0
        lateralre = 0

        endptsrec = list()

        # Left up direction
        if mask[c][0][m][n] == 1:
            sn = n
            sm = m
            intlog = 0
            while(True):
                sn -= 1
                sm -= 1
                if(sn >=0 and sm >= 0 and sn < width and sm < height):
                    if (semantics[c][0][sm][sn] != semancat) or (semantics[c][0][sm+1][sn] != semancat) or (semantics[c][0][sm][sn+1] != semancat): break
                    if (mask[c][0][sm][sn] != 1) or (mask[c][0][sm+1][sn] != 1) or (mask[c][0][sm][sn+1] != 1): break
                    intlog += (-log[c][0][sm+1][sn] -log[c][1][sm][sn] -log[c][1][sm][sn+1] -log[c][0][sm][sn]) / 2
                    totcounts += 1
                    lateralre += torch.exp(-intlog) * depthin[c][0][sm][sn]
                else:
                    break
            endptsrec.append(np.array([sn, sm]))

            # right up direction
            sn = n
            sm = m
            intlog = 0
            while(True):
                sn += 1
                sm -= 1
                if (sn >= 0 and sm >= 0 and sn < width and sm < height):
                    if (semantics[c][0][sm][sn] != semancat) or (semantics[c][0][sm+1][sn] != semancat) or (semantics[c][0][sm][sn-1] != semancat): break
                    if (mask[c][0][sm][sn] != 1) or (mask[c][0][sm + 1][sn] != 1) or (mask[c][0][sm][sn - 1] != 1): break
                    intlog += (log[c][0][sm+1][sn-1] -log[c][1][sm][sn] -log[c][1][sm][sn-1] +log[c][0][sm][sn-1]) / 2
                    totcounts += 1
                    lateralre += torch.exp(-intlog) * depthin[c][0][sm][sn]
                else:
                    break
            endptsrec.append(np.array([sn, sm]))

            # Left down direction
            sn = n
            sm = m
            intlog = 0
            while(True):
                sn -= 1
                sm += 1
                if(sn >=0 and sm >= 0 and sn < width and sm < height):
                    if (semantics[c][0][sm][sn] != semancat) or (semantics[c][0][sm-1][sn] != semancat) or (semantics[c][0][sm][sn+1] != semancat): break
                    if (mask[c][0][sm][sn] != 1) or (mask[c][0][sm-1][sn] != 1) or (mask[c][0][sm][sn+1] != 1): break
                    intlog += (-log[c][0][sm-1][sn] +log[c][1][sm-1][sn] +log[c][1][sm-1][sn+1] -log[c][0][sm][sn]) / 2
                    totcounts += 1
                    lateralre += torch.exp(-intlog) * depthin[c][0][sm][sn]
                else: break
            endptsrec.append(np.array([sn, sm]))

            # Right down direction
            sn = n
            sm = m
            intlog = 0
            while(True):
                sn += 1
                sm += 1
                if(sn >=0 and sm >= 0 and sn < width and sm < height):
                    if (semantics[c][0][sm][sn] != semancat) or (semantics[c][0][sm][sn-1] != semancat) or (semantics[c][0][sm-1][sn] != semancat): break
                    if (mask[c][0][sm][sn] != 1) or (mask[c][0][sm][sn-1] != 1) or (mask[c][0][sm-1][sn] != 1): break
                    intlog += (log[c][0][sm-1][sn-1] +log[c][1][sm-1][sn] +log[c][1][sm-1][sn-1] +log[c][0][sm][sn-1]) / 2
                    totcounts += 1
                    lateralre += torch.exp(-intlog) * depthin[c][0][sm][sn]
                else: break
            endptsrec.append(np.array([sn, sm]))

            # Left
            sm = m
            intlog = 0
            for sn in range(n-1, -1, -1):
                if semantics[c][0][sm][sn] != semancat: break
                if mask[c][0][sm][sn] != 1: break
                intlog += -log[c][0][sm][sn]
                totcounts += 1
                lateralre += torch.exp(-intlog) * depthin[c][0][sm][sn]
            endptsrec.append(np.array([sn, sm]))

            # Right
            sm = m
            intlog = 0
            for sn in range(n+1, width, 1):
                if semantics[c][0][sm][sn] != semancat: break
                if mask[c][0][sm][sn] != 1: break
                intlog += log[c][0][sm][sn-1]
                totcounts += 1
                lateralre += torch.exp(-intlog) * depthin[c][0][sm][sn]
            endptsrec.append(np.array([sn, sm]))

            # Up
            sn = n
            intlog = 0
            for sm in range(m-1, -1, -1):
                if semantics[c][0][sm][sn] != semancat: break
                if mask[c][0][sm][sn] != 1: break
                intlog += -log[c][1][sm][sn];
                totcounts += 1;
                lateralre += torch.exp(-intlog) * depthin[c][0][sm][sn]
            endptsrec.append(np.array([sn, sm]))

            # Down
            sn = n
            intlog = 0
            for sm in range(m+1, height, 1):
                if semantics[c][0][sm][sn] != semancat: break
                if mask[c][0][sm][sn] != 1: break
                intlog += log[c][1][sm-1][sn]
                totcounts += 1
                lateralre += torch.exp(-intlog) * depthin[c][0][sm][sn]
            endptsrec.append(np.array([sn, sm]))

            if totcounts > 0:
                opteddepth = lam * depthin[c][0][m][n] + (1 - lam) * lateralre / totcounts
            else:
                opteddepth = depthin[c][0][m][n]

            assert torch.abs(opteddepth - refdepth) < 1e-2
        return np.array(endptsrec)
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
