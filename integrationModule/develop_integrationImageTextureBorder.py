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
parser.add_argument("--svpath",                     type=str,                               help="path to kitti gt file")

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

    def develop_forward(self):
        self.set_eval()
        import shapeintegration_cuda
        outputs = dict()

        for batch_idx in range(self.val_dataset.__len__()):
            # batch_idx = self.val_dataset.filenames.index('2011_09_26/2011_09_26_drive_0059_sync 0000000014 l')
            inputs = self.val_dataset.__getitem__(batch_idx)

            for key, ipt in inputs.items():
                if not key == 'tag':
                    inputs[key] = ipt.to(self.device).unsqueeze(0)

            with torch.no_grad():
                outputs_depth = self.depthmodels['depthdecoder'](self.depthmodels['depthencoder'](inputs['color']))
                outputs_ang = self.angmodels['angdecoder'](self.angmodels['angencoder'](inputs['color']))

            pred_depth = F.interpolate(outputs_depth['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=False)
            _, pred_depth = disp_to_depth(pred_depth, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
            pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

            pred_ang = F.interpolate(outputs_ang['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=False)
            pred_ang = (pred_ang - 0.5) * 2 * np.pi
            pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang)
            pred_log = pred_log.contiguous()

            rgbresized = F.interpolate(inputs['color'], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=False)

            mask = torch.zeros_like(pred_depth, dtype=torch.int)
            mask[:, :, int(0.40810811 * self.opt.crph):, :] = 1

            interestmask = inputs['semanticspred_cat']
            interestmask = interestmask.detach().cpu().numpy()[0,0,:,:]
            interestmask = interestmask >= 2
            xx, yy = np.meshgrid(range(self.opt.crpw), range(self.opt.crph), indexing='xy')

            samplenum = 30
            canditx = xx[interestmask]
            candity = yy[interestmask]

            # plt.figure(figsize=(16, 9))
            # plt.imshow(tensor2rgb(rgbresized, ind=0))
            # for kk in range(samplenum):
            #     rndind = np.random.randint(0, len(canditx) - 1, 1)[0]
            #     n = canditx[rndind]
            #     m = candity[rndind]
            #     c = 0
            #
            #     plt.scatter(n, m, c='r', s=1)
            #
            #     textureT = 0.2
            #     ptsopt = self.getoptrange(rgbresized, mask, c, m, n, textureT)
            #     if len(ptsopt) > 0:
            #         ptsopt = np.array(ptsopt)
            #         plt.scatter(ptsopt[:, 0], ptsopt[:, 1], c='g', s=1)
            # figname = inputs['tag'].split(' ')[0].split('/')[1] + '_' + inputs['tag'].split(' ')[1]
            # plt.savefig(os.path.join(self.opt.svpath, figname))
            # plt.close()

            itnum = 3
            lamb = 0.05
            textureT = 0.2
            depth_optedin = pred_depth.clone()
            depth_optedout = torch.zeros_like(depth_optedin)
            for i in range(itnum):
                if i == 0:
                    outputs[('depth_opted', i)] = depth_optedin.clone()
                elif i == 1:
                    outputs[('depth_opted', i)] = depth_optedout.clone()
                elif i == itnum - 1:
                    outputs[('depth_opted', i)] = depth_optedout.clone()
                shapeintegration_cuda.shapeIntegration_crf_forward(pred_ang, pred_log, rgbresized, mask, pred_depth, depth_optedin, depth_optedout, self.opt.crph, self.opt.crpw, 1, lamb, textureT)
                depth_optedin = depth_optedout.clone()

            # vls
            sfnorm_opted_figs = list()
            depth_opted_figs = list()
            for i in range(itnum):
                if i == 0 or i == 1 or i == itnum - 1:
                    sfnorm_opted = self.sfnormOptimizer.depth2norm(outputs[('depth_opted', i)], intrinsic=inputs['K'])
                    sfnorm_opted_figs.append(np.array(tensor2rgb((sfnorm_opted + 1) / 2, ind=0)))
                    figd = tensor2disp(1 / outputs[('depth_opted', i)], vmax=0.25, ind=0)
                    depth_opted_figs.append(np.array(figd))

            figrgb = tensor2rgb(F.interpolate(inputs['color'], [self.opt.crph, self.opt.crpw], mode='bilinear', align_corners=True), ind=0)
            figborderinfo = np.concatenate([np.array(figrgb), np.array(figrgb)], axis=1)

            sfnorm_opted_fig = np.concatenate(sfnorm_opted_figs, axis=0)
            depth_opted_fig = np.concatenate(depth_opted_figs, axis=0)
            figanghoview = np.concatenate([sfnorm_opted_fig, depth_opted_fig], axis=1)
            figanghoview = np.concatenate([figanghoview, figborderinfo], axis=0)
            figname = inputs['tag'].split(' ')[0].split('/')[1] + '_' + inputs['tag'].split(' ')[1]
            pil.fromarray(figanghoview).save(os.path.join(self.opt.svpath, "{}.png".format(figname)))

    def getoptrange(self, rgb, mask, c, m, n, textureT):
        ptsrec = list()
        height, width = rgb.shape[2:4]
        textureref = rgb[c, :, m, n]

        if mask[c, 0, m, n] > 0:
            breakpos = False
            breakneg = False
            textureDiffh = 0
            for ii in range(width * 2):
                if breakpos and breakneg:
                    break
                if (ii & 1) == False:
                    if (breakpos): continue
                    sn = int(n + int(ii / 2) + 1)
                    if sn >= width:
                        breakpos = True
                        continue
                    elif mask[c][0][m][sn] == 0:
                        breakpos = True
                        continue
                    elif textureDiffh > textureT:
                        breakpos = True
                        continue
                    else:
                        textureDiffh = textureDiffh + torch.sqrt(torch.sum((textureref - rgb[c, :, m, sn]) ** 2))
                        ptsrec.append(np.array([sn, m]))
                else:
                    if breakneg:
                        continue
                    sn = int(n - int(ii / 2) - 1)
                    if sn < 0:
                        breakneg = True
                        continue
                    elif mask[c][0][m][sn] == 0:
                        breakneg = True
                        continue
                    elif textureDiffh > textureT:
                        breakneg = True
                        continue
                    else:
                        textureDiffh = textureDiffh + torch.sqrt(torch.sum((textureref - rgb[c, :, m, sn]) ** 2))
                        ptsrec.append(np.array([sn, m]))

        if mask[c, 0, m, n] > 0:
            breakpos = False
            breakneg = False
            textureDiffv = 0
            for ii in range(height * 2):
                if breakpos and breakneg:
                    break
                if (ii & 1) == False:
                    if breakpos:
                        continue
                    sm = int(m + int(ii / 2) + 1)
                    if sm >= height:
                        breakpos = True
                        continue
                    elif mask[c][0][sm][n] == 0:
                        breakpos = True
                        continue
                    elif textureDiffv > textureT:
                        breakpos = True
                        continue
                    else:
                        textureDiffv = textureDiffv + torch.sqrt(torch.sum((textureref - rgb[c, :, sm, n]) ** 2))
                        ptsrec.append(np.array([n, sm]))
                else:
                    if breakneg:
                        continue
                    sm = int(m - int(ii / 2) - 1)
                    if sm < 0:
                        breakneg = True
                        continue
                    elif mask[c][0][sm][n] == 0:
                        breakneg = True
                        continue
                    elif textureDiffv > textureT:
                        breakneg = True
                        continue
                    else:
                        textureDiffv = textureDiffv + torch.sqrt(torch.sum((textureref - rgb[c, :, sm, n]) ** 2))
                        ptsrec.append(np.array([n, sm]))

        return ptsrec

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
    trainer.develop_forward()
