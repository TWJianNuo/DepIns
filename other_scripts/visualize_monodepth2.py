# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!
from options import MonodepthOptions

import matlab
import matlab.engine
eng = matlab.engine.start_matlab()

class Visualizer:
    def __init__(self, options):
        self.opt = options
        fpath = os.path.join(os.path.dirname(__file__), "../splits", self.opt.split, "{}_files.txt")
        self.train_filenames = readlines(fpath.format("train"))
        num_train_samples = len(self.train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        train_dataset = datasets.KITTIRAWDataset(
            self.opt.data_path, self.train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True and not self.opt.noAug, load_detect=self.opt.predins,
            detect_path=self.opt.detect_path, load_seman = self.opt.loadSeman, load_pose=self.opt.loadPose,
            loadPredDepth = self.opt.loadPredDepth, predDepthPath = self.opt.predDepthPath)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size,  shuffle = not self.opt.noshuffle,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

    def visualize(self):

        for batch_idx, inputs in enumerate(self.train_loader):

            for key, ipt in inputs.items():
                if key not in ['entry_tag']:
                    inputs[key] = ipt.to(torch.device("cuda"))

            self.check_depthMap(inputs)

    def check_depthMap(self, inputs):
        xx, yy = np.meshgrid(range(self.opt.width), range(self.opt.height), indexing='xy')
        xx = torch.from_numpy(xx).unsqueeze(0).unsqueeze(0).expand([self.opt.batch_size, 1, -1, -1]).cuda().float()
        yy = torch.from_numpy(yy).unsqueeze(0).unsqueeze(0).expand([self.opt.batch_size, 1, -1, -1]).cuda().float()
        pixelLocs = torch.cat([xx, yy], dim=1)
        predDepth = inputs['predDepth']
        pts3d = backProjTo3d(pixelLocs, predDepth, inputs['invcamK'])
        mono_projected2d, _, _ = project_3dptsTo2dpts(pts3d=pts3d, camKs=inputs['camK'])
        mono_sampledColor = sampleImgs(inputs[('color', 0, 0)], mono_projected2d)

        downsample_rat = 10
        mapping = {'l' : 'image_02', 'r' : 'image_03'}
        for i in range(self.opt.batch_size):
            drawIndex = i

            draw_mono_sampledColor = mono_sampledColor[drawIndex, :, :, :].detach().cpu().view(3, -1).permute([1,0]).numpy()[::downsample_rat, :]
            drawX_mono = pts3d[drawIndex, 0, :, :].detach().cpu().numpy().flatten()[::downsample_rat]
            drawY_mono = pts3d[drawIndex, 1, :, :].detach().cpu().numpy().flatten()[::downsample_rat]
            drawZ_mono = pts3d[drawIndex, 2, :, :].detach().cpu().numpy().flatten()[::downsample_rat]
            draw_mono_sampledColor = matlab.double(draw_mono_sampledColor.tolist())
            drawX_mono = matlab.double(drawX_mono.tolist())
            drawY_mono = matlab.double(drawY_mono.tolist())
            drawZ_mono = matlab.double(drawZ_mono.tolist())
            eng.eval('figure(\'visible\', \'off\')', nargout=0)
            h = eng.scatter3(drawX_mono, drawY_mono, drawZ_mono, 5, draw_mono_sampledColor, 'filled', nargout = 0)
            eng.eval('axis equal', nargout = 0)
            xlim = matlab.double([0, 50])
            ylim = matlab.double([-10, 10])
            zlim = matlab.double([-5, 5])
            eng.xlim(xlim, nargout=0)
            eng.ylim(ylim, nargout=0)
            eng.zlim(zlim, nargout=0)
            eng.eval('view([-79 17])', nargout=0)
            eng.eval('camzoom(1.2)', nargout=0)
            eng.eval('grid off', nargout=0)
            # eng.eval('set(gca,\'YTickLabel\',[]);', nargout=0)
            # eng.eval('set(gca,\'XTickLabel\',[]);', nargout=0)
            # eng.eval('set(gca,\'ZTickLabel\',[]);', nargout=0)
            eng.eval('set(gca, \'XColor\', \'none\', \'YColor\', \'none\', \'ZColor\', \'none\')', nargout=0)


            entry_index = inputs['indicesRec'][i].cpu().numpy()
            comps = self.train_filenames[entry_index].split(' ')
            file_save_add = os.path.join('/media/shengjie/other/Depins/Depins/visualization/monodepth2_3dvisualization', comps[0], mapping[comps[2]])
            os.makedirs(file_save_add, exist_ok=True)
            file_save_add = os.path.join(file_save_add, comps[1] + '.png')
            file_save_add = '\'' + file_save_add + '\''
            eng.eval('saveas(gcf,' + file_save_add + ')', nargout=0)
            eng.eval('close all', nargout=0)

            print("Img %d saved" % entry_index)

            # prepare to draw
            # drawIndex = 0
            # projected2d, projecDepth, selector = project_3dptsTo2dpts(pts3d=inputs['velo'], camKs=inputs['camK'])
            # sampledColor = sampleImgs(inputs[('color', 0, 0)], projected2d)
            # drawVelo = inputs['velo'][drawIndex, :, :].cpu().numpy()
            # drawSelector = selector[drawIndex, 0, :].cpu().numpy() > 0
            # drawX = drawVelo[drawSelector, 0]
            # drawY = drawVelo[drawSelector, 1]
            # drawZ = drawVelo[drawSelector, 2]
            # drawColor = sampledColor[drawIndex, :, :].cpu().permute([1, 0]).numpy()[drawSelector, :]
            #
            # drawPts3dVal = pts3d[drawIndex, :, :, :].view(4, -1).cpu().permute([1, 0]).numpy()
            # downsample_rat = 10
            #
            # fig = plt.figure()
            # ax = Axes3D(fig)
            # ax.scatter(drawPts3dVal[::downsample_rat, 0], drawPts3dVal[::downsample_rat, 1],
            #            drawPts3dVal[::downsample_rat, 2], s=0.7, c='r')
            # ax.scatter(drawX, drawY, drawZ, s=1, c=drawColor)
            # plt.show()






options = MonodepthOptions()
opts = options.parse()
if __name__ == "__main__":
    visualizer = Visualizer(opts)
    visualizer.visualize()


