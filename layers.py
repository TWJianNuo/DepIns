# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from scipy.spatial.transform import Rotation as R
import math

import copy

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img, semantics_mask = None):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    if semantics_mask is None:
        return grad_disp_x.mean() + grad_disp_y.mean()
    else:
        return torch.sum((grad_disp_x  + grad_disp_y) * semantics_mask) / (torch.sum(semantics_mask) + 1)


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

class Pooler(nn.Module):
    # This is an irregular roi align framework
    def __init__(self, batch_size, shrinkScale, imageHeight, maxLoad = 100, featureSize = 14):
        super(Pooler, self).__init__()
        self.batch_size = batch_size
        self.maxLoad = maxLoad
        self.featureSize = featureSize
        self.shrinkScale = shrinkScale
        self.imageHeight = imageHeight
        self.indicator = torch.from_numpy(np.array(range(0, self.batch_size))).unsqueeze(1).repeat([1, self.maxLoad]).view(-1).cuda()
        xx, yy = np.meshgrid(range(self.featureSize), range(self.featureSize), indexing='xy')
        self.xx = torch.from_numpy(xx).cuda().float()
        self.yy = torch.from_numpy(yy).cuda().float()

    def translateROIToGrid(self, rois, dscale):
        # To make it 14 by 14 size
        # Input Res Layer 3 features, 16x times downsampled

        # Select out valid proposals
        rois_flatten = rois.view(-1, 4)
        valSel = rois_flatten[:, 0] > 0
        valBInd = self.indicator[valSel]
        valRois = rois_flatten[valSel]

        # Initialize
        valNum = len(valBInd)
        gridxx = self.xx.unsqueeze(0).repeat(valNum, 1, 1)
        gridyy = self.yy.unsqueeze(0).repeat(valNum, 1, 1)

        # Add scale and bias
        # For x
        biasx = valRois[:, 0]
        scalex = valRois[:, 2] - valRois[:, 0]
        gridxx_e = self.scale_bias(gridxx, biasx, scalex, dscale)
        # For y
        biasy = valRois[:, 1] + valBInd.float() * float(self.imageHeight)
        scaley = valRois[:, 3] - valRois[:, 1]
        gridyy_e = self.scale_bias(gridyy, biasy, scaley, dscale)

        return gridxx_e, gridyy_e, valRois, valBInd

    def translateROIToGridFlat(self, rois, dscale):
        # To make it 14 by 14 size
        # Input Res Layer 3 features, 16x times downsampled

        # Select out valid proposals
        rois_flatten = rois.view(-1, 4)
        valSel = rois_flatten[:, 0] > 0
        valBInd = self.indicator[valSel]
        valRois = rois_flatten[valSel]

        # Initialize
        valNum = len(valBInd)
        gridxx = self.xx.unsqueeze(0).repeat(valNum, 1, 1)
        gridyy = self.yy.unsqueeze(0).repeat(valNum, 1, 1)

        # Add scale and bias
        # For x
        biasx = valRois[:, 0]
        scalex = valRois[:, 2] - valRois[:, 0]
        gridxx_e = self.scale_bias(gridxx, biasx, scalex, dscale)
        # For y
        biasy = valRois[:, 1]
        scaley = valRois[:, 3] - valRois[:, 1]
        gridyy_e = self.scale_bias(gridyy, biasy, scaley, dscale)

        return gridxx_e, gridyy_e, valRois, valBInd

    def scale_bias(self, gridcoord, bias, scale, dscale):
        tot_num = len(bias)
        bias_e = bias.unsqueeze(1).expand(-1, self.featureSize * self.featureSize).view(tot_num, self.featureSize, self.featureSize)
        scale_e = scale.unsqueeze(1).expand(-1, self.featureSize * self.featureSize).view(tot_num, self.featureSize, self.featureSize)
        gridcoord_e = gridcoord / float(self.featureSize - 1) * scale_e + bias_e

        gridcoord_e = gridcoord_e / float(dscale)
        return gridcoord_e

    def bilinearSample(self, sfeature, gridx, gridy):
        batch_size, channels, height, width = sfeature.shape

        gridxs = (gridx / (width - 1) - 0.5) * 2
        gridys = (gridy / (height - 1) - 0.5) * 2
        gridxs = gridxs.view(-1, self.featureSize)
        gridys = gridys.view(-1, self.featureSize)
        gridcoords = torch.stack([gridxs, gridys], dim=2).unsqueeze(0)
        sampledFeatures = F.grid_sample(sfeature, gridcoords, mode='bilinear', padding_mode='border')
        # tensor2rgb(sampledFeatures, ind=0).show()
        return sampledFeatures

    def get_grids(self, rois):
        dscale = self.shrinkScale
        gridxx_e, gridyy_e, valRois, valBInd = self.translateROIToGridFlat(rois, dscale)
        # gridxx_e = gridxx_e.unsqueeze(1)
        # gridyy_e = gridyy_e.unsqueeze(1)
        return gridxx_e, gridyy_e, valRois, valBInd

    def forward(self, features, rois, isflapped = True):
        dscale = self.shrinkScale
        gridxx_e, gridyy_e, valRois, valBInd = self.translateROIToGrid(rois, dscale)

        # --For debug--
        # features = F.interpolate(features, size = [int(features.shape[2] / dscale), int(features.shape[3] / dscale)], mode = 'bilinear', align_corners = True)
        # -- End --
        batch_size, channels, height, width = features.shape
        features_flat = features.permute([1,0,2,3])
        features_flat = features_flat.contiguous().view(channels, batch_size * height, width).unsqueeze(0)
        sampledFeatures = self.bilinearSample(features_flat, gridxx_e, gridyy_e)
        if isflapped:
            sampledFeatures = sampledFeatures.view([1, channels, len(valBInd), self.featureSize, self.featureSize]).squeeze(0).permute(1,0,2,3)

        # --Debug visualization--
        # figrgb = tensor2rgb(features_flat, ind = 0)
        # figplt, ax = plt.subplots(1)
        # ax.imshow(figrgb)
        # gridxx_e_flat = gridxx_e.view(len(valBInd), -1).cpu().numpy()
        # gridyy_e_flat = gridyy_e.view(len(valBInd), -1).cpu().numpy()
        # for k in range(len(valBInd)):
        #     plt.scatter(gridxx_e_flat, gridyy_e_flat, s = 0.1, c = 'c')
        # figplt.show()
        # -- End --

        return sampledFeatures, valBInd



# From Project: https://github.com/facebookresearch/maskrcnn-benchmark
# THe following operation is to handle special case when no objects is proposed
class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None

class ConvTranspose2d(torch.nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(ConvTranspose2d, self).forward(x)
        # get output shape

        output_shape = [
            (i - 1) * d - 2 * p + (di * (k - 1) + 1) + op
            for i, p, di, k, d, op in zip(
                x.shape[-2:],
                self.padding,
                self.dilation,
                self.kernel_size,
                self.stride,
                self.output_padding,
            )
        ]
        output_shape = [x.shape[0], self.bias.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)

class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        if x.numel() > 0:
            return super(Conv2d, self).forward(x)
        # get output shape

        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)


class ComputeSurfaceNormal(nn.Module):
    def __init__(self, height, width, batch_size, minDepth, maxDepth):
        super(ComputeSurfaceNormal, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xx = xx.flatten().astype(np.float32)
        yy = yy.flatten().astype(np.float32)
        self.pix_coords = np.expand_dims(np.stack([xx, yy, np.ones(self.width * self.height).astype(np.float32)], axis=1), axis=0).repeat(self.batch_size, axis=0)
        self.pix_coords = torch.from_numpy(self.pix_coords).permute(0,2,1)
        self.ones = torch.ones(self.batch_size, 1, self.height * self.width)
        self.pix_coords = self.pix_coords.cuda()
        self.ones = self.ones.cuda()
        self.init_gradconv()

        self.minDepth = minDepth
        self.maxDepth = maxDepth


    def init_gradconv(self):
        weightsx = torch.Tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([[1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.convy.weight = nn.Parameter(weightsy,requires_grad=False)


    def forward(self, depthMap, invcamK):
        depthMap = depthMap * 0.5 + 0.5
        depthMap = depthMap * (self.maxDepth - self.minDepth) + self.minDepth

        depthMap = depthMap.view(self.batch_size, -1)
        cam_coords = self.pix_coords * torch.stack([depthMap, depthMap, depthMap], dim=1)
        cam_coords = torch.cat([cam_coords, self.ones], dim=1)
        veh_coords = torch.matmul(invcamK, cam_coords)
        veh_coords = veh_coords.view(self.batch_size, 4, self.height, self.width)
        veh_coords = veh_coords
        changex = torch.cat([self.convx(veh_coords[:, 0:1, :, :]), self.convx(veh_coords[:, 1:2, :, :]), self.convx(veh_coords[:, 2:3, :, :])], dim=1)
        changey = torch.cat([self.convy(veh_coords[:, 0:1, :, :]), self.convy(veh_coords[:, 1:2, :, :]), self.convy(veh_coords[:, 2:3, :, :])], dim=1)
        surfnorm = torch.cross(changex, changey, dim=1)
        surfnorm = F.normalize(surfnorm, dim = 1)
        return surfnorm

    def visualization_forward(self, depthMap, invcamK):
        depthMap = depthMap.view(self.batch_size, -1)
        cam_coords = self.pix_coords * torch.stack([depthMap, depthMap, depthMap], dim=1)
        cam_coords = torch.cat([cam_coords, self.ones], dim=1)
        veh_coords = torch.matmul(invcamK, cam_coords)
        veh_coords = veh_coords.view(self.batch_size, 4, self.height, self.width)
        veh_coords = veh_coords
        changex = torch.cat([self.convx(veh_coords[:, 0:1, :, :]), self.convx(veh_coords[:, 1:2, :, :]), self.convx(veh_coords[:, 2:3, :, :])], dim=1)
        changey = torch.cat([self.convy(veh_coords[:, 0:1, :, :]), self.convy(veh_coords[:, 1:2, :, :]), self.convy(veh_coords[:, 2:3, :, :])], dim=1)
        surfnorm = torch.cross(changex, changey, dim=1)
        surfnorm = F.normalize(surfnorm, dim = 1)
        return surfnorm



class BackProj3D(nn.Module):
    def __init__(self, height, width, batch_size):
        super(BackProj3D, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size

        # Init grid points
        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.xx = torch.from_numpy(xx).unsqueeze(0).unsqueeze(0).expand([self.batch_size, 1, -1, -1]).float()
        self.yy = torch.from_numpy(yy).unsqueeze(0).unsqueeze(0).expand([self.batch_size, 1, -1, -1]).float()
        self.pixelLocs = nn.Parameter(torch.cat([self.xx, self.yy], dim=1), requires_grad=False)
    def forward(self, predDepth, invcamK):
        pts3d = backProjTo3d(self.pixelLocs, predDepth, invcamK)
        return pts3d

class SampleDepthMap2PointCloud(nn.Module):
    def __init__(self, height, width, batch_size, ptsCloundNum = 10000):
        super(SampleDepthMap2PointCloud, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.ptsCloundNum = ptsCloundNum
        self.maxDepth = 40

        self.bck = BackProj3D(height=self.height, width=self.width, batch_size=self.batch_size)
        self.bind = torch.arange(0, self.ptsCloundNum).view(1, self.ptsCloundNum).expand(self.batch_size, -1)
        self.bind = self.bind[:, torch.randperm(self.ptsCloundNum)].float()
        self.bind = nn.Parameter(self.bind, requires_grad=False)

        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        ii = torch.arange(0, self.batch_size).view(self.batch_size, 1, 1, 1).expand(-1, 1, self.height, self.width).float()
        xx = torch.from_numpy(xx).unsqueeze(0).unsqueeze(0).expand([self.batch_size, 1, -1, -1]).float()
        yy = torch.from_numpy(yy).unsqueeze(0).unsqueeze(0).expand([self.batch_size, 1, -1, -1]).float()
        self.lind = nn.Parameter(xx + yy * self.width + ii * self.width * self.height, requires_grad=False)

        self.bias_helper = torch.zeros(self.batch_size, self.batch_size)
        for i in range(self.batch_size):
            self.bias_helper[i, 0:i] = 1
        self.bias_helper = nn.Parameter(self.bias_helper, requires_grad=False)
    def forward(self, predDepth, invcamK, semanticLabel):
        pts3d = self.bck(predDepth=predDepth, invcamK=invcamK)
        selector = predDepth < self.maxDepth

        # Random shuffle
        permute_index = torch.randperm(self.width * self.height)

        lind_lineared = self.lind.view(self.batch_size, 1, -1)
        lind_lineared = lind_lineared[:,:,permute_index]

        selector_lineared = selector.view(self.batch_size, 1, -1)
        selector_lineared = selector_lineared[:,:,permute_index]

        # Compute valid points within channel
        valid_number = torch.sum(selector_lineared, dim=[2])
        valid_number = valid_number.float()


        selected_pos = lind_lineared[selector_lineared]

        sampled_ind = torch.remainder(self.bind, (valid_number.clone()).expand([-1, self.ptsCloundNum]))
        sampled_ind = (torch.sum(self.bias_helper * valid_number.t().expand(-1, self.batch_size), dim=1, keepdim=True)).expand(-1, self.ptsCloundNum) + sampled_ind
        sampled_ind = sampled_ind.view(-1)

        selected_pos = selected_pos[sampled_ind.long()].long()

        pts3d_sel = pts3d.permute([0,2,3,1]).contiguous().view(-1, 4)[selected_pos, :]
        pts3d_sel = pts3d_sel.view([self.batch_size, self.ptsCloundNum, 4]).permute([0,2,1])[:,0:3,:]



class DistillPtCloud(nn.Module):
    def __init__(self, height, width, batch_size, ptsCloundNum = 10000):
        super(DistillPtCloud, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.ptsCloundNum = ptsCloundNum
        self.maxDepth = 40

        self.bck = BackProj3D(height=self.height, width=self.width, batch_size=self.batch_size)
        self.bind = torch.arange(0, self.ptsCloundNum).view(1, self.ptsCloundNum).expand(self.batch_size, -1)
        self.bind = self.bind[:, torch.randperm(self.ptsCloundNum)].float()
        self.bind = nn.Parameter(self.bind, requires_grad=False)

        self.bind_helper = torch.arange(0, self.batch_size).view(self.batch_size, 1).expand(-1, self.ptsCloundNum).float()
        self.bind_helper = nn.Parameter(self.bind_helper, requires_grad=False)

        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        ii = torch.arange(0, self.batch_size).view(self.batch_size, 1, 1, 1).expand(-1, 1, self.height, self.width).float()
        xx = torch.from_numpy(xx).unsqueeze(0).unsqueeze(0).expand([self.batch_size, 1, -1, -1]).float()
        yy = torch.from_numpy(yy).unsqueeze(0).unsqueeze(0).expand([self.batch_size, 1, -1, -1]).float()
        self.lind = nn.Parameter(xx + yy * self.width + ii * self.width * self.height, requires_grad=False)

        # 2d Convolution for shrinking
        weights = torch.tensor([[1., 1., 1.],
                                [1., 1., 1.],
                                [1., 1., 1.]])
        weights = weights.view(1, 1, 3, 3)
        self.shrinkConv = nn.Conv2d(1, 1, 3, bias=False, padding=1)
        self.shrinkConv.weight = nn.Parameter(weights, requires_grad=False)
        self.bar = 6

        self.bias_helper = torch.zeros(self.batch_size, self.batch_size)
        for i in range(self.batch_size):
            self.bias_helper[i, 0:i] = 1
        self.bias_helper = nn.Parameter(self.bias_helper, requires_grad=False)

        self.permute_index = torch.randperm(self.width * self.height)

    def forward(self, predDepth, invcamK, semanticLabel, is_shrink = False):
        pts3d = self.bck(predDepth = predDepth, invcamK = invcamK)
        # Visualization
        # draw_index = 0
        # import matlab
        # import matlab.engine
        # self.eng = matlab.engine.start_matlab()
        # draw_x_pred = matlab.double(pts3d[draw_index, 0, :, :].view(-1).detach().cpu().numpy().tolist())
        # draw_y_pred = matlab.double(pts3d[draw_index, 1, :, :].view(-1).detach().cpu().numpy().tolist())
        # draw_z_pred = matlab.double(pts3d[draw_index, 2, :, :].view(-1).detach().cpu().numpy().tolist())
        # self.eng.eval('figure()', nargout=0)
        # self.eng.scatter3(draw_x_pred, draw_y_pred, draw_z_pred, 5, 'r', 'filled', nargout=0)
        # self.eng.eval('axis equal', nargout = 0)
        # xlim = matlab.double([0, 50])
        # ylim = matlab.double([-10, 10])
        # zlim = matlab.double([-5, 5])
        # self.eng.xlim(xlim, nargout=0)
        # self.eng.ylim(ylim, nargout=0)
        # self.eng.zlim(zlim, nargout=0)
        # self.eng.eval('view([-79 17])', nargout=0)
        # self.eng.eval('camzoom(1.2)', nargout=0)
        # self.eng.eval('grid off', nargout=0)



        # Shrink
        # pts3d = nn.Parameter(pts3d, requires_grad=True)

        pts_minNum = 0
        typeind = 5  # Pole
        selector = semanticLabel == typeind
        selector = selector * (predDepth < self.maxDepth)
        if is_shrink:
            selector = self.shrinkConv(selector.float()) > self.bar

        # Random shuffle


        lind_lineared = self.lind.view(self.batch_size, 1, -1)
        lind_lineared = lind_lineared[:,:,self.permute_index]

        selector_lineared = selector.view(self.batch_size, 1, -1)
        selector_lineared = selector_lineared[:,:,self.permute_index]

        # Compute valid points within channel
        valid_number = torch.sum(selector_lineared, dim=[2])
        valid_number = valid_number.float()
        valid_batch_indicator = (valid_number > pts_minNum).float()

        if torch.sum(valid_batch_indicator) == 0:
            return torch.zeros([self.batch_size, 3, self.ptsCloundNum], device = torch.device("cuda"), dtype = torch.float32), valid_batch_indicator

        selected_pos = lind_lineared[selector_lineared]

        tmp = valid_number.clone()
        tmp[tmp == 0] = 1
        sampled_ind = torch.remainder(self.bind, (tmp).expand([-1, self.ptsCloundNum]))

        sampled_ind = (torch.sum(self.bias_helper * valid_number.t().expand(-1, self.batch_size), dim=1, keepdim=True) * valid_batch_indicator.float()).expand(-1, self.ptsCloundNum) + sampled_ind
        sampled_ind = sampled_ind.view(-1)

        selected_pos = selected_pos[sampled_ind.long()].long()

        pts3d_sel = pts3d.permute([0,2,3,1]).contiguous().view(-1, 4)[selected_pos, :]
        pts3d_sel = pts3d_sel.view([self.batch_size, self.ptsCloundNum, 4]).permute([0,2,1])[:,0:3,:]

        # torch.sum(pts3d_sel).backward()
        # a = pts3d.grad
        # torch.sum(torch.abs(a) > 0, dim=[1,2,3])

        # visual_tmp = torch.zeros_like(semanticLabel)
        # visual_tmp.view(-1)[lind_sel] = 1
        # tensor2disp(visual_tmp, ind=0, vmax=1).show()
        # tensor2semantic(semanticLabel, ind=0).show()


        # Visualization
        # draw_index = 0
        # import matlab
        # import matlab.engine
        # self.eng = matlab.engine.start_matlab()
        # draw_x_pred = matlab.double(pts3d_sel[draw_index, 0, :].view(-1).detach().cpu().numpy().tolist())
        # draw_y_pred = matlab.double(pts3d_sel[draw_index, 1, :].view(-1).detach().cpu().numpy().tolist())
        # draw_z_pred = matlab.double(pts3d_sel[draw_index, 2, :].view(-1).detach().cpu().numpy().tolist())
        # self.eng.eval('figure()', nargout=0)
        # self.eng.scatter3(draw_x_pred, draw_y_pred, draw_z_pred, 5, 'r', 'filled', nargout=0)
        # self.eng.eval('axis equal', nargout = 0)
        # xlim = matlab.double([0, 50])
        # ylim = matlab.double([-10, 10])
        # zlim = matlab.double([-5, 5])
        # self.eng.xlim(xlim, nargout=0)
        # self.eng.ylim(ylim, nargout=0)
        # self.eng.zlim(zlim, nargout=0)
        # self.eng.eval('view([-79 17])', nargout=0)
        # self.eng.eval('camzoom(1.2)', nargout=0)
        # self.eng.eval('grid off', nargout=0)
        return pts3d_sel, valid_batch_indicator


class Proj2Oview(nn.Module):
    def __init__(self, height, width, batch_size):
        super(Proj2Oview, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size

        self.oview = 0
        self.bck = BackProj3D(height=self.height, width=self.width, batch_size=self.batch_size)

        self.o_trans = np.array([-3, 6, 5]) # x, y, z
        self.o_rot = np.array([-15, 15, 0])  # yaw, pitch, roll
        self.mv2aff()

        weights = torch.tensor([[1., 1., 1.],
                                [1., 1., 1.],
                                [1., 1., 1.]])
        weights = weights.view(1, 1, 3, 3)
        self.bar = 6
        self.shrinkConv = nn.Conv2d(1, 1, 3, bias=False, padding=1)
        self.shrinkConv.weight = nn.Parameter(weights, requires_grad=False)

        self.depthbar = 60

        # Parzen Window Size
        self.pws = 5 # 5 by 5 window
        weights = torch.ones([self.pws, self.pws])
        weights = weights.view(1, 1, self.pws, self.pws)
        self.expandConv = nn.Conv2d(1, 1, self.pws, bias=False, padding=1)
        self.expandConv.weight = nn.Parameter(weights, requires_grad=False)

        # Indice helper
        self.chhelper = torch.arange(0, self.batch_size).view(self.batch_size, 1, 1, 1).expand([-1, 1, self.height, self.width]).float()
        self.chhelper = nn.Parameter(self.chhelper, requires_grad=False)

        # Gaussian Kernl
        self.gausk_s = 3
        self.gausk = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = self.gausk_s, stride = 1, padding = int((self.gausk_s - 1) / 2), bias = False)
        self.gausk.weight = nn.Parameter(self.get_gausk_w(kernel_size=self.gausk_s, sigma=1), requires_grad=False)

        # Multi-scale Number
        self.mscale = 3

        self.eng = None
        self.dsrate = 10

        # Init grid points
        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.xx = torch.from_numpy(xx).unsqueeze(0).unsqueeze(0).expand([self.batch_size, 1, -1, -1]).float().cuda()
        self.yy = torch.from_numpy(yy).unsqueeze(0).unsqueeze(0).expand([self.batch_size, 1, -1, -1]).float().cuda()
        self.ones = torch.ones_like(self.xx).cuda()
        self.pixelocs = torch.cat([self.xx, self.yy, self.ones], dim=1)

    def get_gausk_w(self, kernel_size=3, sigma=2):
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

        return gaussian_kernel

    def mv2aff(self):
        x, y, z = self.o_trans
        T = np.matrix([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

        yaw, pitch, roll = self.o_rot / 180 * np.pi
        yawMatrix = np.matrix([
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw), np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        pitchMatrix = np.matrix([
            [np.cos(pitch), 0, np.sin(pitch), 0],
            [0, 1, 0, 0],
            [-np.sin(pitch), 0, np.cos(pitch), 0],
            [0, 0, 0, 1]
        ])

        rollMatrix = np.matrix([
            [1, 0, 0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll), np.cos(roll), 0],
            [0, 0, 0, 1]
        ])

        R = yawMatrix * pitchMatrix * rollMatrix

        # self.mvaff = nn.Parameter(torch.from_numpy(T @ R), requires_grad=False)
        # self.mvaff = torch.from_numpy(T @ R).float().cuda()
        self.R = torch.from_numpy(R).float().cuda()
        self.T = torch.from_numpy(T).float().cuda()

    def mv_cam(self, extrinsic, R, T):
        campos = torch.inverse(extrinsic) @ torch.from_numpy(np.array([[0], [0], [0], [1]], dtype=np.float32)).cuda()
        campos_mvd = T @ campos

        r_ex = torch.transpose(R[0:3,0:3] @ torch.transpose(extrinsic[:,0:3,0:3], 1, 2), 1, 2)
        t_ex = - r_ex @ campos_mvd[:,0:3,:]
        # t_ex = torch.cat([t_ex, torch.ones([self.batch_size, 1, 1], dtype=torch.float, device=torch.device("cuda"))], dim=1)
        nextrinsic = torch.cat([r_ex, t_ex], dim=2)
        nextrinsic = torch.cat([nextrinsic, torch.from_numpy(np.array([[[0,0,0,1]]], dtype=np.float32)).expand(self.batch_size, -1, -1).cuda()], dim=1)
        return nextrinsic

    def erpipolar_rendering(self, depthmap, semanticmap, intrinsic, extrinsic):
        # Compute Mask
        addmask = self.post_mask(depthmap = depthmap, semanticmap = semanticmap)

        invcamK = torch.inverse(intrinsic @ extrinsic)
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)
        projected2d, projecteddepth, selector = self.proj2d(pts3d=pts3d, intrinsic=intrinsic, extrinsic=extrinsic, R=self.R, T=self.T, addmask = addmask)

        self.get_essM(intrinsic=intrinsic, extrinsic=extrinsic, R=self.R, T=self.T, projected2d=projected2d, projecteddepth=projecteddepth, pts3d=pts3d, depthmap=depthmap)

    def convertT_(self, T):
        T_ = torch.zeros([self.batch_size, 3, 3], device=torch.device("cuda"), dtype=torch.float32)

        T_[:, 0, 1] = -T[:, 2, 0]
        T_[:, 0, 2] = T[:, 1, 0]
        T_[:, 1, 0] = T[:, 2, 0]
        T_[:, 1, 2] = -T[:, 0, 0]
        T_[:, 2, 0] = -T[:, 1, 0]
        T_[:, 2, 1] = T[:, 0, 0]
        return T_

    def check_2ptsray(self, projectionMatrix, cameraPos, pts3d):
        c = 0
        xx = 20
        yy = 25
        num = 100
        projectionMatrix_np = projectionMatrix[c].detach().cpu().numpy()
        cameraPos_np = cameraPos[c].detach().cpu().numpy()
        cameraPos_np = cameraPos_np[:,0][0:3]
        pts3d_np = pts3d[c, :, yy, xx].detach().cpu().numpy()
        pts3d_np = pts3d_np[0:3]

        lam = np.arange(0.5, num, 1)
        lam = np.repeat(np.expand_dims(lam, axis = 0), 3, axis = 0)
        pts3d_np_e = lam * np.repeat(np.expand_dims(cameraPos_np, axis = 1), num, axis = 1) + (1 - lam) * np.repeat(np.expand_dims(pts3d_np, axis = 1), num, axis = 1)
        # pts3d_np_e = lam * np.repeat(np.expand_dims(cameraPos_np, axis=1), num, axis=1)
        pts3d_np_e = np.concatenate([pts3d_np_e, np.ones([1, num])], axis=0)

        pts3d_np_e_p = projectionMatrix_np @ pts3d_np_e
        x = pts3d_np_e_p[0, :] / pts3d_np_e_p[2, :]
        y = pts3d_np_e_p[1, :] / pts3d_np_e_p[2, :]
        if np.var(x) + np.var(y) < 1e-3:
            return True
        else:
            return False

    def check_epipole(self, epipole_cam2, epipole3d, projectionM_old, projectionM_new):
        c = 0
        xx = 120
        yy = 25

        d1 = 2
        d2 = 300
        d3 = 40

        projectionM_old_np = projectionM_old[c, :, :].detach().cpu().numpy()
        projectionM_new_np = projectionM_new[c, :, :].detach().cpu().numpy()
        epipole_np = epipole_cam2[c, :, :].detach().cpu().numpy()
        epipole_x = epipole_np[0] / epipole_np[2]
        epipole_y = epipole_np[1] / epipole_np[2]


        pts1_c = np.array([[xx * d1, yy * d1, d1, 1]]).transpose()
        pts1_w = np.linalg.inv(projectionM_old_np) @ pts1_c
        pts1_cc = projectionM_new_np @ pts1_w
        pts1x = pts1_cc[0] / pts1_cc[2]
        pts1y = pts1_cc[1] / pts1_cc[2]

        pts2_c = np.array([[xx * d2, yy * d2, d2, 1]]).transpose()
        pts2_w = np.linalg.inv(projectionM_old_np) @ pts2_c
        pts2_cc = projectionM_new_np @ pts2_w
        pts2x = pts2_cc[0] / pts2_cc[2]
        pts2y = pts2_cc[1] / pts2_cc[2]

        pts3_c = np.array([[xx * d3, yy * d3, d3, 1]]).transpose()
        pts3_w = np.linalg.inv(projectionM_old_np) @ pts3_c
        pts3_cc = projectionM_new_np @ pts3_w
        pts3x = pts3_cc[0] / pts3_cc[2]
        pts3y = pts3_cc[1] / pts3_cc[2]


        # 2d colinearity,rat1 should be equal to rat2
        rat1 = (epipole_y - pts1y) / (epipole_x - pts1x)
        rat2 = (epipole_y - pts2y) / (epipole_x - pts2x)

        rat1 = (pts3y - pts1y) / (pts3x - pts1x)
        rat2 = (pts3y - pts2y) / (pts3x - pts2x)

        # 3d colinearity, l1, l2, l3 sum of the shorter two equals longest one
        epipole3d_np = epipole3d[c,:].detach().cpu().numpy()
        l1 = np.sqrt(np.sum((pts1_w[0:3] - pts2_w[0:3]) * (pts1_w[0:3] - pts2_w[0:3])))
        l2 = np.sqrt(np.sum((epipole3d_np[0:3] - pts2_w[0:3]) * (epipole3d_np[0:3] - pts2_w[0:3])))
        l3 = np.sqrt(np.sum((epipole3d_np[0:3] - pts1_w[0:3]) * (epipole3d_np[0:3] - pts1_w[0:3])))

        l1 = np.sqrt(np.sum((pts1_cc[0:3] - pts2_cc[0:3]) * (pts1_cc[0:3] - pts2_cc[0:3])))
        l2 = np.sqrt(np.sum((epipole_np[0:3] - pts2_cc[0:3]) * (epipole_np[0:3] - pts2_cc[0:3])))
        l3 = np.sqrt(np.sum((epipole_np[0:3] - pts1_cc[0:3]) * (epipole_np[0:3] - pts1_cc[0:3])))

    def debug_FM(self, projectionM_old, projectionM_new, Cold):
        c = 0
        xx = 120
        yy = 25

        d1 = 2
        d2 = 3
        d3 = 40

        projectionM_old_np = projectionM_old[c, :, :].detach().cpu().numpy()
        projectionM_new_np = projectionM_new[c, :, :].detach().cpu().numpy()

        Cold_np = Cold[c, :, :].detach().cpu().numpy()
        Cold_np_new = projectionM_new_np @ Cold_np

        pts1_c = np.array([[xx * d1, yy * d1, d1, 1]]).transpose()
        pts1_w = np.linalg.inv(projectionM_old_np) @ pts1_c
        pts1_cc = projectionM_new_np @ pts1_w

        pts2_c = np.array([[xx * d2, yy * d2, d2, 1]]).transpose()
        pts2_w = np.linalg.inv(projectionM_old_np) @ pts2_c
        pts2_cc = projectionM_new_np @ pts2_w

        pts3_c = np.array([[xx * d3, yy * d3, d3, 1]]).transpose()
        pts3_w = np.linalg.inv(projectionM_old_np) @ pts3_c
        pts3_cc = projectionM_new_np @ pts3_w

        l = np.cross(Cold_np_new[0:3,0], pts1_cc[0:3,0])
        l = np.expand_dims(l, axis=1)
        pts2_cc[0:3, :].transpose() @ l

        Cold_np_new_ = np.zeros([3,3])
        Cold_np_new_[0, 1] = -Cold_np_new[2, 0]
        Cold_np_new_[0, 2] = Cold_np_new[1, 0]
        Cold_np_new_[1, 0] = Cold_np_new[2, 0]
        Cold_np_new_[1, 2] = -Cold_np_new[0, 0]
        Cold_np_new_[2, 0] = -Cold_np_new[1, 0]
        Cold_np_new_[2, 1] = Cold_np_new[0, 0]
        ll = Cold_np_new_ @ ((projectionM_new_np @ (np.linalg.inv(projectionM_old_np) @ pts1_c))[0:3,:])
        pts1_cc[0:3, :].transpose() @ ll

        c1 = (projectionM_new_np[0:3,:] @ (np.linalg.pinv(projectionM_old_np[0:3,:]) @ pts1_c[0:3]))
        c2 = (projectionM_new_np[0:3,:] @ (Cold_np))
        lll = np.expand_dims(np.cross(c2[:,0], c1[:,0]), axis=1)
        t1 = (projectionM_new_np @ ((np.linalg.pinv(projectionM_old_np[0:3,:]) @ pts1_c[0:3]) + 10 * Cold_np))[0:3,:]
        t2 = (projectionM_new_np @ ((np.linalg.pinv(projectionM_old_np[0:3, :]) @ pts1_c[0:3]) + 30 * Cold_np))[0:3, :]
        t3 = (projectionM_new_np @ ((np.linalg.pinv(projectionM_old_np[0:3, :]) @ pts1_c[0:3]) + 20 * Cold_np))[0:3, :]

        a1 = np.expand_dims(np.concatenate([c1[0] / c1[2], c1[1] / c1[2], np.array([1])], axis=0), axis=1)
        a2 = np.expand_dims(np.concatenate([c2[0] / c2[2], c2[1] / c2[2], np.array([1])], axis=0), axis=1)
        a3 = np.cross(a1[:,0], a2[:,0])

        t2t = np.expand_dims(np.concatenate([t2[0] / t2[2], t2[1] / t2[2], np.array([1])], axis=0), axis=1)
        t3t = np.expand_dims(np.concatenate([t3[0] / t3[2], t3[1] / t3[2], np.array([1])], axis=0), axis=1)
        t2t.transpose() @ a3
        t3t.transpose() @ a3

        l1 = np.sqrt(np.sum((t1 - t2)**2))
        l2 = np.sqrt(np.sum((t1 - t3) ** 2))
        l3 = np.sqrt(np.sum((t2 - t3) ** 2))

        pts1_cc[0:3, :].transpose() @ lll


        i1 = projectionM_new @ Cold
        i1_ = self.convertT_(i1[:, 0:3, :])
        i1__np = i1_[c,:,:].detach().cpu().numpy()
        FM = i1_ @ (projectionM_new @ torch.inverse(projectionM_old))[:, 0:3, :]
        FM_np = FM[c, :, :].detach().cpu().numpy()

        a = (projectionM_new @ torch.inverse(projectionM_old))[:, 0:3, :]
        a = a[c, :, :].cpu().numpy()
        a @ pts1_c


        b = (projectionM_new[0].cpu().numpy() @ torch.inverse(projectionM_old)[0].cpu().numpy()) @ np.array([[xx * d1, yy * d1, d1, 1]]).transpose()
        pts1_cc[0:3, :].transpose() @ i1__np @ b[0:3,:]

        pts1_cc[0:3, :].transpose() @ FM_np @ pts1_c


    def org_intrinsic(self, intrinsic):
        intrinsic_44 = copy.deepcopy(intrinsic)
        intrinsic_44[:, 0:3, 3] = 0
        added_extrinsic = torch.inverse(intrinsic_44) @ intrinsic
        return intrinsic_44, added_extrinsic

    def check_FM(self, projectionM_old, projectionM_new, pts3d, FM):
        c = 0
        xx = 90
        yy = 30

        pts3d_np = pts3d[c, :, yy, xx].detach().cpu().numpy()
        pts3d_np = np.expand_dims(pts3d_np, axis=1)

        projectionM_old_np = projectionM_old[c, :, :].detach().cpu().numpy()
        projectionM_new_np = projectionM_new[c, :, :].detach().cpu().numpy()
        FM_np = FM[c, :, :].detach().cpu().numpy()

        pts2d_old = projectionM_old_np @ pts3d_np
        pts2d_old = pts2d_old[0:3, :]
        pts2d_new = projectionM_new_np @ pts3d_np
        pts2d_new = pts2d_new[0:3, :]

        pts2d_new.transpose() @ FM_np @ pts2d_old

    def get_essM(self, intrinsic, extrinsic, R, T, projected2d, projecteddepth, pts3d, depthmap):
        intrinsic_44, added_extrinsic = self.org_intrinsic(intrinsic)
        extrinsic_old = added_extrinsic @ extrinsic
        extrinsic_new = added_extrinsic @ self.mv_cam(extrinsic=extrinsic, R=R, T=T)

        Pold = intrinsic_44 @ extrinsic_old
        Pnew = intrinsic_44 @ extrinsic_new
        Cold = torch.inverse(Pold) @ torch.tensor([0, 0, 0, 1]).view(1, 4, 1).expand([self.batch_size, -1, -1]).float().cuda()

        # Pold34 = Pold[:, 0:3, :]
        # Pnew34 = Pnew[:, 0:3, :]
        # Pold34_pinv = torch.transpose(Pold34, dim0=1, dim1=2) @ torch.inverse(Pold34 @ torch.transpose(Pold34, dim0=1, dim1=2))

        rand_dmap = torch.rand([self.batch_size, 1, self.height, self.width]).float().cuda()
        # randx = torch.cat([self.xx * rand_dmap, self.yy * rand_dmap, rand_dmap], dim=1)
        randx = torch.cat([self.xx * rand_dmap, self.yy * rand_dmap, rand_dmap, self.ones], dim=1)

        # Cold_new = Pnew34 @ Cold
        Cold_new = Pnew @ Cold
        Cold_new = Cold_new[:, 0:3, :]
        Cold_new[:, 0, :] = Cold_new[:, 0, :] / Cold_new[:, 2, :]
        Cold_new[:, 1, :] = Cold_new[:, 1, :] / Cold_new[:, 2, :]
        Cold_new[:, 2, :] = Cold_new[:, 2, :] / Cold_new[:, 2, :]
        Cold_new_m = self.convertT_(Cold_new)

        # tmpM = Pnew34 @ Pold34_pinv
        tmpM = Pnew @ torch.inverse(Pold)
        tmpM = tmpM.view(self.batch_size, 1, 1, 4, 4).expand(-1, self.height, self.width, -1, -1)

        randx_e = randx.permute([0,2,3,1]).unsqueeze(4)
        randx_new = torch.matmul(tmpM, randx_e)

        randx_new = randx_new[:,:,:,0:3,:]
        randx_new[:, :, :, 0, :] = randx_new[:, :, :, 0, :] / randx_new[:, :, :, 2, :]
        randx_new[:, :, :, 1, :] = randx_new[:, :, :, 1, :] / randx_new[:, :, :, 2, :]
        randx_new[:, :, :, 2, :] = randx_new[:, :, :, 2, :] / randx_new[:, :, :, 2, :]

        epipoLine = Cold_new_m.view(self.batch_size, 1, 1, 3, 3).expand(-1, self.height, self.width, -1, -1) @ randx_new

        projected2d_e = torch.cat([projected2d, torch.ones([self.batch_size, 1, self.height, self.width]).float().cuda()], dim = 1)
        ck = torch.sum(projected2d_e.permute([0,2,3,1]).unsqueeze(4) * epipoLine, dim=[3,4])







        i1 = Pnew @ Cold
        i1_ = self.convertT_(i1[:, 0:3, :])
        i2 = (Pnew @ torch.inverse(Pold))[:, 0:3, 0:3]
        FM = i1_ @ i2

        self.check_FM(projectionM_old=Pold, projectionM_new=Pnew, pts3d=pts3d, FM=FM)

        Pold = intrinsic @ extrinsic
        Cold = torch.inverse(Pold) @ torch.tensor([0, 0, 0, 1]).view(1,4,1).expand([self.batch_size, -1, -1]).float().cuda()
        # Cold = torch.inverse(extrinsic) @ torch.tensor([0, 0, 0, 1]).view(1, 4, 1).expand([self.batch_size, -1, -1]).float().cuda()
        assert self.check_2ptsray(projectionMatrix=Pold, cameraPos=Cold, pts3d=pts3d) == True

        Exnew = self.mv_cam(extrinsic=extrinsic, R=R, T=T)
        # intrinsic_tmp = copy.deepcopy(intrinsic)
        # intrinsic_tmp[:, 0:3, 3] = 0
        # Pnew = intrinsic_tmp @ Exnew
        Pnew = intrinsic @ Exnew

        i1 = Pnew @ Cold
        i1_ = self.convertT_(i1[:, 0:3, :])
        i2 = (Pnew @ torch.inverse(Pold))[:, 0:3, 0:3]
        FM = i1_ @ i2
        self.debug_FM(projectionM_old = Pold, projectionM_new = Pnew, Cold = Cold)
        self.check_epipole(epipole_cam2 = i1, epipole3d = Cold, projectionM_old = Pold, projectionM_new = Pnew)


        nextrinsic = self.mv_cam(extrinsic=extrinsic, R=R, T=T)
        RT_old2new = nextrinsic @ torch.inverse(extrinsic)
        Rn = RT_old2new[:, 0:3, 0:3]
        Tn = RT_old2new[:, 0:3, 3].unsqueeze(2)

        assert self.check_RT(pts3d=pts3d, extrinsic=extrinsic, nextrinsic=nextrinsic, Rn=Rn, Tn=Tn) == True

        T_ = self.convertT_(Tn)
        essM = T_ @ Rn

        p_intrinsic = intrinsic[:,0:3,0:3]
        p_extrinsic_old = (intrinsic @ extrinsic)[:,0:3,0:3] @ torch.inverse(p_intrinsic)

        x = 20
        y = 20
        c = 0
        x2 = torch.cat([projected2d[c, :, y, x], torch.Tensor([1]).cuda()], dim=0).unsqueeze(1).cuda()
        x1 = torch.Tensor([x, y, 1]).unsqueeze(1).cuda()
        X2 = x2 * projecteddepth[c, 0, y, x]
        X1 = x1 * depthmap[c, 0, y, x]
        X2 - (Rn[c] @ X1 + Tn[c])

        essM_e = essM.view(self.batch_size, 1, 1, 3, 3).expand(-1, self.height, self.width, -1, -1)
        epipoG = essM_e @ self.pixelocs.permute([0,2,3,1]).unsqueeze(4)

        res = epipoG.squeeze(4)
        res = res[:,:,:,0] * projected2d[:,0,:,:] + res[:,:,:,1] * projected2d[:,1,:,:] + res[:,:,:,2]

        self.check_epipoG(epipoG = epipoG, intrinsic = intrinsic, extrinsic = extrinsic, nextrinsic = nextrinsic)

    def check_RT(self, pts3d, extrinsic, nextrinsic, Rn, Tn):
        pts3d_old = extrinsic.view(self.batch_size, 1, 1, 4, 4).expand(-1, self.height, self.width, -1, -1) @ pts3d.permute([0,2,3,1]).unsqueeze(4)
        pts3d_old = pts3d_old[:,:,:,0:3,:]
        pts3d_new = nextrinsic.view(self.batch_size, 1, 1, 4, 4).expand(-1, self.height, self.width, -1, -1) @ pts3d.permute([0,2,3,1]).unsqueeze(4)
        pts3d_new = pts3d_new[:,:,:,0:3,:]

        pts3d_new_recon = Rn.view(self.batch_size, 1, 1, 3, 3).expand(-1, self.height, self.width, -1, -1) @ pts3d_old + Tn.view(self.batch_size, 1, 1, 3, 1).expand(-1, self.height, self.width, -1, -1)
        dis = torch.mean(torch.abs(pts3d_new - pts3d_new_recon))
        if dis < 1e-1:
            return True
        else:
            return False

    def check_epipoG(self, epipoG, intrinsic, extrinsic, nextrinsic):
        rndnum = 100
        maxDepth = 50
        minDepth = 2
        rd = int(np.random.randint(self.batch_size, size=1))

        extrinsic_np = extrinsic[rd, :, :].detach().cpu().numpy()
        nextrinsic_np = nextrinsic[rd, :, :].detach().cpu().numpy()
        intrinsic_np = intrinsic[rd, :, :].detach().cpu().numpy()

        rx = np.random.randint(self.width, size=rndnum)
        ry = np.random.randint(self.height, size=rndnum)
        rz = np.random.rand(100) * (maxDepth - minDepth) + minDepth
        ro = np.ones(rndnum)
        epipoG_np = epipoG[rd, ry.tolist(), rx.tolist(), :, :].squeeze(2).detach().cpu().numpy()

        pts2d = np.stack([rx * rz, ry * rz, rz, ro], axis=1)
        pts3d = (np.linalg.inv(intrinsic_np * extrinsic_np) @ pts2d.transpose()).transpose()

        pts3dn = (intrinsic_np @ nextrinsic_np @ pts3d.transpose()).transpose()
        pts3dn2d = np.stack([pts3dn[:, 0] / pts3dn[:, 2], pts3dn[:, 1] / pts3dn[:, 2], np.ones(rndnum)], axis=1)

        p2l = epipoG_np[:,0] * pts3dn2d[:,0] + epipoG_np[:,1] * pts3dn2d[:,1] + epipoG_np[:,2] * pts3dn2d[:,2]






    def proj2d(self, pts3d, intrinsic, extrinsic, R, T, addmask = None):
        nextrinsic = self.mv_cam(extrinsic=extrinsic, R=R, T=T)
        camKs = intrinsic @ nextrinsic

        camKs_e = camKs.view(self.batch_size, 1, 1, 4, 4).expand(-1, self.height, self.width, -1, -1)
        pts3d_e = pts3d.permute([0,2,3,1]).unsqueeze(4)
        projected3d = torch.matmul(camKs_e, pts3d_e).squeeze(4).permute(0, 3, 1, 2)

        projected2d = torch.stack([projected3d[:, 0, :, :] / projected3d[:, 2, :, :], projected3d[:, 1, :, :] / projected3d[:, 2, :, :]], dim=1)
        projecteddepth = projected3d[:, 2, :, :].unsqueeze(1)
        selector = (projected2d[:, 0, :] > 0) * (projected2d[:, 0, :] < self.width- 1) * (projected2d[:, 1, :] > 0) * (
                projected2d[:, 1, :] < self.height - 1) * (projecteddepth[:,0,:,:] > 0)
        selector = selector.unsqueeze(1)

        if addmask is not None:
            selector = selector * addmask
        return projected2d, projecteddepth, selector

    def debug_visualization(self, depthmap, semanticmap, intrinsic, extrinsic):
        invcamK = torch.inverse(intrinsic @ extrinsic)
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)

        self.o_trans = np.array([-3, 6, 5]) # x, y, z
        self.o_rot = np.array([-15, 15, 0])  # yaw, pitch, roll
        self.mv2aff()

        projected2d, projecteddepth, selector = self.proj2d(pts3d=pts3d, intrinsic=intrinsic, extrinsic=extrinsic, R=self.R, T=self.T)
        self.visualize3d(pts3d = pts3d, extrinsic = extrinsic, R = self.R, T = self.T)
        self.visualize2d(projected2d = projected2d, projecteddepth = projecteddepth)
        print("visible ratio % f" % (torch.sum(selector).float() / self.batch_size / self.width / self.height))

    def post_mask(self, depthmap, semanticmap):
        # Post Semantic Mask
        visibletype = [5]  # pole
        # visibletype = [2, 3, 4]  # building, wall, fence
        # visibletype = [11, 12, 16, 17]  # person, rider, motorcycle, bicycle
        # visibletype = [13, 14, 15, 16]  # car, truck, bus, train
        addmask = torch.zeros_like(semanticmap)
        for vt in visibletype:
            addmask = addmask + (semanticmap == vt)
        addmask = addmask > 0

        # Shrink Semantic Mask
        addmask = self.shrinkConv(addmask.float()) > self.bar

        # Only supervise close area
        addmask = addmask * (depthmap < self.depthbar)

        return addmask

    def print(self, depthmap, semanticmap, intrinsic, extrinsic, ppath):
        self.dsrate = 1
        addmask = self.post_mask(depthmap = depthmap, semanticmap = semanticmap)

        invcamK = torch.inverse(intrinsic @ extrinsic)
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)
        projected2d, projecteddepth, selector = self.proj2d(pts3d=pts3d, intrinsic=intrinsic, extrinsic=extrinsic, R=self.R, T=self.T, addmask = addmask)

        # Print
        self.print2d(projected2d = projected2d, projecteddepth = projecteddepth, selector = selector, ppath = ppath)

    def pdf_estimation(self, depthmap, semanticmap, intrinsic, extrinsic):
        # Compute Mask
        addmask = self.post_mask(depthmap = depthmap, semanticmap = semanticmap)

        invcamK = torch.inverse(intrinsic @ extrinsic)
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)
        projected2d, projecteddepth, selector = self.proj2d(pts3d=pts3d, intrinsic=intrinsic, extrinsic=extrinsic, R=self.R, T=self.T, addmask = addmask)

        sx = projected2d[:, 0, :, :][selector.squeeze(1)]
        sy = projected2d[:, 1, :, :][selector.squeeze(1)]
        sc = self.chhelper[selector]

        rendered_hmap_rlaxs = dict()
        for i in range(self.mscale):
            resacle_fac = 2 ** i
            ssx = sx / resacle_fac
            ssy = sy / resacle_fac
            rsx = torch.round(ssx)
            rsy = torch.round(ssy)
            val2d = torch.sigmoid(torch.sqrt((ssx - rsx).pow(2) + (ssy - rsy).pow(2)))

            # Assign value
            rendered_hmap = torch.zeros([self.batch_size, 1, self.height // resacle_fac, self.width // resacle_fac], dtype=torch.float32, device=torch.device("cuda"))
            rendered_hmap[sc.long(), 0, rsy.long(), rsx.long()] = val2d

            # Parzen Density Estimation
            rendered_hmap_rlax = self.gausk(rendered_hmap)
            rendered_hmap_rlaxs[str(i)] = F.interpolate(rendered_hmap_rlax, size=[self.height, self.width], mode='bilinear', align_corners=False)

        return rendered_hmap_rlaxs

    def print2d(self, projected2d, projecteddepth, selector, ppath):
        import matlab
        import matlab.engine
        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
        for draw_index in range(self.batch_size):
            if torch.sum(selector[draw_index]) > 100:

                xx = projected2d[draw_index, 0, selector[draw_index, 0, :]]
                yy = projected2d[draw_index, 1, selector[draw_index, 0, :]]
                dd = projecteddepth[draw_index, 0, selector[draw_index, 0, :]]

                draw_x_pred = matlab.double(xx.detach()[::self.dsrate].cpu().numpy().tolist())
                draw_y_pred = matlab.double(yy.detach()[::self.dsrate].cpu().numpy().tolist())
                draw_z_pred = matlab.double((1.0 / (dd.detach()[::self.dsrate].cpu().numpy())).tolist())
                self.eng.eval('figure(\'visible\', \'off\')', nargout=0)
                self.eng.scatter(draw_x_pred, draw_y_pred, 1, draw_z_pred, nargout=0)
                self.eng.eval('axis equal', nargout = 0)
                self.eng.eval('xlim([{} {}])'.format(0, self.width), nargout=0)
                self.eng.eval('ylim([{} {}])'.format(0, self.height), nargout=0)
                self.eng.eval('axis ij'.format(0, self.height), nargout=0)
                svc = 'saveas(gcf,\'{}.png\')'.format(ppath[draw_index])
                self.eng.eval(svc, nargout=0)
                self.eng.eval('close all', nargout=0)

    def visualize2d(self, projected2d, projecteddepth):
        draw_index = 0
        import matlab
        import matlab.engine
        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
        draw_x_pred = matlab.double(projected2d[draw_index, 0, :].view(-1).detach()[::self.dsrate].cpu().numpy().tolist())
        draw_y_pred = matlab.double(projected2d[draw_index, 1, :].view(-1).detach()[::self.dsrate].cpu().numpy().tolist())
        draw_z_pred = matlab.double((1.0 / (projecteddepth[draw_index, 0, :].view(-1).detach()[::self.dsrate].cpu().numpy())).tolist())
        self.eng.eval('figure()', nargout=0)
        self.eng.scatter(draw_x_pred, draw_y_pred, 1, draw_z_pred, nargout=0)
        self.eng.eval('axis equal', nargout = 0)
        self.eng.eval('xlim([{} {}])'.format(0, self.width), nargout=0)
        self.eng.eval('ylim([{} {}])'.format(0, self.height), nargout=0)
        self.eng.eval('axis ij'.format(0, self.height), nargout=0)
        # self.eng.eval('colormap jet', nargout=0)

    def visualize3d(self, pts3d, extrinsic, R, T):
        nextrinsic = self.mv_cam(extrinsic=extrinsic, R = R, T = T)

        draw_index = 0
        import matlab
        import matlab.engine
        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
        draw_x_pred = matlab.double(pts3d[draw_index, 0, :].view(-1).detach()[::self.dsrate].cpu().numpy().tolist())
        draw_y_pred = matlab.double(pts3d[draw_index, 1, :].view(-1).detach()[::self.dsrate].cpu().numpy().tolist())
        draw_z_pred = matlab.double(pts3d[draw_index, 2, :].view(-1).detach()[::self.dsrate].cpu().numpy().tolist())

        nnextrinsicn = nextrinsic[draw_index, :, :].detach().cpu().numpy()
        camPos = np.linalg.inv(nnextrinsicn) @ np.array([[0], [0], [0], [1]])
        camDir = np.linalg.inv(nnextrinsicn) @ np.array([[0], [0], [1], [1]]) - camPos
        draw_camPos = matlab.double(camPos.tolist())
        draw_camDir = matlab.double(camDir.tolist())

        self.eng.eval('figure()', nargout=0)
        self.eng.scatter3(draw_x_pred, draw_y_pred, draw_z_pred, 5, 'g', 'filled', nargout=0)
        self.eng.eval('axis equal', nargout = 0)
        xlim = matlab.double([0, 50])
        ylim = matlab.double([-10, 10])
        zlim = matlab.double([-5, 5])
        self.eng.xlim(xlim, nargout=0)
        self.eng.ylim(ylim, nargout=0)
        self.eng.zlim(zlim, nargout=0)
        self.eng.eval('view([-79 17])', nargout=0)
        self.eng.eval('camzoom(1.2)', nargout=0)
        self.eng.eval('grid off', nargout=0)
        self.eng.eval('hold on', nargout=0)
        self.eng.scatter3(draw_camPos[0], draw_camPos[1], draw_camPos[2], 30, 'r', 'filled', nargout=0)
        self.eng.eval('hold on', nargout=0)
        self.eng.quiver3(draw_camPos[0], draw_camPos[1], draw_camPos[2], draw_camDir[0], draw_camDir[1], draw_camDir[2], nargout=0)



