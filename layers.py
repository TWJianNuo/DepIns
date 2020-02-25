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
from Oview_Gan import eppl_render, eppl_pix2dgrad2depth, eppl_pix2dgrad2depth_l2

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
        # self.selfmv2aff()

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

        # Init Gaussian Sigma
        self.biasParam = 1
        # sigma = np.array([[self.biasParam, 0], [0, 0.03]])
        sigma = np.array([[self.biasParam, 0], [0, 0.2]])
        self.sigma = torch.from_numpy(sigma).float().cuda()
        self.sigma = self.sigma.view(1,1,1,2,2).expand(self.batch_size, self.height, self.width, -1, -1)

        # Init kernel window size
        self.kws = 2.3
        # Init sense range
        self.sr = 7

        self.eps = 1e-6
        self.sampleNum = 10

    def get_camtrail(self, extrinsic, intrinsic):
        ws = 3
        y = 10
        d = 20
        xl = 0 - ws
        xr = self.width + ws
        lw = torch.linspace(start=0.1, end=0.9, steps=self.sampleNum).cuda()
        lw = lw.view(1,self.sampleNum,1,1).expand(self.batch_size, -1, 4, 1)

        projM = intrinsic @ extrinsic
        campos = torch.inverse(extrinsic) @ torch.from_numpy(np.array([[0], [0], [0], [1]], dtype=np.float32)).cuda()
        camdir = torch.inverse(extrinsic) @ torch.from_numpy(np.array([[0], [0], [1], [1]], dtype=np.float32)).cuda() - campos
        viewbias = torch.from_numpy(np.array([23, 0, 0, 0], dtype=np.float32)).view(1,4,1).expand([self.batch_size, -1, -1]).cuda()

        anchor3dl = torch.inverse(projM) @ torch.from_numpy(np.array([xl * d, y * d, d, 1])).view(1,4,1).expand(self.batch_size,-1,-1).float().cuda()
        camtraill = anchor3dl.unsqueeze(1).expand([-1, self.sampleNum, -1, -1]) * lw + campos.unsqueeze(1).expand([-1, self.sampleNum, -1, -1]) * (1 - lw)

        anchor3dr = torch.inverse(projM) @ torch.from_numpy(np.array([xr * d, y * d, d, 1])).view(1,4,1).expand(self.batch_size,-1,-1).float().cuda()
        camtrailr = anchor3dr.unsqueeze(1).expand([-1, self.sampleNum, -1, -1]) * lw + campos.unsqueeze(1).expand([-1, self.sampleNum, -1, -1]) * (1 - lw)

        camtrail = torch.cat([camtraill, camtrailr], dim=1)
        return camtrail, campos, viewbias, camdir


    def cvtCamtrail2Extrsic(self, camtrail, campos, viewbias, camdir, extrinsic):
        diffvec = (campos + viewbias).unsqueeze(1).expand(self.batch_size, self.sampleNum * 2, 4, -1) - camtrail
        diffvec = diffvec / torch.norm(diffvec, dim=[2, 3], keepdim=True)

        diffvece = diffvec[:,:,0:3,0]
        camdire = camdir.unsqueeze(1).expand(-1, self.sampleNum * 2, -1, -1)[:,:,0:3,0]

        v = torch.cross(camdire, diffvece, dim=2)
        s = torch.norm(v, dim=2, keepdim=True)
        c = torch.sum(camdire * diffvece, dim=2, keepdim=True)

        V = torch.zeros([self.batch_size, self.sampleNum * 2, 3, 3], dtype=torch.float32, device=torch.device("cuda"))
        V[:, :, 0, 1] = -v[:, :, 2]
        V[:, :, 0, 2] = v[:, :, 1]
        V[:, :, 1, 0] = v[:, :, 2]
        V[:, :, 1, 2] = -v[:, :, 0]
        V[:, :, 2, 0] = -v[:, :, 1]
        V[:, :, 2, 1] = v[:, :, 0]

        ce = c.unsqueeze(3).expand(-1,-1,3,3)
        R = torch.eye(3).view(1, 1, 3, 3).expand([self.batch_size, self.sampleNum * 2, -1, -1]).cuda() + V + V @ V * (1 / (1 + ce))


        r_ex = torch.transpose(R @ torch.transpose(extrinsic[:,0:3,0:3], 1, 2).unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1]), 2, 3)
        t_ex = - r_ex @ camtrail[:,:,0:3,:]
        nextrinsic = torch.cat([r_ex, t_ex], dim=3)
        addr = torch.from_numpy(np.array([[[0,0,0,1]]], dtype=np.float32)).unsqueeze(2).expand(self.batch_size, self.sampleNum * 2, -1, -1).cuda()
        nextrinsic = torch.cat([nextrinsic, addr], dim=2)

        return nextrinsic

    def check_nextrinsic(self, campos, viewbias, intrinsic, nextrinsic):
        vpts = (campos + viewbias).unsqueeze(1).expand(self.batch_size, self.sampleNum * 2, 4, -1)
        nPs = intrinsic.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1]) @ nextrinsic
        pvpts = nPs @ vpts
        pvpts[:, :, 0, :] = pvpts[:, :, 0, :] / pvpts[:, :, 2, :]
        pvpts[:, :, 1, :] = pvpts[:, :, 1, :] / pvpts[:, :, 2, :]
        pvpts = pvpts[:,:,0:2,:]
        return pvpts

    def vsl_camtrail(self, pts3d, camtrail, campos, viewbias, camdir):
        draw_index = 0
        import matlab
        import matlab.engine
        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
        draw_x_pred = matlab.double(pts3d[draw_index, 0, :].view(-1).detach()[::self.dsrate].cpu().numpy().tolist())
        draw_y_pred = matlab.double(pts3d[draw_index, 1, :].view(-1).detach()[::self.dsrate].cpu().numpy().tolist())
        draw_z_pred = matlab.double(pts3d[draw_index, 2, :].view(-1).detach()[::self.dsrate].cpu().numpy().tolist())

        draw_x_camtral = matlab.double(camtrail[draw_index, :, 0, 0].detach().cpu().numpy().tolist())
        draw_y_camtral = matlab.double(camtrail[draw_index, :, 1, 0].detach().cpu().numpy().tolist())
        draw_z_camtral = matlab.double(camtrail[draw_index, :, 2, 0].detach().cpu().numpy().tolist())

        draw_campos = matlab.double(campos[draw_index, :, :].tolist())
        draw_view = matlab.double((campos + viewbias)[draw_index, :, :].tolist())
        draw_camdir = matlab.double(10 * camdir[draw_index, :, :].tolist())
        viewvec = (campos + viewbias).unsqueeze(1).expand(self.batch_size, self.sampleNum * 2, 4, -1) - camtrail
        viewvec = viewvec / torch.norm(viewvec, dim=[2,3], keepdim=True)
        draw_viewvec = matlab.double(viewvec[draw_index, :, :, 0].t().cpu().numpy().tolist())

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
        self.eng.eval('hold on', nargout=0)
        self.eng.scatter3(draw_x_camtral, draw_y_camtral, draw_z_camtral, 10, 'r', 'filled', nargout=0)
        self.eng.scatter3(draw_campos[0], draw_campos[1], draw_campos[2], 10, 'k', 'filled', nargout=0)
        self.eng.scatter3(draw_view[0], draw_view[1], draw_view[2], 10, 'r', 'filled', nargout=0)
        self.eng.quiver3(draw_x_camtral, draw_y_camtral, draw_z_camtral, draw_viewvec[0], draw_viewvec[1], draw_viewvec[2], nargout=0)
        self.eng.quiver3(draw_campos[0], draw_campos[1], draw_campos[2], draw_camdir[0], draw_camdir[1], draw_camdir[2], nargout=0)

    def proj2de(self, pts3d, intrinsic, nextrinsic, addmask = None):
        intrinsice = intrinsic.unsqueeze(1).expand(-1, self.sampleNum * 2, -1, -1)
        camKs = intrinsice @ nextrinsic

        camKs_e = camKs.view(self.batch_size, self.sampleNum * 2, 1, 1, 4, 4).expand(-1, -1, self.height, self.width, -1, -1)
        pts3d_e = pts3d.permute([0,2,3,1]).unsqueeze(4).unsqueeze(1).expand([-1,self.sampleNum * 2, -1, -1, -1, -1])
        projected3d = torch.matmul(camKs_e, pts3d_e).squeeze(5).permute(0, 1, 4, 2, 3)

        projecteddepth = projected3d[:, :, 2, :, :] + self.eps
        projected2d = torch.stack([projected3d[:, :, 0, :, :] / projecteddepth, projected3d[:, :, 1, :, :] / projecteddepth], dim=2)
        selector = (projected2d[:, :, 0, :, :] > 0) * (projected2d[:, :, 0, :, :] < self.width- 1) * (projected2d[:, :, 1, :, :] > 0) * (
                projected2d[:, :, 1, :, :] < self.height - 1) * (projecteddepth > 0)
        selector = selector.unsqueeze(2)
        projecteddepth = projecteddepth.unsqueeze(2)

        if addmask is not None:
            selector = selector * addmask.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1 , -1])
        return projected2d, projecteddepth, selector

    def visualize2d_e(self, projected2d, projecteddepth, pvpts = None, selector = None):
        draw_index = 0
        import matlab
        import matlab.engine
        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
        for i in range(self.sampleNum * 2):
            renderflag = True
            if selector is None:
                draw_x_pred = matlab.double(projected2d[draw_index, i, 0, :].view(-1).detach()[::self.dsrate].cpu().numpy().tolist())
                draw_y_pred = matlab.double(projected2d[draw_index, i, 1, :].view(-1).detach()[::self.dsrate].cpu().numpy().tolist())
                draw_z_pred = matlab.double((1.0 / (projecteddepth[draw_index, i, 0, :].view(-1).detach()[::self.dsrate].cpu().numpy())).tolist())
            else:
                if torch.sum(selector[draw_index, i, 0]) > 2:
                    draw_x_pred = matlab.double(projected2d[draw_index, i, 0, selector[draw_index, i, 0]].view(-1).detach().cpu().numpy().tolist())
                    draw_y_pred = matlab.double(projected2d[draw_index, i, 1, selector[draw_index, i, 0]].view(-1).detach().cpu().numpy().tolist())
                    draw_z_pred = matlab.double((1.0 / (projecteddepth[draw_index, i, 0, selector[draw_index, i, 0]].view(-1).detach().cpu().numpy())).tolist())
                else:
                    renderflag = False
            self.eng.eval('subplot(5,4,' + str(i + 1) + ')', nargout=0)
            if renderflag:
                self.eng.scatter(draw_x_pred, draw_y_pred, 1, draw_z_pred, nargout=0)
            self.eng.eval('axis equal', nargout = 0)
            self.eng.eval('xlim([{} {}])'.format(0, self.width), nargout=0)
            self.eng.eval('ylim([{} {}])'.format(0, self.height), nargout=0)
            self.eng.eval('axis ij'.format(0, self.height), nargout=0)
            if pvpts is not None:
                draw_pvpts = matlab.double(pvpts[draw_index, i, :, :].detach().cpu().numpy().tolist())
                self.eng.eval('hold on', nargout=0)
                self.eng.scatter(draw_pvpts[0], draw_pvpts[1], 1, 'r', nargout=0)


    def erpipolar_rendering(self, depthmap, semanticmap, intrinsic, extrinsic):
        # Compute Mask
        addmask = self.post_mask(depthmap = depthmap, semanticmap = semanticmap)

        invcamK = torch.inverse(intrinsic @ extrinsic)
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)

        # Generate Camera Track
        camtrail, campos, viewbias, camdir = self.get_camtrail(extrinsic, intrinsic)
        # self.vsl_camtrail(pts3d, camtrail, campos, viewbias, camdir)
        nextrinsic = self.cvtCamtrail2Extrsic(camtrail, campos, viewbias, camdir, extrinsic)
        projected2d, projecteddepth, selector = self.proj2de(pts3d=pts3d, intrinsic=intrinsic, nextrinsic=nextrinsic, addmask = addmask)

        # pvpts = self.check_nextrinsic(campos, viewbias, intrinsic, nextrinsic)
        # self.visualize2d_e(projected2d, projecteddepth, pvpts)

        epipoLine, Pcombined = self.get_eppl(intrinsic=intrinsic, extrinsic = extrinsic, nextrinsic=nextrinsic)
        r_sigma, inv_r_sigma, rotM = self.eppl2CovM(epipoLine)
        # self.vslGauss(projected2d, depthmap, epipoLine, intrinsic, extrinsic, nextrinsic, r_sigma, inv_r_sigma)

        rimg, grad2d, _, depthmapnp_grad = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap.cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

        # self.show_rendered_eppl(rimg)
        # depthmapnp_grad = eppl_pix2dgrad2depth(grad2d = grad2d, Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap.cpu().numpy(), bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
        return rimg, addmask

    def gradcheckGeneral(self, grad2d, depthmapnp_grad, Pcombined, depthmap, invcamK, intrinsic, nextrinsic, addmask, selector, projected2d, inv_r_sigma):
        ratio = 1
        grad2d = grad2d
        Pcombinednp = Pcombined.cpu().numpy()
        depthmapnp = depthmap.cpu().numpy()
        bs = self.batch_size
        samplesz = self.sampleNum * 2
        height = self.height
        width = self.width

        mask = selector.detach().cpu().numpy()
        testid = 45156
        poses = np.argwhere(mask[:, :, 0, :, :] == 1)
        c = poses[testid, 0]
        sz = poses[testid, 1]
        yy = poses[testid, 2]
        xx = poses[testid, 3]
        for tt in range(2):
            delta = 1e-3 * ratio
            depthvalPlus = copy.deepcopy(depthmap)
            depthvalPlus[c, 0, yy, xx] = depthvalPlus[c, 0, yy, xx] + delta
            depthvalMinus = copy.deepcopy(depthmap)
            depthvalMinus[c, 0, yy, xx] = depthvalMinus[c, 0, yy, xx] - delta

            pts3dPlus = self.bck(predDepth=depthvalPlus, invcamK=invcamK)
            pts3dMinus = self.bck(predDepth=depthvalMinus, invcamK=invcamK)

            projected2dPlus, projecteddepthPlus, selectorPlus = self.proj2de(pts3d=pts3dPlus, intrinsic=intrinsic, nextrinsic=nextrinsic,
                                                                 addmask=addmask)
            projected2dMinus, projecteddepthMinus, selectorMinus = self.proj2de(pts3d=pts3dMinus, intrinsic=intrinsic, nextrinsic=nextrinsic,
                                                                 addmask=addmask)

            rimgPlus, grad2d, _, _ = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2dPlus.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap.cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
            rimgMinus, grad2d, _, _ = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2dMinus.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap.cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

            pts2d = projected2d.permute([0, 1, 3, 4, 2]).detach().cpu().numpy()
            sr = self.sr
            srhalf = int((sr - 1) / 2)
            s1 = 0
            s2 = 0
            for sz in range(self.sampleNum * 2):
                ctx = int(np.round_(pts2d[c, sz, yy, xx, 0]))
                cty = int(np.round_(pts2d[c, sz, yy, xx, 1]))
                for i in range(ctx - srhalf, ctx + srhalf + 1):
                    for j in range(cty - srhalf, cty + srhalf + 1):
                        if i >= 0 and i < width and j >= 0 and j < height:
                            s1 = s1 + rimgPlus[c, sz, j, i]
                            s2 = s2 + rimgMinus[c, sz, j, i]
            # depthmapnp_grad[c, 0, yy, xx] / ((s1 - s2) / 2 /delta)
            # ((s1 - s2) / 2 / delta)
            if tt == 0:
                ratio = 1e-3 / np.abs((s1 - s2)) * ratio
            else:
                print(depthmapnp_grad[c, 0, yy, xx] / ((s1 - s2) / 2 /delta))


    def erpipolar_rendering_test(self, depthmap, semanticmap, intrinsic, extrinsic):
        # Compute Mask
        addmask = self.post_mask(depthmap = depthmap, semanticmap = semanticmap)
        # Generate Camera Track
        camtrail, campos, viewbias, camdir = self.get_camtrail(extrinsic, intrinsic)
        nextrinsic = self.cvtCamtrail2Extrsic(camtrail, campos, viewbias, camdir, extrinsic)
        invcamK = torch.inverse(intrinsic @ extrinsic)
        epipoLine, Pcombined = self.get_eppl(intrinsic=intrinsic, extrinsic=extrinsic, nextrinsic=nextrinsic)
        r_sigma, inv_r_sigma, rotM = self.eppl2CovM(epipoLine)


        # Compute GT Mask
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)
        projected2d, projecteddepth, selector = self.proj2de(pts3d=pts3d, intrinsic=intrinsic, nextrinsic=nextrinsic, addmask = addmask)
        rimg_gt, grad2d, _ = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector.detach().cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

        depthmap_ns = depthmap + torch.randn(depthmap.shape, device=torch.device("cuda")) * 1e-2
        pts3d_ns = self.bck(predDepth=depthmap_ns, invcamK=invcamK)
        projected2d_ns, projecteddepth_ns, selector_ns = self.proj2de(pts3d=pts3d_ns, intrinsic=intrinsic, nextrinsic=nextrinsic, addmask = addmask)
        rimg_ns, grad2d_ns, _ = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d_ns.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector_ns.detach().cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

        # self.show_rendered_eppl(rimg)
        # depthmapnp_grad = eppl_pix2dgrad2depth_l2(grad2d = grad2d_ns, Pcombinednp = Pcombined.cpu().numpy(), rimg_gt = rimg_gt, rimg_ns = rimg_ns, depthmapnp = depthmap_ns.cpu().numpy(), bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
        depthmapnp_grad = eppl_pix2dgrad2depth(grad2d=grad2d_ns, Pcombinednp=Pcombined.cpu().numpy(),
                                               depthmapnp=depthmap_ns.cpu().numpy(), bs=self.batch_size,
                                               samplesz=self.sampleNum * 2, height=self.height, width=self.width)

        return


    def gradcheckl2(self, grad2d, depthmapnp_grad, Pcombined, depthmap_ns, invcamK, intrinsic, nextrinsic, addmask, selector, projected2d, inv_r_sigma, rimg_gt):
        ratio = 1
        height = self.height
        width = self.width

        mask = selector_ns.detach().cpu().numpy()
        testid = 100
        poses = np.argwhere(mask[:, :, 0, :, :] == 1)
        c = poses[testid, 0]
        sz = poses[testid, 1]
        yy = poses[testid, 2]
        xx = poses[testid, 3]
        for tt in range(2):
            delta = 1e-4 * ratio
            depthvalPlus = copy.deepcopy(depthmap_ns)
            depthvalPlus[c, 0, yy, xx] = depthvalPlus[c, 0, yy, xx] + delta
            depthvalMinus = copy.deepcopy(depthmap_ns)
            depthvalMinus[c, 0, yy, xx] = depthvalMinus[c, 0, yy, xx] - delta

            pts3dPlus = self.bck(predDepth=depthvalPlus, invcamK=invcamK)
            pts3dMinus = self.bck(predDepth=depthvalMinus, invcamK=invcamK)

            projected2dPlus, projecteddepthPlus, selectorPlus = self.proj2de(pts3d=pts3dPlus, intrinsic=intrinsic, nextrinsic=nextrinsic,
                                                                 addmask=addmask)
            projected2dMinus, projecteddepthMinus, selectorMinus = self.proj2de(pts3d=pts3dMinus, intrinsic=intrinsic, nextrinsic=nextrinsic,
                                                                 addmask=addmask)

            rimgPlus, grad2d, _ = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2dPlus.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selectorPlus.detach().cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
            rimgMinus, grad2d, _ = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2dMinus.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selectorMinus.detach().cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

            pts2d = projected2d.permute([0, 1, 3, 4, 2]).detach().cpu().numpy()
            sr = self.sr
            srhalf = int((sr - 1) / 2)
            s1 = 0
            s2 = 0
            s3 = 0
            s4 = 0
            for sz in range(self.sampleNum * 2):
                ctx = int(np.round_(pts2d[c, sz, yy, xx, 0]))
                cty = int(np.round_(pts2d[c, sz, yy, xx, 1]))
                for i in range(ctx - srhalf, ctx + srhalf + 1):
                    for j in range(cty - srhalf, cty + srhalf + 1):
                        if i >= 0 and i < width and j >= 0 and j < height:
                            s3 = s3 + (rimgPlus[c, sz, j, i] - rimg_gt[c, sz, j, i]) ** 2
                            s4 = s4 + (rimgMinus[c, sz, j, i] - rimg_gt[c, sz, j, i]) ** 2
                            s1 = s1 + rimgPlus[c, sz, j, i]
                            s2 = s2 + rimgMinus[c, sz, j, i]
            # depthmapnp_grad[c, 0, yy, xx] / ((s1 - s2) / 2 /delta)
            # ((s1 - s2) / 2 / delta)
            if tt == 0:
                ratio = 1e-4 / np.abs((s1 - s2))
            else:
                print(depthmapnp_grad[c, 0, yy, xx] / ((s1 - s2) / 2 /delta))

    def get_reprojected_pts2d(self, Pcombinednp, depthval, c, xx, yy, samplesz):
        pts25d = np.stack([np.ones(samplesz) * xx, np.ones(samplesz) * yy, np.ones(samplesz) * depthval, np.ones(samplesz)], axis = 1)
        pts25d = np.expand_dims(pts25d, axis = 2)
        P = Pcombinednp[c, :, :, :]
        projected = P @ pts25d
        projected2d = projected[:,0:2,:]
        projected2d[:, 0, :] = projected2d[:, 0, :] / projected[:, 2, :]
        projected2d[:, 1, :] = projected2d[:, 1, :] / projected[:, 2, :]

        return projected2d

    def show_rendered_eppl(self, rimg):
        vmax = rimg.max() * 0.3
        rimg_t = torch.from_numpy(rimg).unsqueeze(2)
        bz = 0
        imgt = list()
        for i in range(10):
            imgt.append(np.array(tensor2disp(rimg_t[bz], vmax=vmax, ind=i)))
        pil.fromarray(np.concatenate(imgt, axis=0)).show()

    def get_eppl(self, intrinsic, extrinsic, nextrinsic):
        intrinsic_44, added_extrinsic = self.org_intrinsic(intrinsic)
        intrinsic_44e = intrinsic_44.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1])
        added_extrinsice = added_extrinsic.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1])
        extrinsice = extrinsic.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1])
        extrinsic_old = added_extrinsice @ extrinsice
        extrinsic_new = added_extrinsice @ nextrinsic

        Pold = intrinsic_44e @ extrinsic_old
        Pnew = intrinsic_44e @ extrinsic_new
        Cold = torch.inverse(Pold) @ torch.tensor([0, 0, 0, 1]).view(1, 1, 4, 1).expand([self.batch_size, self.sampleNum * 2, -1, -1]).float().cuda()
        Pcombined = Pnew @ torch.inverse(Pold)

        rand_dmap = torch.rand([self.batch_size, 1, 1, self.height, self.width]).expand([-1, self.sampleNum * 2, -1, -1, -1]).cuda() + 2
        xxe = self.xx.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1, -1])
        yye = self.yy.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1, -1])
        onese = self.ones.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1, -1])
        randx = torch.cat([xxe * rand_dmap, yye * rand_dmap, rand_dmap, onese], dim=2)

        Cold_new = Pnew @ Cold
        Cold_new = Cold_new[:, :, 0:3, :]
        Cold_new[:, :, 0, :] = Cold_new[:, :, 0, :] / Cold_new[:, :, 2, :]
        Cold_new[:, :, 1, :] = Cold_new[:, :, 1, :] / Cold_new[:, :, 2, :]
        Cold_new[:, :, 2, :] = Cold_new[:, :, 2, :] / Cold_new[:, :, 2, :]
        Cold_newn = Cold_new / torch.norm(Cold_new, dim = [2,3], keepdim=True).expand([-1,-1,3,-1])
        Cold_newne = Cold_newn.view(self.batch_size, self.sampleNum * 2, 1, 1, 3, 1).expand([-1, -1, self.height, self.width, -1, -1])

        tmpM = Pnew @ torch.inverse(Pold)
        tmpM = tmpM.view(self.batch_size, self.sampleNum * 2, 1, 1, 4, 4).expand(-1, -1, self.height, self.width, -1, -1)

        randx_e = randx.permute([0,1,3,4,2]).unsqueeze(5)
        randx_new = torch.matmul(tmpM, randx_e)

        randx_new = randx_new[:,:,:,:,0:3,:]
        randx_new[:, :, :, :, 0, :] = randx_new[:, :, :, :, 0, :] / randx_new[:, :, :, :, 2, :]
        randx_new[:, :, :, :, 1, :] = randx_new[:, :, :, :, 1, :] / randx_new[:, :, :, :, 2, :]
        randx_new[:, :, :, :, 2, :] = randx_new[:, :, :, :, 2, :] / randx_new[:, :, :, :, 2, :]
        randx_newn = randx_new / (torch.norm(randx_new, dim = [4,5], keepdim=True).expand([-1,-1,-1,-1,3,-1]) + self.eps)

        epipoLine = torch.cross(Cold_newne, randx_newn, dim = 4)
        epipoLine = epipoLine.squeeze(5).permute([0,1,4,2,3])

        # self.rnd_ck_epp(Pold, Pnew, epipoLine)
        return epipoLine, Pcombined

    def org_intrinsic(self, intrinsic):
        intrinsic_44 = copy.deepcopy(intrinsic)
        intrinsic_44[:, 0:3, 3] = 0
        added_extrinsic = torch.inverse(intrinsic_44) @ intrinsic
        return intrinsic_44, added_extrinsic

    def eppl2CovM(self, epipoLine):
        # Turn Epipolar Line to Covarian Matrix
        ln = torch.sqrt(epipoLine[:,:,0,:,:].pow(2) + epipoLine[:,:,1,:,:].pow(2))
        ldeg = torch.acos(epipoLine[:,:,1,:,:] / ln)

        rotM = torch.stack([torch.stack([torch.cos(ldeg), torch.sin(ldeg)], dim=4), torch.stack([-torch.sin(ldeg), torch.cos(ldeg)], dim=4)], dim=5)
        r_sigma = rotM @ self.sigma.unsqueeze(1).expand([-1,self.sampleNum*2,-1,-1,-1,-1]) @ rotM.transpose(dim0=4, dim1=5)
        r_sigma = r_sigma / (torch.norm(r_sigma, dim = [4, 5], keepdim = True) + self.eps)

        determinant = r_sigma[:,:,:,:,0,0] * r_sigma[:,:,:,:,1,1] - r_sigma[:,:,:,:,0,1] * r_sigma[:,:,:,:,1,0]
        determinant = determinant.unsqueeze(4).unsqueeze(5)
        inv_r_sigma = torch.stack([torch.stack([r_sigma[:,:,:,:,1,1], -r_sigma[:,:,:,:,1,0]], dim=4), torch.stack([-r_sigma[:,:,:,:,0,1], r_sigma[:,:,:,:,0,0]], dim=4)], dim=5) / determinant
        return r_sigma, inv_r_sigma, rotM

    def rnd_ck_epp(self, Pold, Pnew, epipoLine):
        rand_dmap = torch.rand([self.batch_size, self.sampleNum * 2, 1, self.height, self.width]).float().cuda() + 10
        xxe = self.xx.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1, -1])
        yye = self.yy.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1, -1])
        onese = self.ones.unsqueeze(1).expand([-1, self.sampleNum * 2, -1, -1, -1])

        randx = torch.cat([xxe * rand_dmap, yye * rand_dmap, rand_dmap, onese], dim=2)
        tmpM = Pnew @ torch.inverse(Pold)
        tmpM = tmpM.view(self.batch_size, self.sampleNum * 2, 1, 1, 4, 4).expand(-1, -1, self.height, self.width, -1, -1)
        randx_e = randx.permute([0,1,3,4,2]).unsqueeze(5)
        randx_new = torch.matmul(tmpM, randx_e)

        randx_new = randx_new[:,:,:,:,0:3,:]
        randx_new[:, :, :, :, 0, :] = randx_new[:, :, :, :, 0, :] / randx_new[:, :, :, :, 2, :]
        randx_new[:, :, :, :, 1, :] = randx_new[:, :, :, :, 1, :] / randx_new[:, :, :, :, 2, :]
        randx_new[:, :, :, :, 2, :] = randx_new[:, :, :, :, 2, :] / randx_new[:, :, :, :, 2, :]
        randx_new = randx_new.squeeze(5).permute([0,1,4,2,3])

        ck = torch.sum(randx_new * epipoLine, dim=[2])
        if torch.mean(ck) < 1:
            return True
        else:
            return False

    def vslGauss(self, projected2d, depthmap, epipoLine, intrinsic, extrinsic, nextrinsic, r_sigma, inv_r_sigma, rotM):
        # Assign test index
        bz = 0
        c = 0
        h = 20
        w = 100
        mu = projected2d[bz, c, :, h, w].detach().cpu().numpy()

        bsnum = 100
        nsdpeth = depthmap[bz, 0, h, w].detach().cpu().numpy() * np.ones(bsnum) + np.random.normal(0, 0.2, bsnum)
        nsdpeth = nsdpeth * self.kws
        nspts = np.stack([np.ones(bsnum) * w * nsdpeth, np.ones(bsnum) * h * nsdpeth, nsdpeth, np.ones(bsnum)], axis=0)

        intrinsicnp = intrinsic[bz].cpu().numpy()
        extrinsicnp = extrinsic[bz].cpu().numpy()
        nextrinsicnp = nextrinsic[bz, c].cpu().numpy()
        pnspts = intrinsicnp @ nextrinsicnp @ np.linalg.inv(intrinsicnp @ extrinsicnp) @ nspts
        pnspts[0, :] = pnspts[0, :] / pnspts[2, :]
        pnspts[1, :] = pnspts[1, :] / pnspts[2, :]
        pnspts = pnspts[0:2, :]
        # pnspts = (pnspts - np.expand_dims(mu, axis=1)) /  + np.expand_dims(mu, axis=1)


        # epipoLinenp = epipoLine[bz, c, :, h, w].cpu().numpy()
        # eppy = (-pnspts[0, :] * epipoLinenp[0] - epipoLinenp[2]) / epipoLinenp[1]


        sigmaM = r_sigma[bz, c, h, w, :, :].detach().cpu().numpy()
        inv_sigmaM = inv_r_sigma[bz, c, h, w, :, :].detach().cpu().numpy()
        sxx, syy = np.random.multivariate_normal(mu, sigmaM, 5000).T
        sxx = (sxx - mu[0]) * self.kws + mu[0]
        syy = (syy - mu[1]) * self.kws + mu[1]

        spnum = 100
        x = np.linspace(-3 * self.kws, 3 * self.kws, spnum) + mu[0]
        y = np.linspace(-3 * self.kws, 3 * self.kws, spnum) + mu[1]

        xx, yy = np.meshgrid(x, y, indexing='xy')
        xx = np.fliplr(xx)
        cx = (xx - mu[0]) / self.kws
        cy = (yy - mu[1]) / self.kws
        v = inv_sigmaM[0, 0] * cx * cx + inv_sigmaM[1, 0] * cx * cy + inv_sigmaM[0, 1] * cx * cy + inv_sigmaM[1, 1] * cy * cy
        v = np.abs(np.exp(-v / 2) / 2 / np.pi - 0.05)
        v = v.max() - v

        # Draw Approximation elipsoid
        rotMnp = rotM[bz, c, h, w, :, :].cpu().numpy()
        elsnum = 100
        sigmax = np.sqrt(self.sigma[0,0,0,0,0].cpu().numpy())
        sigmay = np.sqrt(self.sigma[0,0,0,1,1].cpu().numpy())
        s = np.sqrt(5.991)
        eltheta = np.linspace(0, 2 * np.pi, elsnum)
        eltheta = np.concatenate([eltheta, np.array([eltheta[0]])], axis=0)
        elx = np.cos(eltheta) * sigmax * s * self.kws
        ely = np.sin(eltheta) * sigmay * s * self.kws
        elxx = elx * rotMnp[0,0] + ely * rotMnp[0,1]
        elyy = elx * rotMnp[1,0] + ely * rotMnp[1,1]
        elxx = elxx + mu[0]
        elyy = elyy + mu[1]



        import matlab
        import matlab.engine
        if self.eng is None:
            self.eng = matlab.engine.start_matlab()
        drawsxx = matlab.double(sxx.tolist())
        drawsyy = matlab.double(syy.tolist())

        pnsptsxx = matlab.double(pnspts[0,:].tolist())
        pnsptsyy = matlab.double(pnspts[1,:].tolist())

        # draweppy = matlab.double(eppy.tolist())

        drawelx = matlab.double(elxx.tolist())
        drawely = matlab.double(elyy.tolist())
        self.eng.eval('figure()', nargout=0)
        self.eng.scatter(drawsxx, drawsyy, 5, 'g', 'filled', nargout=0)
        self.eng.eval('axis equal', nargout = 0)
        self.eng.eval('hold on', nargout=0)
        # self.eng.scatter(pnsptsxx, draweppy, 5, 'k', 'filled', nargout=0)
        self.eng.scatter(pnsptsxx, pnsptsyy, 5, 'k', 'filled', nargout=0)
        self.eng.plot(drawelx, drawely, 'Color', 'c', nargout=0)
        xlim = matlab.double([mu[0] - 3 * self.kws, mu[0] + 3 * self.kws])
        ylim = matlab.double([mu[1] - 3 * self.kws, mu[1] + 3 * self.kws])
        self.eng.xlim(xlim, nargout=0)
        self.eng.ylim(ylim, nargout=0)

        drawxx = matlab.double(xx.flatten().tolist())
        drawyy = matlab.double(yy.flatten().tolist())
        drawv = matlab.double(v.flatten().tolist())
        self.eng.eval('figure()', nargout=0)
        self.eng.scatter(drawxx, drawyy, 5, drawv, 'filled', nargout=0)
        self.eng.eval('axis equal', nargout = 0)
        xlim = matlab.double([mu[0] - 3 * self.kws, mu[0] + 3 * self.kws])
        ylim = matlab.double([mu[1] - 3 * self.kws, mu[1] + 3 * self.kws])
        self.eng.xlim(xlim, nargout=0)
        self.eng.ylim(ylim, nargout=0)


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

    def print(self, depthmap, semanticmap, intrinsic, extrinsic, ppath):
        self.dsrate = 1
        addmask = self.post_mask(depthmap = depthmap, semanticmap = semanticmap)

        invcamK = torch.inverse(intrinsic @ extrinsic)
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)
        projected2d, projecteddepth, selector = self.proj2d(pts3d=pts3d, intrinsic=intrinsic, extrinsic=extrinsic, R=self.R, T=self.T, addmask = addmask)

        # Print
        self.print2d(projected2d = projected2d, projecteddepth = projecteddepth, selector = selector, ppath = ppath)

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

    def debug_visualization(self, depthmap, semanticmap, intrinsic, extrinsic):
        invcamK = torch.inverse(intrinsic @ extrinsic)
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)

        self.o_trans = np.array([-3, 6, 5]) # x, y, z
        self.o_rot = np.array([-15, 15, 0])  # yaw, pitch, roll
        self.selfmv2aff()

        projected2d, projecteddepth, selector = self.proj2d(pts3d=pts3d, intrinsic=intrinsic, extrinsic=extrinsic, R=self.R, T=self.T)
        self.visualize3d(pts3d = pts3d, extrinsic = extrinsic, R = self.R, T = self.T)
        self.visualize2d(projected2d = projected2d, projecteddepth = projecteddepth)
        print("visible ratio % f" % (torch.sum(selector).float() / self.batch_size / self.width / self.height))

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

    def selfmv2aff(self):
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