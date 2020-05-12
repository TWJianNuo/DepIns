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
from Oview_Gan import eppl_render, eppl_render_l2, eppl_render_l1, eppl_render_l1_sfgrad

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

class localGeomDesp(nn.Module):
    def __init__(self, height, width, batch_size, ptspair):
        super(localGeomDesp, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.ptspair = ptspair

        # Init grid points
        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.xx = torch.from_numpy(xx).unsqueeze(0).unsqueeze(3).expand([self.batch_size, -1, -1, 1]).float()
        self.yy = torch.from_numpy(yy).unsqueeze(0).unsqueeze(3).expand([self.batch_size, -1, -1, 1]).float()
        self.pixelLocs = nn.Parameter(torch.cat([self.xx, self.yy, torch.ones_like(self.xx)], dim=3), requires_grad=False)

        self.interestedLocs = torch.cat([self.xx, self.yy, torch.ones_like(self.xx)], dim=3).clone().view(self.batch_size, 1, 1, self.height, self.width, 3).repeat([1, len(self.ptspair), 2, 1,1, 1])

        for i in range(len(self.ptspair)):
            for j in range(2):
                self.interestedLocs[:,i,j,:,:,:] = self.interestedLocs[:,i,j,:,:,:] + torch.from_numpy(np.array(self.ptspair[i][j])).float().view([1,1,1,3]).expand([self.batch_size, self.height, self.width, -1])

        self.interestedLocs = nn.Parameter(self.interestedLocs, requires_grad = False)

        w = 4
        weight = np.zeros([len(self.ptspair) * 2, 1, int(w * 2 + 1), int(w * 2 + 1)])
        for i in range(len(self.ptspair)):
            for j in range(2):
                weight[i * 2 + j, 0, -self.ptspair[i][j][1] + w, -self.ptspair[i][j][0] + w] = 1
        self.copyConv = torch.nn.Conv2d(len(self.ptspair) * 2, len(self.ptspair) * 2, int(w * 2 + 1), stride=1, padding=w, bias=False, groups = len(self.ptspair) * 2)
        self.copyConv.weight = torch.nn.Parameter(torch.from_numpy(weight.astype(np.float32)), requires_grad=False)
    def forward(self, predNorm, depthmap, invIn):
        predNorm_ = predNorm.view(self.batch_size, len(self.ptspair), 3, self.height, self.width).permute([0,1,3,4,2]).unsqueeze(4)
        intrinsic_c = invIn[:,0:3,0:3].view(self.batch_size,1,1,1,3,3)
        normIn = torch.matmul(predNorm_, intrinsic_c)

        p2d_ex = self.pixelLocs.unsqueeze(1).unsqueeze(5).expand([-1, len(self.ptspair), -1, -1, -1, -1])
        k = -torch.matmul(normIn, p2d_ex).squeeze(4).squeeze(4) * depthmap.expand([-1, len(self.ptspair), -1, -1])

        predDepth = normIn.unsqueeze(2).expand([-1,-1,2,-1,-1,-1,-1]) @ self.interestedLocs.unsqueeze(6)
        predDepth = predDepth.squeeze(5).squeeze(5)
        predDepth = -k.unsqueeze(2).expand([-1,-1,2,-1,-1]) / (predDepth)
        # ptspred3d = predDepth.unsqueeze(5).expand([-1,-1,-1,-1,-1,3]) * (intrinsic_c.unsqueeze(2).expand([-1, len(self.pixelLocs), 2, self.height, self.width, -1, -1]) @ self.interestedLocs.unsqueeze(6)).squeeze(6)
        # ck = torch.sum(ptspred3d * predNorm_.squeeze(4).unsqueeze(2).expand([-1,-1,2,-1,-1,-1]), axis = [5])
        # ck = ck + k.unsqueeze(2).expand([-1,-1,2,-1,-1])
        # torch.abs(ck).max()
        predDepth_ = predDepth.view(self.batch_size, len(self.ptspair) * 2, self.height, self.width)
        predDepth_ = self.copyConv(predDepth_)

        # import random
        # for i in range(100):
        #     bz = random.randint(0, self.batch_size - 1)
        #     chn = random.randint(0, 2 * len(self.ptspair) - 1)
        #     hn = random.randint(0, self.height - 1)
        #     wn = random.randint(0, self.width - 1)
        #
        #     dpval = predDepth_[bz, chn, hn, wn]
        #     sPts2d = torch.Tensor([wn, hn, 1]).float().cuda()
        #     sPts3d = invIn[bz,0:3,0:3] @ sPts2d.unsqueeze(1) * dpval
        #
        #     deltax = self.ptspair[int(chn / 2)][chn % 2][0]
        #     deltay = self.ptspair[int(chn / 2)][chn % 2][1]
        #     planeDir = predNorm_[bz, int(chn / 2), hn - deltay, wn - deltax, :, :]
        #     sk = k[bz, int(chn / 2), hn - deltay, wn - deltax]
        #
        #     val = (planeDir @ sPts3d)[0][0] + sk
        #
        #     print(val)





        # depthmap = torch.clamp(depthmap, min=0, max=100)
        # pts3d = invIn[:,0:3,0:3].view(self.batch_size,1,1,3,3).expand([-1, self.height, self.width, 3, 3]) @ self.pixelLocs.unsqueeze(4)
        # pts3d = pts3d * depthmap.squeeze(1).unsqueeze(3).unsqueeze(4).expand([-1, -1, -1, 3, 1])
        # pts3d = pts3d.squeeze(4).permute(0,3,1,2).contiguous()
        #
        # ck_p = torch.sum(predNorm.view(self.batch_size, len(self.ptspair), 3, self.height, self.width) * pts3d.unsqueeze(1).expand([-1,len(self.ptspair),-1,-1,-1]), axis = [2])
        # ck_p = ck_p + k
        # torch.abs(ck_p).max()
        #
        # tensor2disp((torch.abs(ck_p) > 100)[:,0:1,:,:], vmax = 1, ind = 0).show()
        # import matlab
        # import matlab.engine
        # eng = matlab.engine.start_matlab()
        #
        # sampleR = 5
        # viewIndex = 0
        # dx = pts3d[viewIndex, 0, :, :].cpu().detach().numpy().flatten()[::sampleR]
        # dy = pts3d[viewIndex, 1, :, :].cpu().detach().numpy().flatten()[::sampleR]
        # dz = pts3d[viewIndex, 2, :, :].cpu().detach().numpy().flatten()[::sampleR]
        #
        # dx = matlab.double(dx.tolist())
        # dy = matlab.double(dy.tolist())
        # dz = matlab.double(dz.tolist())
        #
        # eng.eval('close all', nargout=0)
        # eng.eval('figure()', nargout=0)
        # eng.eval('hold on', nargout=0)
        # eng.scatter3(dx, dy, dz, 5, 'filled', 'g', nargout=0)
        # eng.eval('axis equal', nargout=0)
        # eng.eval('grid off', nargout=0)
        # eng.eval('xlabel(\'X\')', nargout=0)
        # eng.eval('ylabel(\'Y\')', nargout=0)
        # eng.eval('zlabel(\'Z\')', nargout=0)
        # eng.eval('xlim([0 50])', nargout=0)
        # eng.eval('ylim([-40 40])', nargout=0)
        # eng.eval('zlim([-3 10])', nargout=0)
        return predDepth_


class LinGeomDesp(nn.Module):
    def __init__(self, height, width, batch_size, ptspair, invIn):
        super(LinGeomDesp, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.ptspair = ptspair

        # Init grid points
        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.xx = torch.from_numpy(xx).unsqueeze(0).unsqueeze(3).expand([self.batch_size, -1, -1, 1]).float()
        self.yy = torch.from_numpy(yy).unsqueeze(0).unsqueeze(3).expand([self.batch_size, -1, -1, 1]).float()
        self.pixelLocs = nn.Parameter(torch.cat([self.xx, self.yy, torch.ones_like(self.xx)], dim=3), requires_grad=False)

        self.interestedLocs = torch.cat([self.xx, self.yy, torch.ones_like(self.xx)], dim=3).clone().view(self.batch_size, 1, 1, self.height, self.width, 3).repeat([1, len(self.ptspair), 3, 1,1, 1])

        for i in range(len(self.ptspair)):
            for j in range(3):
                if j != 0:
                    self.interestedLocs[:,i,j,:,:,:] = self.interestedLocs[:,i,j,:,:,:] + torch.from_numpy(np.array(self.ptspair[i][j-1])).float().view([1,1,1,3]).expand([self.batch_size, self.height, self.width, -1])

        # x_dir = list()
        # for i in range(len(self.ptspair)):
        #     tmp = np.array(self.ptspair[i][1]) - np.array(self.ptspair[i][0])
        #     tmp = tmp / np.sqrt(np.sum(tmp ** 2))
        #     x_dir.append(tmp)
        # x_dir = np.stack(x_dir, axis=0)
        # x_dir = torch.from_numpy(x_dir).float().view(1, len(self.ptspair), 1, 1, 3).expand(
        #     [self.batch_size, -1, self.height, self.width, -1])
        # # Compute vertical direction
        # vert_dir_tmp = torch.from_numpy(invIn).view(1,1,1,1,1,3,3).expand([self.batch_size, len(self.ptspair), 3, self.height, self.width, -1, -1]).float() @ self.interestedLocs.unsqueeze(6)
        # vert_dir = torch.cross(vert_dir_tmp[:,:,1,:,:,:,0], vert_dir_tmp[:,:,2,:,:,:,0], dim=4)
        # z_dir = vert_dir / torch.norm(vert_dir, keepdim = True, dim = 4).expand([-1,-1,-1,-1,3])
        # y_dir = torch.cross(z_dir, x_dir, dim = 4)

        y_dir = list()
        for i in range(len(self.ptspair)):
            tmp = np.array(self.ptspair[i][1]) - np.array(self.ptspair[i][0])
            tmp = tmp / np.sqrt(np.sum(tmp ** 2))
            y_dir.append(tmp)
        y_dir = np.stack(y_dir, axis=0)
        y_dir = torch.from_numpy(y_dir).float().view(1, len(self.ptspair), 1, 1, 3).expand(
            [self.batch_size, -1, self.height, self.width, -1])
        # Compute vertical direction
        vert_dir_tmp = torch.from_numpy(invIn).view(1,1,1,1,1,3,3).expand([self.batch_size, len(self.ptspair), 3, self.height, self.width, -1, -1]).float() @ self.interestedLocs.unsqueeze(6)
        vert_dir = torch.cross(vert_dir_tmp[:,:,1,:,:,:,0], vert_dir_tmp[:,:,2,:,:,:,0], dim=4)
        z_dir = vert_dir / torch.norm(vert_dir, keepdim = True, dim = 4).expand([-1,-1,-1,-1,3])
        x_dir = torch.cross(z_dir, y_dir, dim = 4)

        self.x_dir = torch.nn.Parameter(x_dir, requires_grad=False)
        self.z_dir = torch.nn.Parameter(z_dir, requires_grad=False)
        self.y_dir = torch.nn.Parameter(y_dir, requires_grad=False)
        self.invIn = torch.nn.Parameter(torch.from_numpy(invIn).float(), requires_grad=False)

        # self.interestedLocs = nn.Parameter(self.interestedLocs, requires_grad = False)

        w = 4
        weightl = np.zeros([len(self.ptspair), 1, int(w * 2 + 1), int(w * 2 + 1)])
        for i in range(len(self.ptspair)):
            weightl[i, 0, self.ptspair[i][0][1] + w, self.ptspair[i][0][0] + w] = 1
        self.copyConv_thetal = torch.nn.Conv2d(len(self.ptspair), len(self.ptspair), int(w * 2 + 1), stride=1, padding=w, bias=False, groups=len(self.ptspair))
        self.copyConv_thetal.weight = torch.nn.Parameter(torch.from_numpy(weightl.astype(np.float32)), requires_grad=False)

        weightr = np.zeros([len(self.ptspair), 1, int(w * 2 + 1), int(w * 2 + 1)])
        for i in range(len(self.ptspair)):
            weightr[i, 0, self.ptspair[i][1][1] + w, self.ptspair[i][1][0] + w] = 1
        self.copyConv_thetar = torch.nn.Conv2d(len(self.ptspair), len(self.ptspair), int(w * 2 + 1), stride=1, padding=w, bias=False, groups=len(self.ptspair))
        self.copyConv_thetar.weight = torch.nn.Parameter(torch.from_numpy(weightr.astype(np.float32)), requires_grad=False)
    def get_theta(self, depthmap):
        # depthmap = torch.clamp(depthmap, min=0.1, max = 80)

        invIn_ex = self.invIn.view(1,1,1,3,3).expand([self.batch_size, self.height, self.width, -1, -1])
        pts3d = (invIn_ex @ self.pixelLocs.unsqueeze(4)) * depthmap.squeeze(1).unsqueeze(3).unsqueeze(4).expand([-1,-1,-1,3,-1])
        pts3d_ex = pts3d.unsqueeze(1).squeeze(5).expand([-1,len(self.ptspair),-1,-1,-1])
        pts3d_nx = torch.sum(pts3d_ex * self.x_dir, dim=[4])
        pts3d_ny = torch.sum(pts3d_ex * self.y_dir,dim=[4])
        pts3d_nz = torch.sum(pts3d_ex * self.z_dir,dim=[4])



        pts3d_nxl = self.copyConv_thetal(pts3d_nx)
        pts3d_nyl = self.copyConv_thetal(pts3d_ny)
        pts3d_nzl = self.copyConv_thetal(pts3d_nz)

        pts3d_nxr = self.copyConv_thetar(pts3d_nx)
        pts3d_nyr = self.copyConv_thetar(pts3d_ny)
        pts3d_nzr = self.copyConv_thetar(pts3d_nz)


        pts3d_n = torch.stack([pts3d_nx, pts3d_ny, pts3d_nz], dim=4)
        pts3d_l = torch.stack([pts3d_nxl, pts3d_nyl, pts3d_nzl], dim=4)
        pts3d_r = torch.stack([pts3d_nxr, pts3d_nyr, pts3d_nzr], dim=4)
        theta1 = (pts3d_nxl - pts3d_nx) / (torch.norm(pts3d_n - pts3d_l, dim=4))
        theta1 = torch.clamp(theta1, min=-0.999, max=0.999)
        theta1 = torch.acos(theta1)
        theta2 = (pts3d_nx - pts3d_nxr) / (torch.norm(pts3d_n - pts3d_r, dim=4))
        theta2 = torch.clamp(theta2, min=-1, max=1)
        theta2 = torch.acos(theta2)
        theta2 = theta2 - theta1
        return theta1, theta1

        # counts, bins = np.histogram(theta1[:,0,:,:].detach().cpu().numpy().flatten())
        # plt.hist(bins[:-1], bins, weights=counts)
        # counts, bins = np.histogram(theta1[:,1,:,:].detach().cpu().numpy().flatten())
        # plt.hist(bins[:-1], bins, weights=counts)
        # tensor2disp(theta1[:,1:2,:,:], vmax = 3.14, ind = 0).show()
        # tensor2disp(theta1[:, 0:1, :, :], vmax=3.14, ind=0).show()
        # import random
        # bz = random.randint(0, self.batch_size-1)
        # ch = random.randint(0, len(self.ptspair)-1)
        # h = random.randint(0, self.height-1)
        # w = random.randint(0, self.width - 1)
        #
        # target_lx = pts3d_nx[bz,ch,h + self.ptspair[ch][0][1],w + self.ptspair[ch][0][0]]
        # lx = pts3d_nxl[bz,ch,h,w]
        #
        # target_ly = pts3d_ny[bz,ch,h + self.ptspair[ch][0][1],w + self.ptspair[ch][0][0]]
        # ly = pts3d_nyl[bz,ch,h,w]
        #
        # target_lz = pts3d_nz[bz,ch,h + self.ptspair[ch][0][1],w + self.ptspair[ch][0][0]]
        # lz = pts3d_nzl[bz,ch,h,w]
        #
        # target_rx = pts3d_nx[bz,ch,h + self.ptspair[ch][1][1],w + self.ptspair[ch][1][0]]
        # rx = pts3d_nxr[bz,ch,h,w]
        #
        # target_ry = pts3d_ny[bz,ch,h + self.ptspair[ch][1][1],w + self.ptspair[ch][1][0]]
        # ry = pts3d_nyr[bz,ch,h,w]
        #
        # target_rz = pts3d_nz[bz,ch,h + self.ptspair[ch][1][1],w + self.ptspair[ch][1][0]]
        # rz = pts3d_nzr[bz,ch,h,w]
        #
        # assert (torch.abs(target_lx - lx) + torch.abs(target_ly - ly) + torch.abs(target_lz - lz)) + (torch.abs(target_rx - rx) + torch.abs(target_ry - ry) + torch.abs(target_rz - rz)) < 1e-3

        # import matlab
        # import matlab.engine
        # eng = matlab.engine.start_matlab()
        #
        # sampleR = 5
        # viewIndex = 0
        # dx = pts3d[viewIndex, :, :, 0, 0].cpu().detach().numpy().flatten()[::sampleR]
        # dy = pts3d[viewIndex, :, :, 1, 0].cpu().detach().numpy().flatten()[::sampleR]
        # dz = pts3d[viewIndex, :, :, 2, 0].cpu().detach().numpy().flatten()[::sampleR]
        #
        # dx = matlab.double(dx.tolist())
        # dy = matlab.double(dy.tolist())
        # dz = matlab.double(dz.tolist())
        #
        # eng.eval('close all', nargout=0)
        # eng.eval('figure()', nargout=0)
        # eng.eval('hold on', nargout=0)
        # eng.scatter3(dx, dy, dz, 5, 'filled', 'g', nargout=0)
        # eng.eval('axis equal', nargout=0)
        # eng.eval('grid off', nargout=0)
        # eng.eval('xlabel(\'X\')', nargout=0)
        # eng.eval('ylabel(\'Y\')', nargout=0)
        # eng.eval('zlabel(\'Z\')', nargout=0)
        # # eng.eval('xlim([0 50])', nargout=0)
        # # eng.eval('ylim([-40 40])', nargout=0)
        # # eng.eval('zlim([-3 10])', nargout=0)
        # tensor2disp(depthmap, vmax=80, ind=0).show()
        # return theta1, theta2
    def forward(self, predNorm, depthmap, invIn):
        predNorm_ = predNorm.view(self.batch_size, len(self.ptspair), 3, self.height, self.width).permute([0,1,3,4,2]).unsqueeze(4)
        intrinsic_c = invIn[:,0:3,0:3].view(self.batch_size,1,1,1,3,3)
        normIn = torch.matmul(predNorm_, intrinsic_c)

        p2d_ex = self.pixelLocs.unsqueeze(1).unsqueeze(5).expand([-1, len(self.ptspair), -1, -1, -1, -1])
        k = -torch.matmul(normIn, p2d_ex).squeeze(4).squeeze(4) * depthmap.expand([-1, len(self.ptspair), -1, -1])

        predDepth = normIn.unsqueeze(2).expand([-1,-1,2,-1,-1,-1,-1]) @ self.interestedLocs.unsqueeze(6)
        predDepth = predDepth.squeeze(5).squeeze(5)
        predDepth = -k.unsqueeze(2).expand([-1,-1,2,-1,-1]) / (predDepth)
        # ptspred3d = predDepth.unsqueeze(5).expand([-1,-1,-1,-1,-1,3]) * (intrinsic_c.unsqueeze(2).expand([-1, len(self.pixelLocs), 2, self.height, self.width, -1, -1]) @ self.interestedLocs.unsqueeze(6)).squeeze(6)
        # ck = torch.sum(ptspred3d * predNorm_.squeeze(4).unsqueeze(2).expand([-1,-1,2,-1,-1,-1]), axis = [5])
        # ck = ck + k.unsqueeze(2).expand([-1,-1,2,-1,-1])
        # torch.abs(ck).max()
        predDepth_ = predDepth.view(self.batch_size, len(self.ptspair) * 2, self.height, self.width)
        predDepth_ = self.copyConv(predDepth_)

        # import random
        # for i in range(100):
        #     bz = random.randint(0, self.batch_size - 1)
        #     chn = random.randint(0, 2 * len(self.ptspair) - 1)
        #     hn = random.randint(0, self.height - 1)
        #     wn = random.randint(0, self.width - 1)
        #
        #     dpval = predDepth_[bz, chn, hn, wn]
        #     sPts2d = torch.Tensor([wn, hn, 1]).float().cuda()
        #     sPts3d = invIn[bz,0:3,0:3] @ sPts2d.unsqueeze(1) * dpval
        #
        #     deltax = self.ptspair[int(chn / 2)][chn % 2][0]
        #     deltay = self.ptspair[int(chn / 2)][chn % 2][1]
        #     planeDir = predNorm_[bz, int(chn / 2), hn - deltay, wn - deltax, :, :]
        #     sk = k[bz, int(chn / 2), hn - deltay, wn - deltax]
        #
        #     val = (planeDir @ sPts3d)[0][0] + sk
        #
        #     print(val)





        # depthmap = torch.clamp(depthmap, min=0, max=100)
        # pts3d = invIn[:,0:3,0:3].view(self.batch_size,1,1,3,3).expand([-1, self.height, self.width, 3, 3]) @ self.pixelLocs.unsqueeze(4)
        # pts3d = pts3d * depthmap.squeeze(1).unsqueeze(3).unsqueeze(4).expand([-1, -1, -1, 3, 1])
        # pts3d = pts3d.squeeze(4).permute(0,3,1,2).contiguous()
        #
        # ck_p = torch.sum(predNorm.view(self.batch_size, len(self.ptspair), 3, self.height, self.width) * pts3d.unsqueeze(1).expand([-1,len(self.ptspair),-1,-1,-1]), axis = [2])
        # ck_p = ck_p + k
        # torch.abs(ck_p).max()
        #
        # tensor2disp((torch.abs(ck_p) > 100)[:,0:1,:,:], vmax = 1, ind = 0).show()
        # import matlab
        # import matlab.engine
        # eng = matlab.engine.start_matlab()
        #
        # sampleR = 5
        # viewIndex = 0
        # dx = pts3d[viewIndex, 0, :, :].cpu().detach().numpy().flatten()[::sampleR]
        # dy = pts3d[viewIndex, 1, :, :].cpu().detach().numpy().flatten()[::sampleR]
        # dz = pts3d[viewIndex, 2, :, :].cpu().detach().numpy().flatten()[::sampleR]
        #
        # dx = matlab.double(dx.tolist())
        # dy = matlab.double(dy.tolist())
        # dz = matlab.double(dz.tolist())
        #
        # eng.eval('close all', nargout=0)
        # eng.eval('figure()', nargout=0)
        # eng.eval('hold on', nargout=0)
        # eng.scatter3(dx, dy, dz, 5, 'filled', 'g', nargout=0)
        # eng.eval('axis equal', nargout=0)
        # eng.eval('grid off', nargout=0)
        # eng.eval('xlabel(\'X\')', nargout=0)
        # eng.eval('ylabel(\'Y\')', nargout=0)
        # eng.eval('zlabel(\'Z\')', nargout=0)
        # eng.eval('xlim([0 50])', nargout=0)
        # eng.eval('ylim([-40 40])', nargout=0)
        # eng.eval('zlim([-3 10])', nargout=0)
        return predDepth_
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
    def __init__(self, height, width, batch_size, lr, sampleNum = 10):
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
        # self.sampleNum = 10
        self.sampleNum = sampleNum

        self.lr = lr

    def get_camtrail(self, extrinsic, intrinsic):
        ws = 3
        y = 10
        d = 20
        xl = 0 - ws
        xr = self.width + ws

        st = np.log(0.05)
        ed = np.log(0.9)
        lw = torch.linspace(start=st, end=ed, steps=self.sampleNum).cuda()
        lw = torch.exp(lw)
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


    def erpipolar_rendering(self, depthmap, semanticmap, intrinsic, extrinsic, addmask_gt = None):
        # Compute Mask
        if addmask_gt is None:
            addmask = self.post_mask(depthmap = depthmap, semanticmap = semanticmap)
        else:
            addmask = addmask_gt

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
        return rimg, addmask, depthmapnp_grad

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
        testid = 45169
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


    def erpipolar_rendering_test_iterate(self, depthmap, semanticmap, intrinsic, extrinsic, writer):
        # Compute Mask
        addmask = self.post_mask(depthmap = depthmap, semanticmap = semanticmap)
        # Generate Camera Track
        camtrail, campos, viewbias, camdir = self.get_camtrail(extrinsic, intrinsic)
        nextrinsic = self.cvtCamtrail2Extrsic(camtrail, campos, viewbias, camdir, extrinsic)
        invcamK = torch.inverse(intrinsic @ extrinsic)
        epipoLine, Pcombined = self.get_eppl(intrinsic=intrinsic, extrinsic=extrinsic, nextrinsic=nextrinsic)
        r_sigma, inv_r_sigma, rotM = self.eppl2CovM(epipoLine)

        # Create log file


        # Compute GT Mask
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)
        projected2d, projecteddepth, selector = self.proj2de(pts3d=pts3d, intrinsic=intrinsic, nextrinsic=nextrinsic, addmask = addmask)
        rimg_gt, grad2d, _, depthmapnp_grad = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap.cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

        fig_gt = self.show_rendered_eppl(rimg_gt)

        depthmap_ns = depthmap + torch.randn(depthmap.shape, device=torch.device("cuda")) * 1
        # lr = 1e-4
        # lr = 1e-5
        # lr = 1e-6
        # lossrec = list()
        for i in range(200000000):
            pts3d_ns = self.bck(predDepth=depthmap_ns, invcamK=invcamK)
            projected2d_ns, projecteddepth_ns, selector_ns = self.proj2de(pts3d=pts3d_ns, intrinsic=intrinsic, nextrinsic=nextrinsic, addmask = addmask)
            # rimg_ns, grad2d_ns, _, depthmapnp_grad = eppl_render_l2(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d_ns.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector_ns.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap_ns.cpu().numpy(), rimg_gt = rimg_gt, kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
            rimg_ns, grad2d_ns, _, depthmapnp_grad = eppl_render_l1(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d_ns.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector_ns.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap_ns.cpu().numpy(), rimg_gt = rimg_gt, kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
            # rimg_ns, grad2d_ns, _, depthmapnp_grad = eppl_render_l1_sfgrad(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(),
            #                                                         pts2d=projected2d_ns.permute(
            #                                                             [0, 1, 3, 4, 2]).detach().cpu().numpy(),
            #                                                         mask=selector_ns.detach().cpu().numpy(),
            #                                                         Pcombinednp=Pcombined.cpu().numpy(),
            #                                                         depthmapnp=depthmap_ns.cpu().numpy(),
            #                                                         rimg_gt=rimg_gt, kws=self.kws, sr=self.sr,
            #                                                         bs=self.batch_size, samplesz=self.sampleNum * 2,
            #                                                         height=self.height, width=self.width)
            depthmap_ns = depthmap_ns - torch.from_numpy(depthmapnp_grad).cuda() * torch.Tensor([self.lr])[0].cuda()
            # lossrec.append(torch.sum(torch.abs(depthmap_ns - depthmap) * addmask.float()))
            val = torch.sum(torch.abs(depthmap_ns - depthmap) * addmask.float()).cpu().numpy()
            writer.add_scalar("Abs Diff", val, i)
            print(val)
            if np.mod(i, 50) == 0:
                fig_ns = self.show_rendered_eppl(rimg_ns)
                figcombined = pil.fromarray(np.concatenate([np.array(fig_gt), np.array(fig_ns)], axis=1))
                figcombinedT = torch.Tensor(np.array(figcombined)).permute([2,0,1]).float() / 255
                if os.path.isdir('/media/shengjie/other/Depins/Depins/visualization/oview_iterate'):
                    figcombined.save('/media/shengjie/other/Depins/Depins/visualization/oview_iterate/' + str(i) + '.png')
                writer.add_image('imresult', figcombinedT, i)
        return

    def show_rendered_eppl(self, rimg):
        vmax = rimg.max() * 0.5
        rimg_t = torch.from_numpy(rimg).unsqueeze(2)
        bz = 0
        imgt = list()
        for i in list(np.linspace(0,self.sampleNum -1 ,4).astype(np.int)):
        # for i in list(np.linspace(0,self.sampleNum -1 ,self.sampleNum).astype(np.int)):
            imgt.append(np.array(tensor2disp(rimg_t[bz], vmax=vmax, ind=i)))
        return pil.fromarray(np.concatenate(imgt, axis=0))

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
        rimg_gt, grad2d, _, depthmapnp_grad = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap.cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
        # rimg_gt, grad2d, _, depthmapnp_grad = eppl_render_l2(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap.cpu().numpy(), rimg_gt = rimg_gt, kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

        # depthmap_ns = depthmap + torch.randn(depthmap.shape, device=torch.device("cuda")) * 1e-2
        depthmap_ns = depthmap + torch.randn(depthmap.shape, device=torch.device("cuda")) * 1e-1
        pts3d_ns = self.bck(predDepth=depthmap_ns, invcamK=invcamK)
        projected2d_ns, projecteddepth_ns, selector_ns = self.proj2de(pts3d=pts3d_ns, intrinsic=intrinsic, nextrinsic=nextrinsic, addmask = addmask)
        rimg_ns, grad2d_ns, _, depthmapnp_grad = eppl_render_l1(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d_ns.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector_ns.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap_ns.cpu().numpy(), rimg_gt = rimg_gt, kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
        # rimg_ns, grad2d_ns, _, depthmapnp_grad = eppl_render_l2(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d_ns.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector_ns.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap_ns.cpu().numpy(), rimg_gt = rimg_gt, kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
        # rimg_ns, grad2d_ns, _, depthmapnp_grad = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d_ns.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector_ns.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap_ns.cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

        return


    def gradcheckl2(self, grad2d, depthmapnp_grad, Pcombined, depthmap_ns, invcamK, intrinsic, nextrinsic, addmask, selector_ns, projected2d_ns, inv_r_sigma, rimg_gt):
        ratio = 1
        height = self.height
        width = self.width

        mask = selector_ns.detach().cpu().numpy()
        testid = 5000
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

            rimgPlus, grad2d, _, _ = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2dPlus.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector_ns.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthvalPlus.cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
            rimgMinus, grad2d, _, _ = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2dMinus.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector_ns.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthvalMinus.cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

            pts2d = projected2d_ns.permute([0, 1, 3, 4, 2]).detach().cpu().numpy()
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
                            s1 = s1 + np.abs(rimgPlus[c, sz, j, i] - rimg_gt[c, sz, j, i])
                            s2 = s2 + np.abs(rimgMinus[c, sz, j, i] - rimg_gt[c, sz, j, i])
                            # s1 = s1 + rimgPlus[c, sz, j, i]
                            # s2 = s2 + rimgMinus[c, sz, j, i]
                            # s1 = s1 + (rimgPlus[c, sz, j, i] - rimg_gt[c, sz, j, i]) ** 2
                            # s2 = s2 + (rimgMinus[c, sz, j, i] - rimg_gt[c, sz, j, i]) ** 2
                            # s1 = s1 + rimgPlus[c, sz, j, i] * rimgPlus[c, sz, j, i]
                            # s2 = s2 + rimgMinus[c, sz, j, i] * rimgMinus[c, sz, j, i]
                            # s1 = s1 + np.log(10 + rimgPlus[c, sz, j, i])
                            # s2 = s2 + np.log(10 + rimgMinus[c, sz, j, i])
            # depthmapnp_grad[c, 0, yy, xx] / ((s1 - s2) / 2 /delta)
            # ((s1 - s2) / 2 / delta)
            if tt == 0:
                ratio = 1e-3 / np.abs((s1 - s2)) * ratio
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




class SelfOccluMask(nn.Module):
    def __init__(self, maxDisp = 21):
        super(SelfOccluMask, self).__init__()
        self.maxDisp = maxDisp
        self.init_kernel()

    def init_kernel(self):
        convweights = torch.zeros(self.maxDisp, 1, 3, self.maxDisp + 2)
        for i in range(0, self.maxDisp):
            convweights[i, 0, :, 0:2] = 1/6
            convweights[i, 0, :, i+2:i+3] = -1/3
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=self.maxDisp, kernel_size=(3,self.maxDisp + 2), stride=1, padding=self.maxDisp, bias=False)
        self.conv.bias = nn.Parameter(torch.arange(self.maxDisp).type(torch.FloatTensor) + 1, requires_grad=False)
        self.conv.weight = nn.Parameter(convweights, requires_grad=False)

        self.detectWidth = 19  # 3 by 7 size kernel
        self.detectHeight = 3
        convWeightsLeft = torch.zeros(1, 1, self.detectHeight, self.detectWidth)
        convWeightsRight = torch.zeros(1, 1, self.detectHeight, self.detectWidth)
        convWeightsLeft[0, 0, :, :int((self.detectWidth + 1) / 2)] = 1
        convWeightsRight[0, 0, :, int((self.detectWidth - 1) / 2):] = 1
        self.convLeft = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                        kernel_size=(self.detectHeight, self.detectWidth), stride=1,
                                        padding=[1, int((self.detectWidth - 1) / 2)], bias=False)
        self.convRight = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                         kernel_size=(self.detectHeight, self.detectWidth), stride=1,
                                         padding=[1, int((self.detectWidth - 1) / 2)], bias=False)
        self.convLeft.weight = nn.Parameter(convWeightsLeft, requires_grad=False)
        self.convRight.weight = nn.Parameter(convWeightsRight, requires_grad=False)

    def forward(self, dispmap, bsline):
        with torch.no_grad():
            maskl = self.computeMask(dispmap, direction='l')
            maskr = self.computeMask(dispmap, direction='r')
            lind = bsline < 0
            rind = bsline > 0
            mask = torch.zeros_like(dispmap)
            mask[lind,:, :, :] = maskl[lind,:, :, :]
            mask[rind, :, :, :] = maskr[rind, :, :, :]
            return mask

    def computeMask(self, dispmap, direction):
        width = dispmap.shape[3]
        if direction == 'l':
            output = self.conv(dispmap)
            output = torch.clamp(output, max=0)
            output = torch.min(output, dim=1, keepdim=True)[0]
            output = output[:, :, self.maxDisp - 1:-(self.maxDisp - 1):, -width:]
            output = torch.tanh(-output)
            mask = (output > 0.05).float()
        elif direction == 'r':
            dispmap_opp = torch.flip(dispmap, dims=[3])
            output_opp = self.conv(dispmap_opp)
            output_opp = torch.clamp(output_opp, max=0)
            output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
            output_opp = output_opp[:, :, self.maxDisp - 1:-(self.maxDisp - 1):, -width:]
            output_opp = torch.tanh(-output_opp)
            mask = (output_opp > 0.05).float()
            mask = torch.flip(mask, dims=[3])
        return mask


class grad_computation_tools(nn.Module):
    def __init__(self, batch_size, height, width):
        super(grad_computation_tools, self).__init__()
        weightsx = torch.Tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convDispx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispx.weight = nn.Parameter(weightsx, requires_grad=False)
        self.convDispy.weight = nn.Parameter(weightsy, requires_grad=False)

        self.disparityTh = 0.011
        self.semanticsTh = 0.6

        self.zeroRange = 2
        self.zero_mask = torch.ones([batch_size, 1, height, width]).cuda()
        self.zero_mask[:, :, :self.zeroRange, :] = 0
        self.zero_mask[:, :, -self.zeroRange:, :] = 0
        self.zero_mask[:, :, :, :self.zeroRange] = 0
        self.zero_mask[:, :, :, -self.zeroRange:] = 0

        self.mask = torch.ones([batch_size, 1, height, width], device=torch.device("cuda"))
        self.mask[:, :, 0:128, :] = 0

        self.foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17,
                               18]  # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle

    def get_disparityEdge(self, disparityMap):
        disparity_grad = torch.abs(self.convDispx(disparityMap)) + torch.abs(self.convDispy(disparityMap))
        disparity_grad = disparity_grad * self.zero_mask
        disparity_grad_bin = disparity_grad > self.disparityTh
        return disparity_grad_bin

    def get_semanticsEdge(self, semanticsMap):
        batch_size, c, height, width = semanticsMap.shape
        foregroundMapGt = torch.ones([batch_size, 1, height, width], dtype=torch.uint8, device=torch.device("cuda"))
        for m in self.foregroundType:
            foregroundMapGt = foregroundMapGt * (semanticsMap != m).byte()
        foregroundMapGt = (1 - foregroundMapGt).float()

        semantics_grad = torch.abs(self.convDispx(foregroundMapGt)) + torch.abs(self.convDispy(foregroundMapGt))
        semantics_grad = semantics_grad * self.zero_mask

        semantics_grad_bin = semantics_grad > self.semanticsTh

        return semantics_grad_bin

class TextureIndicatorM(nn.Module):
    def __init__(self):
        super(TextureIndicatorM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.refl = nn.ReflectionPad2d(1)

    def forward(self, x):
        x = self.refl(x)
        mu_x = self.mu_x_pool(x)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        return sigma_x


class MulScaleBCELoss(nn.Module):
    def __init__(self, scales):
        super(MulScaleBCELoss, self).__init__()
        self.scales = scales
        self.bcel = nn.BCELoss(reduction='none')
    def forward(self, pred_real, asSyn):
        l = 0
        _, _, height, width = pred_real[('syn_prob', 0)].shape
        for i in self.scales:
            if asSyn:
                l += torch.sum(self.bcel(F.interpolate(pred_real[('syn_prob', i)], [height, width], mode="bilinear",align_corners=False), torch.ones_like(pred_real[('syn_prob', 0)])) * pred_real['mask'][-1]) / torch.sum(pred_real['mask'][-1] + 1e-3)
            else:
                l += torch.sum(self.bcel(F.interpolate(pred_real[('syn_prob', i)], [height, width], mode="bilinear",align_corners=False), torch.zeros_like(pred_real[('syn_prob', 0)])) * pred_real['mask'][-1]) / torch.sum(pred_real['mask'][-1] + 1e-3)
        return l





class LocalThetaDesp(nn.Module):
    def __init__(self, height, width, batch_size, intrinsic):
        super(LocalThetaDesp, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.invIn = nn.Parameter(torch.from_numpy(np.linalg.inv(intrinsic)).float(), requires_grad = False)
        # self.ptspair = ptspair

        # Init grid points
        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.xx = torch.from_numpy(xx).float()
        self.yy = torch.from_numpy(yy).float()
        self.pixelLocs = nn.Parameter(torch.stack([self.xx, self.yy, torch.ones_like(self.xx)], dim=2), requires_grad=False)

        # Compute Horizontal Direciton
        hdir1 = self.invIn.unsqueeze(0).unsqueeze(0).expand([self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(3)
        hdir2 = self.invIn.unsqueeze(0).unsqueeze(0).expand([self.height, self.width, -1, -1]) @ (self.pixelLocs + torch.Tensor([1,0,0])).unsqueeze(3)
        hdir3 = torch.cross(hdir1, hdir2)
        hdir3 = hdir3 / torch.norm(hdir3, dim=2, keepdim=True)

        # torch.sum(hdir1[0,0,:,0] * hdir3[0,0,:,0])
        # torch.sum(hdir3[0,0,:,0] * hdir3[0,0,:,0])

        self.interestedLocs = torch.cat([self.xx, self.yy, torch.ones_like(self.xx)], dim=3).clone().view(self.batch_size, 1, 1, self.height, self.width, 3).repeat([1, len(self.ptspair), 3, 1,1, 1])

        for i in range(len(self.ptspair)):
            for j in range(3):
                if j != 0:
                    self.interestedLocs[:,i,j,:,:,:] = self.interestedLocs[:,i,j,:,:,:] + torch.from_numpy(np.array(self.ptspair[i][j-1])).float().view([1,1,1,3]).expand([self.batch_size, self.height, self.width, -1])

        y_dir = list()
        for i in range(len(self.ptspair)):
            tmp = np.array(self.ptspair[i][1]) - np.array(self.ptspair[i][0])
            tmp = tmp / np.sqrt(np.sum(tmp ** 2))
            y_dir.append(tmp)
        y_dir = np.stack(y_dir, axis=0)
        y_dir = torch.from_numpy(y_dir).float().view(1, len(self.ptspair), 1, 1, 3).expand(
            [self.batch_size, -1, self.height, self.width, -1])
        # Compute vertical direction
        vert_dir_tmp = torch.from_numpy(invIn).view(1,1,1,1,1,3,3).expand([self.batch_size, len(self.ptspair), 3, self.height, self.width, -1, -1]).float() @ self.interestedLocs.unsqueeze(6)
        vert_dir = torch.cross(vert_dir_tmp[:,:,1,:,:,:,0], vert_dir_tmp[:,:,2,:,:,:,0], dim=4)
        z_dir = vert_dir / torch.norm(vert_dir, keepdim = True, dim = 4).expand([-1,-1,-1,-1,3])
        x_dir = torch.cross(z_dir, y_dir, dim = 4)

        self.x_dir = torch.nn.Parameter(x_dir, requires_grad=False)
        self.z_dir = torch.nn.Parameter(z_dir, requires_grad=False)
        self.y_dir = torch.nn.Parameter(y_dir, requires_grad=False)
        self.invIn = torch.nn.Parameter(torch.from_numpy(invIn).float(), requires_grad=False)

        w = 4
        weightl = np.zeros([len(self.ptspair), 1, int(w * 2 + 1), int(w * 2 + 1)])
        for i in range(len(self.ptspair)):
            weightl[i, 0, self.ptspair[i][0][1] + w, self.ptspair[i][0][0] + w] = 1
        self.copyConv_thetal = torch.nn.Conv2d(len(self.ptspair), len(self.ptspair), int(w * 2 + 1), stride=1, padding=w, bias=False, groups=len(self.ptspair))
        self.copyConv_thetal.weight = torch.nn.Parameter(torch.from_numpy(weightl.astype(np.float32)), requires_grad=False)

        weightr = np.zeros([len(self.ptspair), 1, int(w * 2 + 1), int(w * 2 + 1)])
        for i in range(len(self.ptspair)):
            weightr[i, 0, self.ptspair[i][1][1] + w, self.ptspair[i][1][0] + w] = 1
        self.copyConv_thetar = torch.nn.Conv2d(len(self.ptspair), len(self.ptspair), int(w * 2 + 1), stride=1, padding=w, bias=False, groups=len(self.ptspair))
        self.copyConv_thetar.weight = torch.nn.Parameter(torch.from_numpy(weightr.astype(np.float32)), requires_grad=False)
    def get_theta(self, depthmap):
        invIn_ex = self.invIn.view(1,1,1,3,3).expand([self.batch_size, self.height, self.width, -1, -1])
        pts3d = (invIn_ex @ self.pixelLocs.unsqueeze(4)) * depthmap.squeeze(1).unsqueeze(3).unsqueeze(4).expand([-1,-1,-1,3,-1])
        pts3d_ex = pts3d.unsqueeze(1).squeeze(5).expand([-1,len(self.ptspair),-1,-1,-1])
        pts3d_nx = torch.sum(pts3d_ex * self.x_dir, dim=[4])
        pts3d_ny = torch.sum(pts3d_ex * self.y_dir,dim=[4])
        pts3d_nz = torch.sum(pts3d_ex * self.z_dir,dim=[4])

        pts3d_nxl = self.copyConv_thetal(pts3d_nx)
        pts3d_nyl = self.copyConv_thetal(pts3d_ny)
        pts3d_nzl = self.copyConv_thetal(pts3d_nz)


        pts3d_n = torch.stack([pts3d_nx, pts3d_ny, pts3d_nz], dim=4)
        pts3d_l = torch.stack([pts3d_nxl, pts3d_nyl, pts3d_nzl], dim=4)
        theta1 = (pts3d_nxl - pts3d_nx) / (torch.norm(pts3d_n - pts3d_l, dim=4))
        theta1 = torch.clamp(theta1, min=-0.999, max=0.999)
        theta1 = torch.acos(theta1)

        return theta1
    def forward(self, predNorm, depthmap, invIn):
        predNorm_ = predNorm.view(self.batch_size, len(self.ptspair), 3, self.height, self.width).permute([0,1,3,4,2]).unsqueeze(4)
        intrinsic_c = invIn[:,0:3,0:3].view(self.batch_size,1,1,1,3,3)
        normIn = torch.matmul(predNorm_, intrinsic_c)

        p2d_ex = self.pixelLocs.unsqueeze(1).unsqueeze(5).expand([-1, len(self.ptspair), -1, -1, -1, -1])
        k = -torch.matmul(normIn, p2d_ex).squeeze(4).squeeze(4) * depthmap.expand([-1, len(self.ptspair), -1, -1])

        predDepth = normIn.unsqueeze(2).expand([-1,-1,2,-1,-1,-1,-1]) @ self.interestedLocs.unsqueeze(6)
        predDepth = predDepth.squeeze(5).squeeze(5)
        predDepth = -k.unsqueeze(2).expand([-1,-1,2,-1,-1]) / (predDepth)
        # ptspred3d = predDepth.unsqueeze(5).expand([-1,-1,-1,-1,-1,3]) * (intrinsic_c.unsqueeze(2).expand([-1, len(self.pixelLocs), 2, self.height, self.width, -1, -1]) @ self.interestedLocs.unsqueeze(6)).squeeze(6)
        # ck = torch.sum(ptspred3d * predNorm_.squeeze(4).unsqueeze(2).expand([-1,-1,2,-1,-1,-1]), axis = [5])
        # ck = ck + k.unsqueeze(2).expand([-1,-1,2,-1,-1])
        # torch.abs(ck).max()
        predDepth_ = predDepth.view(self.batch_size, len(self.ptspair) * 2, self.height, self.width)
        predDepth_ = self.copyConv(predDepth_)
        return predDepth_