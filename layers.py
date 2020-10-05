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
from torch import autograd
import cv2

# import matlab
# import matlab.engine
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

class LocalThetaDesp(nn.Module):
    def __init__(self, height, width, batch_size, intrinsic, extrinsic = None, STEREO_SCALE_FACTOR = 5.4, minDepth = 0.1, maxDepth = 100, patchw = 15, patchh = 3):
        super(LocalThetaDesp, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size

        self.boundStabh = 0.02
        self.boundStabv = 0.02
        self.invIn = nn.Parameter(torch.from_numpy(np.linalg.inv(intrinsic)).float(), requires_grad = False)
        self.intrinsic = nn.Parameter(torch.from_numpy(intrinsic).float(), requires_grad=False)

        # Init grid points
        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.xx = nn.Parameter(torch.from_numpy(xx).float(), requires_grad=False)
        self.yy = nn.Parameter(torch.from_numpy(yy).float(), requires_grad=False)
        self.pixelLocs = nn.Parameter(torch.stack([self.xx, self.yy, torch.ones_like(self.xx)], dim=2), requires_grad=False)

        # Compute Horizontal Direciton
        hdir1 = self.invIn.unsqueeze(0).unsqueeze(0).expand([self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(3)
        hdir2 = self.invIn.unsqueeze(0).unsqueeze(0).expand([self.height, self.width, -1, -1]) @ (self.pixelLocs + torch.Tensor([1,0,0])).unsqueeze(3)
        hdir3 = torch.cross(hdir1, hdir2)
        hdir3 = hdir3 / torch.norm(hdir3, dim=2, keepdim=True)

        # Compute horizontal x axis
        hxd = torch.Tensor([0,0,1]).unsqueeze(1) - torch.sum(hdir3 * torch.Tensor([0,0,1]).unsqueeze(1), dim=[2,3], keepdim=True) * hdir3
        hxd = hxd / torch.norm(hxd, dim=2, keepdim=True)
        hyd = torch.cross(hxd, hdir3)
        hM = torch.stack([hxd.squeeze(3), hyd.squeeze(3)], dim=2)
        self.hM = nn.Parameter(hM, requires_grad = False)

        hdir1p = self.hM @ (hdir1 / torch.norm(hdir1, keepdim=True, dim = 2))
        hdir2p = self.hM @ (hdir2 / torch.norm(hdir2, keepdim=True, dim=2))

        lowerboundh = torch.atan2(hdir1p[:,:,1,0], hdir1p[:,:,0,0])
        lowerboundh = self.convert_htheta(lowerboundh) - float(np.pi) + self.boundStabh
        upperboundh = torch.atan2(hdir2p[:,:,1,0], hdir2p[:,:,0,0])
        upperboundh = self.convert_htheta(upperboundh) - self.boundStabh
        middeltargeth = (lowerboundh + upperboundh) / 2
        self.lowerboundh = nn.Parameter(lowerboundh.unsqueeze(0).unsqueeze(0).expand([self.batch_size,-1,-1,-1]), requires_grad = False)
        self.upperboundh = nn.Parameter(upperboundh.unsqueeze(0).unsqueeze(0).expand([self.batch_size,-1,-1,-1]), requires_grad=False)
        self.middeltargeth = nn.Parameter(middeltargeth.unsqueeze(0).unsqueeze(0).expand([self.batch_size,-1,-1,-1]), requires_grad=False)

        # Compute Vertical Direciton
        vdir1 = self.invIn.unsqueeze(0).unsqueeze(0).expand([self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(3)
        vdir2 = self.invIn.unsqueeze(0).unsqueeze(0).expand([self.height, self.width, -1, -1]) @ (self.pixelLocs + torch.Tensor([0,1,0])).unsqueeze(3)
        vdir3 = torch.cross(vdir1, vdir2)
        vdir3 = vdir3 / torch.norm(vdir3, dim=2, keepdim=True)

        # Compute vertical x axis
        vxd = torch.Tensor([0,0,1]).unsqueeze(1) - torch.sum(vdir3 * torch.Tensor([0,0,1]).unsqueeze(1), dim=[2,3], keepdim=True) * vdir3
        vxd = vxd / torch.norm(vxd, dim=2, keepdim=True)
        vyd = torch.cross(vxd, vdir3)
        vM = torch.stack([vxd.squeeze(3), vyd.squeeze(3)], dim=2)
        self.vM = nn.Parameter(vM, requires_grad = False)

        vdir1p = self.vM @ (vdir1 / torch.norm(vdir1, keepdim=True, dim = 2))
        vdir2p = self.vM @ (vdir2 / torch.norm(vdir2, keepdim=True, dim=2))

        lowerboundv = torch.atan2(vdir1p[:,:,1,0], vdir1p[:,:,0,0])
        lowerboundv = self.convert_vtheta(lowerboundv) - float(np.pi) + self.boundStabv
        upperboundv = torch.atan2(vdir2p[:,:,1,0], vdir2p[:,:,0,0])
        upperboundv = self.convert_vtheta(upperboundv) - self.boundStabv
        middeltargetv = (lowerboundv + upperboundv) / 2
        self.lowerboundv = nn.Parameter(lowerboundv.unsqueeze(0).unsqueeze(0).expand([self.batch_size,-1,-1,-1]), requires_grad = False)
        self.upperboundv = nn.Parameter(upperboundv.unsqueeze(0).unsqueeze(0).expand([self.batch_size,-1,-1,-1]), requires_grad=False)
        self.middeltargetv = nn.Parameter(middeltargetv.unsqueeze(0).unsqueeze(0).expand([self.batch_size,-1,-1,-1]), requires_grad=False)


        weightl = torch.Tensor(
            [[0,0,0],
            [0,-1,1],
            [0,0,0]]
        )
        self.hdiffConv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.hdiffConv.weight = torch.nn.Parameter(weightl.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)


        weightv = torch.Tensor(
            [[0,0,0],
            [0,-1,0],
            [0,1,0]]
        )
        self.vdiffConv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.vdiffConv.weight = torch.nn.Parameter(weightv.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)


        copyl = torch.Tensor(
            [[0,0,0],
            [0,0,1],
            [0,0,0]]
        )
        self.copylConv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.copylConv.weight = torch.nn.Parameter(copyl.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

        copyv = torch.Tensor(
            [[0,0,0],
            [0,0,0],
            [0,1,0]]
        )
        self.copyvConv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.copyvConv.weight = torch.nn.Parameter(copyv.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

        self.mink = -150
        self.maxk = 150


        npts3d = (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(0).unsqueeze(4).expand([self.batch_size,-1,-1,-1,-1]))

        npts3d_shifted_h = (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ (self.pixelLocs + torch.Tensor([1,0,0])).unsqueeze(0).unsqueeze(4).expand([self.batch_size,-1,-1,-1,-1]))
        npts3d_p_h = self.hM.unsqueeze(0).expand([self.batch_size, -1, -1, -1, -1]) @ npts3d
        npts3d_p_shifted_h = self.hM.unsqueeze(0).expand([self.batch_size, -1, -1, -1, -1]) @ npts3d_shifted_h
        self.npts3d_p_h = nn.Parameter(npts3d_p_h, requires_grad=False)
        self.npts3d_p_shifted_h = nn.Parameter(npts3d_p_shifted_h, requires_grad=False)

        npts3d_shifted_v = (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ (self.pixelLocs + torch.Tensor([0,1,0])).unsqueeze(0).unsqueeze(4).expand([self.batch_size,-1,-1,-1,-1]))
        npts3d_p_v = self.vM.unsqueeze(0).expand([self.batch_size, -1, -1, -1, -1]) @ npts3d
        npts3d_p_shifted_v = self.vM.unsqueeze(0).expand([self.batch_size, -1, -1, -1, -1]) @ npts3d_shifted_v
        self.npts3d_p_v = nn.Parameter(npts3d_p_v, requires_grad=False)
        self.npts3d_p_shifted_v = nn.Parameter(npts3d_p_shifted_v, requires_grad=False)

        lossh = torch.Tensor(
            [[0, 1, 1, 0, 0],
             [1, 1, 1, 0, 0],
             [1, 1, 1, 1, 0],
            ]
        )
        gth = torch.Tensor(
            [[0,-1,0,1,0],
             [-1, 0, 0, 1, 0],
             [-1, 0, 0, 0, 1],
            ]
        )
        idh = torch.Tensor(
            [[0,1,0,1,0],
             [1, 0, 0, 1, 0],
             [1, 0, 0, 0, 1],
            ]
        )
        evalcopyh = torch.Tensor(
            [[0,1,0,0,0],
             [1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0],
            ]
        )
        self.lossh = torch.nn.Conv2d(1, 3, [1,5], padding=[0,2], bias=False)
        self.lossh.weight = torch.nn.Parameter(lossh.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)
        self.gth = torch.nn.Conv2d(1, 3, [1,5], padding=[0,2], bias=False)
        self.gth.weight = torch.nn.Parameter(gth.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)
        self.idh = torch.nn.Conv2d(1, 3, [1,5], padding=[0,2], bias=False)
        self.idh.weight = torch.nn.Parameter(idh.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)
        self.evalcopyh = torch.nn.Conv2d(1, 3, [1,5], padding=[0,2], bias=False)
        self.evalcopyh.weight = torch.nn.Parameter(evalcopyh.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)

        lossv = torch.Tensor(
            [[1, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0],
            ]
        )
        gtv = torch.Tensor(
            [[-1, 0, 0, 1, 0, 0, 0, 0, 0],
             [-1, 0, 0, 0, 1, 0, 0, 0, 0],
             [-1, 0, 0, 0, 0, 1, 0, 0, 0],
             [-1, 0, 0, 0, 0, 0, 1, 0, 0],
             [-1, 0, 0, 0, 0, 0, 0, 1, 0],
             [-1, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        idv = torch.Tensor(
            [[1, 0, 0, 1, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        evalcopyv = torch.Tensor(
            [[1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        self.lossv = torch.nn.Conv2d(1, 6, [9,1], padding=[4,0], bias=False)
        self.lossv.weight = torch.nn.Parameter(lossv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)
        self.gtv = torch.nn.Conv2d(1, 6, [9,1], padding=[4,0], bias=False)
        self.gtv.weight = torch.nn.Parameter(gtv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)
        self.idv = torch.nn.Conv2d(1, 6, [9,1], padding=[4,0], bias=False)
        self.idv.weight = torch.nn.Parameter(idv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)
        self.evalcopyv = torch.nn.Conv2d(1, 6, [9,1], padding=[4,0], bias=False)
        self.evalcopyv.weight = torch.nn.Parameter(evalcopyv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)

        selfconh = torch.Tensor(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, -1]
            ]
        )
        self.selfconh = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconh.weight = torch.nn.Parameter(selfconh.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)
        selfconv = torch.Tensor(
            [[0, 0, 0],
             [0, 0, 1],
             [0, -1, 0]
            ]
        )
        self.selfconv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconv.weight = torch.nn.Parameter(selfconv.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

        selfconhIndW = torch.Tensor(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 1]
            ]
        )
        self.selfconhInd = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconhInd.weight = torch.nn.Parameter(selfconhIndW.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)
        selfconvIndW = torch.Tensor(
            [[0, 0, 0],
             [0, 0, 1],
             [0, 1, 0]
            ]
        )
        self.selfconvInd = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconvInd.weight = torch.nn.Parameter(selfconvIndW.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

        sobelx = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
        sobelx = sobelx / 4 / 2
        self.sobelx = nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)
        self.sobelx.weight = nn.Parameter(torch.from_numpy(sobelx).float().unsqueeze(0).unsqueeze(0), requires_grad = False)
        self.sobelx = self.sobelx.cuda()

        rgbgradkx = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])
        rgbgradkx = rgbgradkx / 4 / 2
        self.rgbgradkx = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)
        self.rgbgradkx.weight = nn.Parameter(torch.from_numpy(rgbgradkx).float().unsqueeze(0).unsqueeze(0).expand([-1, 3, -1, -1]), requires_grad = False)
        self.rgbgradkx = self.rgbgradkx.cuda()

        rgbgradky = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])
        rgbgradky = rgbgradky / 4 / 2
        self.rgbgradky = nn.Conv2d(in_channels=3,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)
        self.rgbgradky.weight = nn.Parameter(torch.from_numpy(rgbgradky).float().unsqueeze(0).unsqueeze(0).expand([-1, 3, -1, -1]), requires_grad = False)
        self.rgbgradky = self.rgbgradky.cuda()

        w = 39
        self.h_pool = nn.AvgPool2d(w, 1, padding=int((w-1)/2))

    def convert_htheta(self, htheta):
        htheta = htheta + float(np.pi) / 2 * 3
        htheta = torch.fmod(htheta, float(np.pi) * 2)
        return htheta

    def backconvert_htheta(self, htheta):
        htheta = htheta - float(np.pi) / 2 * 3
        return htheta

    def convert_vtheta(self, vtheta):
        vtheta = vtheta + float(np.pi) / 2 * 3
        vtheta = torch.fmod(vtheta, float(np.pi) * 2)
        return vtheta

    def backconvert_vtheta(self, vtheta):
        vtheta = vtheta - float(np.pi) / 2 * 3
        return vtheta

    def get_theta(self, depthmap):
        pts3d = depthmap.squeeze(1).unsqueeze(3).unsqueeze(4).expand([-1,-1,-1,3,-1]) * (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(0).unsqueeze(4).expand([self.batch_size,-1,-1,-1,-1]))

        hcord = self.hM.unsqueeze(0).expand([self.batch_size, -1, -1, -1, -1]) @ pts3d
        hdify = self.hdiffConv(hcord[:,:,:,1,0].unsqueeze(1))
        hdifx = self.hdiffConv(hcord[:,:,:,0,0].unsqueeze(1))

        htheta = torch.atan2(hdify, hdifx)
        htheta = self.convert_htheta(htheta)
        htheta = torch.clamp(htheta, min = 1e-3, max = float(np.pi) * 2 - 1e-3)

        vcord = self.vM.unsqueeze(0).expand([self.batch_size, -1, -1, -1, -1]) @ pts3d
        vdify = self.vdiffConv(vcord[:,:,:,1,0].unsqueeze(1))
        vdifx = self.vdiffConv(vcord[:,:,:,0,0].unsqueeze(1))

        vtheta = torch.atan2(vdify, vdifx)
        vtheta = self.convert_vtheta(vtheta)
        vtheta = torch.clamp(vtheta, min=1e-3, max=float(np.pi) * 2 - 1e-3)
        return htheta, vtheta

    def get_optimization_kernel(self, optimize_width):
        opt_ratio_accumks_h = list()
        opt_depth_copyks_h = list()
        for i in range(-optimize_width, optimize_width + 1):
            cur_ratio_accumks = np.zeros([1,int(2 * optimize_width + 1)])
            cur_depth_copyks = np.zeros([1, int(2 * optimize_width + 1)])
            if i < 0:
                for m in range(i, 0):
                    cur_ratio_accumks[0, optimize_width + m] = -1
                cur_depth_copyks[0, optimize_width + i] = 1
                opt_ratio_accumks_h.append(cur_ratio_accumks)
                opt_depth_copyks_h.append(cur_depth_copyks)
            elif i > 0:
                for m in range(0, i):
                    cur_ratio_accumks[0, optimize_width + m] = 1
                cur_depth_copyks[0, optimize_width + i] = 1
                opt_ratio_accumks_h.append(cur_ratio_accumks)
                opt_depth_copyks_h.append(cur_depth_copyks)
        opt_ratio_accumks_h = torch.from_numpy(np.concatenate(opt_ratio_accumks_h, axis=0)).float()
        self.opt_ratio_accumks_h = torch.nn.Conv2d(1, int(2 * optimize_width), [1, int(2 * optimize_width + 1)], padding=[0, optimize_width], bias=False)
        self.opt_ratio_accumks_h.weight = torch.nn.Parameter(opt_ratio_accumks_h.unsqueeze(1).unsqueeze(2), requires_grad=False)
        opt_depth_copyks_h = torch.from_numpy(np.concatenate(opt_depth_copyks_h, axis=0)).float()
        self.opt_depth_copyks_h = torch.nn.Conv2d(1, int(2 * optimize_width), [1, int(2 * optimize_width + 1)], padding=[0, optimize_width], bias=False)
        self.opt_depth_copyks_h.weight = torch.nn.Parameter(opt_depth_copyks_h.unsqueeze(1).unsqueeze(2), requires_grad=False)

    def cleaned_path_loss(self, depthmap, htheta, vtheta):
        depthmapl = torch.log(torch.clamp(depthmap, min = 1e-3))
        inboundh = (htheta < self.upperboundh) * (htheta > self.lowerboundh)
        inboundh = inboundh.float()
        outboundh = 1 - inboundh

        bk_htheta = self.backconvert_htheta(htheta)
        npts3d_pdiff_uph = torch.cos(bk_htheta) * self.npts3d_p_h[:,:,:,1,0].unsqueeze(1) - torch.sin(bk_htheta) * self.npts3d_p_h[:,:,:,0,0].unsqueeze(1)
        npts3d_pdiff_downh = torch.cos(bk_htheta) * self.npts3d_p_shifted_h[:, :, :, 1, 0].unsqueeze(1) - torch.sin(bk_htheta) * self.npts3d_p_shifted_h[:, :, :, 0, 0].unsqueeze(1)
        ratiohl = torch.log(torch.clamp(torch.abs(npts3d_pdiff_uph), min = 1e-4)) - torch.log(torch.clamp(torch.abs(npts3d_pdiff_downh), min = 1e-4))

        lossh = self.lossh(ratiohl)
        gth = self.gth(depthmapl)
        indh = (self.idh((depthmap > 0).float()) == 2).float() * (self.lossh(outboundh) == 0).float()
        hloss = (torch.sum(torch.abs(gth - lossh) * inboundh * indh) + torch.sum(torch.abs(self.middeltargeth - htheta) * outboundh * indh) / 20) / (torch.sum(indh) + 1)

        inboundv = (vtheta < self.upperboundv) * (vtheta > self.lowerboundv)
        inboundv = inboundv.float()
        outboundv = 1 - inboundv

        bk_vtheta = self.backconvert_vtheta(vtheta)
        npts3d_pdiff_upv = torch.cos(bk_vtheta) * self.npts3d_p_v[:,:,:,1,0].unsqueeze(1) - torch.sin(bk_vtheta) * self.npts3d_p_v[:,:,:,0,0].unsqueeze(1)
        npts3d_pdiff_downv = torch.cos(bk_vtheta) * self.npts3d_p_shifted_v[:, :, :, 1, 0].unsqueeze(1) - torch.sin(bk_vtheta) * self.npts3d_p_shifted_v[:, :, :, 0, 0].unsqueeze(1)
        ratiovl = torch.log(torch.clamp(torch.abs(npts3d_pdiff_upv), min = 1e-4)) - torch.log(torch.clamp(torch.abs(npts3d_pdiff_downv), min = 1e-4))

        lossv = self.lossv(ratiovl)
        gtv = self.gtv(depthmapl)
        indv = (self.idv((depthmap > 0).float()) == 2).float() * (self.lossv(outboundv) == 0).float()
        vloss = (torch.sum(torch.abs(gtv - lossv) * indv * inboundv) + torch.sum(torch.abs(self.middeltargetv - vtheta) * indv * outboundv) / 20) / (torch.sum(indv) + 1)

        scl_pixelwise = self.selfconh(ratiohl) + self.selfconv(ratiovl)
        scl_mask = (self.selfconvInd(inboundv) == 2).float() * (self.selfconhInd(inboundh) == 2).float()
        scl = torch.sum(torch.abs(scl_pixelwise) * scl_mask) / (torch.sum(scl_mask) + 1)

        return hloss, vloss, scl

    def get_ratio(self, htheta, vtheta):
        bk_htheta = self.backconvert_htheta(htheta)

        npts3d_pdiff_uph = torch.cos(bk_htheta).squeeze(1) * self.npts3d_p_h[:,:,:,1,0] - torch.sin(bk_htheta).squeeze(1) * self.npts3d_p_h[:,:,:,0,0]
        npts3d_pdiff_downh = torch.cos(bk_htheta).squeeze(1) * self.npts3d_p_shifted_h[:, :, :, 1, 0] - torch.sin(bk_htheta).squeeze(1) * self.npts3d_p_shifted_h[:, :, :, 0, 0]
        ratioh = npts3d_pdiff_uph / npts3d_pdiff_downh
        ratioh = torch.clamp(ratioh, min = 1e-3)
        ratiohl = torch.log(ratioh)

        bk_vtheta = self.backconvert_vtheta(vtheta)
        npts3d_pdiff_upv = torch.cos(bk_vtheta).squeeze(1) * self.npts3d_p_v[:,:,:,1,0] - torch.sin(bk_vtheta).squeeze(1) * self.npts3d_p_v[:,:,:,0,0]
        npts3d_pdiff_downv = torch.cos(bk_vtheta).squeeze(1) * self.npts3d_p_shifted_v[:, :, :, 1, 0] - torch.sin(bk_vtheta).squeeze(1) * self.npts3d_p_shifted_v[:, :, :, 0, 0]
        ratiov = npts3d_pdiff_upv / npts3d_pdiff_downv
        ratiov = torch.clamp(ratiov, min = 1e-3)
        ratiovl = torch.log(ratiov)

        ratioh = ratioh.unsqueeze(1)
        ratiohl = ratiohl.unsqueeze(1)
        ratiov = ratiov.unsqueeze(1)
        ratiovl = ratiovl.unsqueeze(1)
        return ratioh, ratiohl, ratiov, ratiovl

    def depth_localgeom_consistency(self, depthmap, htheta, vtheta, rgb, isdebias=False):
        if isdebias:
            htheta_d, vtheta_d = self.get_theta(depthmap.detach())
            debias_hthtea = self.h_pool(htheta_d) + (htheta - self.h_pool(htheta))
        else:
            debias_hthtea = htheta

        optimize_mask = torch.zeros_like(depthmap)
        optimize_mask[:,:,int(0.40810811 * self.height):int(0.99189189 * self.height), int(0.03594771 * self.width):int(0.96405229 * self.width)] = 1

        bk_htheta = self.backconvert_htheta(debias_hthtea)
        dirx_h = torch.cos(bk_htheta)
        diry_h = torch.sin(bk_htheta)
        dir3d_h = self.hM[:,:,0,:].unsqueeze(0).expand([self.batch_size, -1, -1, -1]) * dirx_h.squeeze(1).unsqueeze(3).expand([-1, -1, -1, 3]) + \
                  self.hM[:,:,1,:].unsqueeze(0).expand([self.batch_size, -1, -1, -1]) * diry_h.squeeze(1).unsqueeze(3).expand([-1, -1, -1, 3])
        dir3d_h = dir3d_h / torch.sqrt(torch.sum(dir3d_h * dir3d_h, dim=3, keepdim=True))

        u0 = self.xx.unsqueeze(0).unsqueeze(0).expand([self.batch_size, -1, -1, -1])
        bx = self.intrinsic[0,2]
        fx = self.intrinsic[0,0]
        nx_nz_rat = (dir3d_h[:,:,:,0] / dir3d_h[:,:,:,2]).unsqueeze(1)
        denominator = (u0 - nx_nz_rat * fx - bx)

        absminth = 1e-4
        ckselector = torch.abs(denominator) < absminth
        assert torch.sum(ckselector) == 0, print("Please complete bad gradient clip work")

        derivx = - depthmap / denominator
        num_grad = self.sobelx(depthmap)

        rgb_grad = torch.mean(torch.abs(self.rgbgradkx(rgb)) + torch.abs(self.rgbgradky(rgb)), dim=1, keepdim=True)
        rgb_gradw = torch.exp(-rgb_grad * 2)
        closs = torch.sum(torch.abs(derivx - num_grad) * optimize_mask * rgb_gradw) / (torch.sum(optimize_mask) + 1)

        # # Check for correctness of the derivation
        # pts3d = depthmap.squeeze(1).unsqueeze(3).unsqueeze(4).expand([-1,-1,-1,3,-1]) * (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(0).unsqueeze(4).expand([self.batch_size,-1,-1,-1,-1]))
        #
        # rndw = np.random.randint(0, self.width - 1)
        # rndh = np.random.randint(0, self.height - 1)
        # pts3d_ck = pts3d[0, rndh, rndw, :, 0]
        # dirck = dir3d_h[0, rndh, rndw, :]
        # devt = 0.001
        #
        # pts3d_ck_dev = pts3d_ck + dirck * devt
        # projected_pts3d_ck_dev = self.intrinsic @ pts3d_ck_dev.unsqueeze(1)
        # projected_pts3d_ck_dev[0] = projected_pts3d_ck_dev[0] / projected_pts3d_ck_dev[2]
        # projected_pts3d_ck_dev[1] = projected_pts3d_ck_dev[1] / projected_pts3d_ck_dev[2]
        #
        # num_grad = (projected_pts3d_ck_dev[2] - depthmap[0,0,rndh,rndw]) / (projected_pts3d_ck_dev[0] - rndw)
        # computed_grad = derivx[0,0,rndh,rndw]
        # print(num_grad / computed_grad)

        return closs, derivx, num_grad, rgb_gradw


    def surfnorm_from_localgeom(self, htheta, vtheta):
        bk_htheta = self.backconvert_htheta(htheta)
        dirx_h = torch.cos(bk_htheta)
        diry_h = torch.sin(bk_htheta)
        dir3d_h = self.hM[:,:,0,:].unsqueeze(0).expand([self.batch_size, -1, -1, -1]) * dirx_h.squeeze(1).unsqueeze(3).expand([-1, -1, -1, 3]) + \
                  self.hM[:,:,1,:].unsqueeze(0).expand([self.batch_size, -1, -1, -1]) * diry_h.squeeze(1).unsqueeze(3).expand([-1, -1, -1, 3])
        dir3d_h = dir3d_h / torch.sqrt(torch.sum(dir3d_h * dir3d_h, dim=3, keepdim=True))

        bk_vtheta = self.backconvert_vtheta(vtheta)
        dirx_v = torch.cos(bk_vtheta)
        diry_v = torch.sin(bk_vtheta)
        dir3d_v = self.vM[:,:,0,:].unsqueeze(0).expand([self.batch_size, -1, -1, -1]) * dirx_v.squeeze(1).unsqueeze(3).expand([-1, -1, -1, 3]) + \
                  self.vM[:,:,1,:].unsqueeze(0).expand([self.batch_size, -1, -1, -1]) * diry_v.squeeze(1).unsqueeze(3).expand([-1, -1, -1, 3])
        dir3d_v = dir3d_v / torch.sqrt(torch.sum(dir3d_v * dir3d_v, dim=3, keepdim=True))

        dir3d = torch.cross(dir3d_h, dir3d_v, dim=3)
        return dir3d

    def vls_geompred(self, depthmap, htheta, vtheta, rgb, depthmaplidar, eng=None, instancemap=None):
        optrat = 0.6
        sth = int(self.height * (1 - optrat))
        stw = int(self.width * (1 - optrat) / 2)
        edw = int(self.width * (1 - (1 - optrat) / 2))

        optimize_mask_np = np.zeros_like(instancemap)
        optimize_mask_np[sth::, stw:edw] = 1

        fidalMask_np = np.zeros([self.height, self.width], dtype=np.bool)
        horConsMask_np = np.zeros([self.height, self.width], dtype=np.bool)
        verConsMask_np = np.zeros([self.height, self.width], dtype=np.bool)
        datavalmask = depthmap[0,0,:,:].detach().cpu().numpy() > 0
        init_arb_mask(self.height, self.width, optimize_mask_np, fidalMask_np, horConsMask_np, verConsMask_np, datavalmask)

        penalrat = 2
        rgb_grad = torch.mean(torch.abs(self.rgbgradkx(rgb)) + torch.abs(self.rgbgradky(rgb)), dim=1, keepdim=True)
        rgb_gradw = 1 - torch.exp(-rgb_grad * penalrat)

        from sparsMMul import SparsMMul
        sparsmmul = SparsMMul.apply

        lambdafw = 0.5

        optimize_mask = torch.from_numpy(optimize_mask_np.astype(np.float32)).cuda() == 1

        linearIndexMap = torch.ones([self.height, self.width], device="cuda") * (-1)
        linearIndexMap[optimize_mask] = torch.arange(0, torch.sum(optimize_mask).float(), device="cuda")
        linearIndexMap = linearIndexMap.long()

        # Init Fidality terms
        fidalMask = torch.from_numpy(fidalMask_np).cuda() == 1
        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        xx = torch.from_numpy(xx).cuda()
        yy = torch.from_numpy(yy).cuda()

        # Init Horizontal Consrtain terms
        horConsMask = torch.from_numpy(horConsMask_np).cuda() == 1

        # Init Vertical Consrtain terms
        verConsMask = torch.from_numpy(verConsMask_np).cuda() == 1

        ratioh, ratiohl, ratiov, ratiovl = self.get_ratio(htheta, vtheta)

        fidalrv = depthmap[0,0,:,:][fidalMask] * rgb_gradw[0,0,:,:][fidalMask]
        fidalindy = torch.arange(0, torch.sum(fidalMask), device="cuda", dtype=torch.long)
        fidalindx = linearIndexMap[fidalMask]
        fidalw = torch.ones([torch.sum(fidalMask)], device="cuda", dtype=torch.float) * rgb_gradw[0,0,:,:][fidalMask]

        conshw1 = ratioh[0,0,:,:][horConsMask] * (1 - rgb_gradw[0,0,:,:][horConsMask])
        consindhy1 = torch.arange(torch.sum(fidalMask), torch.sum(fidalMask) + torch.sum(horConsMask), device="cuda", dtype=torch.long)
        consindhx1 = linearIndexMap[yy[horConsMask], xx[horConsMask]]
        conshw2 = -torch.ones([torch.sum(horConsMask)], device="cuda", dtype=torch.float) * (1 - rgb_gradw[0,0,:,:][horConsMask])
        consindhy2 = torch.arange(torch.sum(fidalMask), torch.sum(fidalMask) + torch.sum(horConsMask), device="cuda", dtype=torch.long)
        consindhx2 = linearIndexMap[yy[horConsMask], xx[horConsMask] + 1]
        conshw = torch.cat([conshw1, conshw2], dim=0)
        consindhy = torch.cat([consindhy1, consindhy2], dim=0)
        consindhx = torch.cat([consindhx1, consindhx2], dim=0)
        conshrv = torch.zeros(torch.sum(horConsMask), device="cuda")

        consvw1 = ratiov[0,0,:,:][verConsMask] * (1 - rgb_gradw[0,0,:,:][verConsMask])
        consindvy1 = torch.arange(torch.sum(fidalMask) + torch.sum(horConsMask), torch.sum(fidalMask) + torch.sum(horConsMask) + torch.sum(verConsMask), device="cuda", dtype=torch.long)
        consindvx1 = linearIndexMap[yy[verConsMask], xx[verConsMask]]
        consvw2 = -torch.ones([torch.sum(verConsMask)], device="cuda", dtype=torch.float) * (1 - rgb_gradw[0,0,:,:][verConsMask])
        consindvy2 = torch.arange(torch.sum(fidalMask) + torch.sum(horConsMask), torch.sum(fidalMask) + torch.sum(horConsMask) + torch.sum(verConsMask), device="cuda", dtype=torch.long)
        consindvx2 = linearIndexMap[yy[verConsMask] + 1, xx[verConsMask]]
        consvw = torch.cat([consvw1, consvw2], dim=0)
        consindvy = torch.cat([consindvy1, consindvy2], dim=0)
        consindvx = torch.cat([consindvx1, consindvx2], dim=0)
        consvrv = torch.zeros(torch.sum(verConsMask), device="cuda")

        spsindx = torch.cat([fidalindx, consindhx, consindvx], dim=0).contiguous()
        spsindy = torch.cat([fidalindy, consindhy, consindvy], dim=0).contiguous()
        spsindw = torch.cat([fidalw, conshw, consvw], dim=0).contiguous()
        spsrv = torch.cat([fidalrv, conshrv, consvrv], dim=0).contiguous()

        spsSizeM = torch.Tensor([torch.sum(fidalMask) + torch.sum(horConsMask) + torch.sum(verConsMask), torch.sum(optimize_mask)]).int().cuda()

        th = spsSizeM[0]
        tw = spsSizeM[1]
        sizeM = spsSizeM

        x0 = torch.zeros([tw], device="cuda", dtype=torch.float)
        d0 = spsrv
        r0 = sparsmmul(spsindx, spsindy, sizeM, spsindw, spsrv, True)
        p0 = sparsmmul(spsindx, spsindy, sizeM, spsindw, spsrv, True)
        t0 = sparsmmul(spsindx, spsindy, sizeM, spsindw, p0, False)
        for i in range(10):
            if torch.sum(t0 * t0) < 1:
                break
            alpha = torch.sum(r0 * r0) / torch.sum(t0 * t0)
            x0 = x0 + alpha * p0
            d0 = d0 - alpha * t0
            r1 = sparsmmul(spsindx, spsindy, sizeM, spsindw, d0, True)
            if torch.sum(r0 * r0) < 1:
                break
            beta = torch.sum(r1 * r1) / torch.sum(r0 * r0)
            p0 = r1 + beta * p0
            t0 = sparsmmul(spsindx, spsindy, sizeM, spsindw, p0, False)
            r0 = r1

        depthmaplidar_torch = torch.from_numpy(depthmaplidar).float().unsqueeze(0).unsqueeze(0).cuda()
        recovered_depth = torch.zeros_like(depthmap)
        recovered_depth[0,0,:,:][yy[optimize_mask], xx[optimize_mask]] = x0

        # ====Visualization Part====
        # pts3d_recovered = recovered_depth.squeeze(1).unsqueeze(3).unsqueeze(4).expand([-1, -1, -1, 3, -1]) * (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(0).unsqueeze(4).expand([self.batch_size, -1, -1, -1, -1]))
        # pts3d_org = depthmap.squeeze(1).unsqueeze(3).unsqueeze(4).expand([-1, -1, -1, 3, -1]) * (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(0).unsqueeze(4).expand([self.batch_size, -1, -1, -1, -1]))
        # pts3d_lidar = depthmaplidar_torch.squeeze(1).unsqueeze(3).unsqueeze(4).expand([-1, -1, -1, 3, -1]) * (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(0).unsqueeze(4).expand([self.batch_size, -1, -1, -1, -1]))
        #
        # tensor2disp(1/recovered_depth, vmax=0.2, ind=0).show()
        # tensor2disp(1 / depthmap, vmax=0.2, ind=0).show()
        #
        # htheta_rec, vtheta_rec = self.get_theta(recovered_depth)
        # htheta_org, vtheta_org = self.get_theta(depthmap)
        # tensor2disp(htheta_rec-1, vmax=4, ind=0).show()
        # tensor2disp(htheta_org-1, vmax=4, ind=0).show()
        #
        # vlsareanp = vlsarea.cpu().numpy()[0, 0, :, :] == 1
        # vlsareanp_lidar = (vlsareanp * (depthmaplidar > 0)) == True
        #
        # def extract_pts(torchpts, numpymask):
        #     import matlab
        #     import matlab.engine
        #     nppts = torchpts[0,:,:,:,0].detach().cpu().numpy()
        #     npx = nppts[:, :, 0][numpymask]
        #     npy = nppts[:, :, 1][numpymask]
        #     npz = nppts[:, :, 2][numpymask]
        #
        #     npx = matlab.double(npx.tolist())
        #     npy = matlab.double(npy.tolist())
        #     npz = matlab.double(npz.tolist())
        #
        #     return npx, npy, npz
        #
        # npxorg, npyorg, npzorg = extract_pts(pts3d_org, vlsareanp)
        # npxopted, npyopted, npzopted = extract_pts(pts3d_recovered, vlsareanp)
        # npxlidar, npylidar, npzlidar = extract_pts(pts3d_lidar, vlsareanp)
        #
        # import matlab
        # eng.eval('subplot(1,2,1)', nargout=0)
        # eng.scatter3(npxopted, npyopted, npzopted, 3, 'g', 'filled', nargout=0)
        # eng.eval('axis equal', nargout=0)
        # eng.title('From shape', nargout=0)
        # xlim = eng.eval('xlim', nargout=1)
        # ylim = eng.eval('ylim', nargout=1)
        # zlim = eng.eval('zlim', nargout=1)
        #
        # eng.eval('subplot(1,2,2)', nargout=0)
        # eng.scatter3(npxorg, npyorg, npzorg, 3, 'k', 'filled', nargout=0)
        # eng.eval('axis equal', nargout=0)
        # eng.title('From Depth', nargout=0)
        # eng.xlim(xlim, nargout=0)
        # eng.ylim(ylim, nargout=0)
        # eng.zlim(zlim, nargout=0)

        return recovered_depth

class ConsistLoss(nn.Module):
    def __init__(self):
        super(ConsistLoss, self).__init__()
        weightsl = torch.Tensor([[0., 0., 0.],
                                [-1., 1., 0.],
                                [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)

        weightsr = torch.Tensor([[0., 0., 0.],
                                [0., -1., 1.],
                                [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)


        weightsu = torch.Tensor([[0., 0., 0.],
                                 [0., -1., 0.],
                                 [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        weightsd = torch.Tensor([[0., -1., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)

        self.diffxl = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.diffxr = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.diffyu = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.diffyd = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.diffxl.weight = nn.Parameter(weightsl, requires_grad=False)
        self.diffxr.weight = nn.Parameter(weightsr, requires_grad=False)
        self.diffyu.weight = nn.Parameter(weightsu, requires_grad=False)
        self.diffyd.weight = nn.Parameter(weightsd, requires_grad=False)

    def grad_consistloss(self, depth, gradx, grady, w):
        if torch.abs(gradx).max() >= 9e5 or torch.abs(grady).max() >= 9e5:
            return 0
        else:
            consistl = torch.abs(self.diffxl(depth) - gradx)
            consistr = torch.abs(self.diffxr(depth) - gradx)
            consistu = torch.abs(self.diffyu(depth) - grady)
            consistd = torch.abs(self.diffyd(depth) - grady)

            consistloss = torch.sum((consistl + consistr + consistu + consistd) * w) / (torch.sum(w) + 1)
            # tensor2grad(gradx * w, pos_bar=0.1, neg_bar=-0.1).show()
            # tensor2grad(self.diffxl(depth) * w, pos_bar=0.1, neg_bar=-0.1).show()
            # tensor2grad(grady * w, pos_bar=0.2, neg_bar=-0.2).show()
            # tensor2grad(self.diffyu(depth) * w, pos_bar=0.2, neg_bar=-0.2).show()
            return consistloss

    def linearity_consistloss(self, depthMap, anghw, angvw):
        depthMapl = torch.log(depthMap)
        gradxl = self.diffxl(depthMapl).squeeze(1)
        gradxr = self.diffxr(depthMapl).squeeze(1)
        gradyu = self.diffyu(depthMapl).squeeze(1)
        gradyd = self.diffyd(depthMapl).squeeze(1)

        colinearityloss = (torch.sum(torch.abs(gradxl - gradxr) * anghw) / (torch.sum(anghw) + 1) + torch.sum(torch.abs(gradyu - gradyd) * angvw) / (torch.sum(angvw) + 1)) / 2
        # tensor2disp(anghw, vmax=1, ind=0).show()
        # tensor2disp(angvw, vmax=1, ind=0).show()
        return colinearityloss

class ImageWeightComputer(nn.Module):
    def __init__(self):
        super(ImageWeightComputer, self).__init__()
        rgbgradkx = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])
        rgbgradkx = rgbgradkx / 4 / 2
        self.rgbgradkx = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.rgbgradkx.weight = nn.Parameter(torch.from_numpy(rgbgradkx).float().unsqueeze(0).unsqueeze(0).expand([-1, 3, -1, -1]), requires_grad = False)
        self.rgbgradkx = self.rgbgradkx.cuda()

        rgbgradky = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])
        rgbgradky = rgbgradky / 4 / 2
        self.rgbgradky = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.rgbgradky.weight = nn.Parameter(torch.from_numpy(rgbgradky).float().unsqueeze(0).unsqueeze(0).expand([-1, 3, -1, -1]), requires_grad = False)
        self.rgbgradky = self.rgbgradky.cuda()

        anggradkx = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])
        anggradkx = anggradkx / 4 / 2
        self.anggradkx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.anggradkx.weight = nn.Parameter(torch.from_numpy(anggradkx).float().unsqueeze(0).unsqueeze(0), requires_grad = False)
        self.anggradkx = self.anggradkx.cuda()

        anggradky = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])
        anggradky = anggradky / 4 / 2
        self.anggradky = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.anggradky.weight = nn.Parameter(torch.from_numpy(anggradky).float().unsqueeze(0).unsqueeze(0), requires_grad = False)
        self.anggradky = self.anggradky.cuda()

    def rgbgradw(self, rgb):
        rgb_grad = torch.mean(torch.abs(self.rgbgradkx(rgb)) + torch.abs(self.rgbgradky(rgb)), dim=1, keepdim=True)
        rgb_gradw = torch.exp(-rgb_grad * 2)
        rgb_gradw[rgb_gradw < 0.5] = 0

        return rgb_gradw

    def anggradw(self, angh, angv):
        anghgrad = torch.abs(self.anggradkx(angh)) + torch.abs(self.anggradky(angh))
        anghw = torch.exp(-anghgrad * 20)
        anghw[anghw < 0.5] = 0
        # fig, ax = plt.subplots()
        # im = ax.imshow(anghw[0,0,:,:].detach().cpu().numpy())
        # fig.colorbar(im)
        # tensor2disp(anghw, vmax=1, ind=0).show()

        angvgrad = torch.abs(self.anggradkx(angv)) + torch.abs(self.anggradky(angv))
        angvw = torch.exp(-angvgrad * 20)
        angvw[angvw < 0.5] = 0
        # fig, ax = plt.subplots()
        # im = ax.imshow(angvw[0,0,:,:].detach().cpu().numpy())
        # fig.colorbar(im)
        # tensor2disp(angvw, vmax=1, ind=0).show()
        return anghw, angvw


class SurfaceNormalOptimizer(nn.Module):
    def __init__(self, height, width, batch_size, angw=1e-6, vlossw=0.2, sclw=0):
        super(SurfaceNormalOptimizer, self).__init__()
        # intrinsic: (batch_size, 4, 4)
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.angw = angw
        self.vlossw = vlossw
        self.sclw = sclw

        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.xx = nn.Parameter(torch.from_numpy(np.copy(xx)).unsqueeze(0).repeat([self.batch_size, 1, 1]).float(), requires_grad=False)
        self.yy = nn.Parameter(torch.from_numpy(np.copy(yy)).unsqueeze(0).repeat([self.batch_size, 1, 1]).float(), requires_grad=False)

        xx = xx.flatten().astype(np.float32)
        yy = yy.flatten().astype(np.float32)
        pix_coords = np.expand_dims(np.stack([xx, yy, np.ones(self.width * self.height).astype(np.float32)], axis=1), axis=0).repeat(self.batch_size, axis=0)
        self.pix_coords = nn.Parameter(torch.from_numpy(pix_coords).permute(0, 2, 1), requires_grad=False)
        self.ones = nn.Parameter(torch.ones([self.batch_size, 1, self.height * self.width]), requires_grad=False)
        self.init_gradconv()
        self.init_integration_kernel()
        self.init_scalIndex()
        self.init_patchIntPath()

    def init_scalIndex(self):
        self.xxdict = dict()
        self.yydict = dict()
        self.ccdict = dict()
        for i in range(1, 4):
            cuh = int(self.height / (2 ** i))
            cuw = int(self.width / (2 ** i))

            cx = int(0 + 2 ** i / 2)
            cy = int(0 + 2 ** i / 2)

            cxx, cyy = np.meshgrid(range(cuw), range(cuh), indexing='xy')

            cxx = cxx * (2**i) + cx
            cyy = cyy * (2**i) + cy

            cxxt = nn.Parameter(torch.from_numpy(np.copy(cxx)).unsqueeze(0).repeat([self.batch_size, 1, 1]).long(), requires_grad=False)
            cyyt = nn.Parameter(torch.from_numpy(np.copy(cyy)).unsqueeze(0).repeat([self.batch_size, 1, 1]).long(), requires_grad=False)
            ccct = nn.Parameter(torch.zeros([self.batch_size, cuh, cuw], dtype=torch.long), requires_grad=False)

            self.xxdict["scale_{}".format(i)] = cxxt
            self.yydict["scale_{}".format(i)] = cyyt
            self.ccdict["scale_{}".format(i)] = ccct

    def init_patchIntPath(self):
        self.patchIntPath = dict()
        for i in range(1, 4):
            intpath = list()
            for m in range(0, int(2 ** i)):
                for n in range(0, int(2 ** i)):
                    if m == 0 and n == 0:
                        continue
                    if np.mod(m, 2) == 1:
                        mm = int(-(m+1)/2)
                    else:
                        mm = int(m/2)
                    if np.mod(n, 2) == 1:
                        nn = int(-(n+1)/2)
                    else:
                        nn = int(n/2)

                    if nn < 0:
                        logx1 = nn
                        logy1 = mm
                        ch1 = 0
                        depthx1 = nn + 1
                        depthy1 = mm
                        sign1 = -1
                    elif nn > 0:
                        logx1 = nn - 1
                        logy1 = mm
                        ch1 = 0
                        depthx1 = nn - 1
                        depthy1 = mm
                        sign1 = 1

                    if mm > 0:
                        logx2 = nn
                        logy2 = mm - 1
                        ch2 = 1
                        depthx2 = nn
                        depthy2 = mm - 1
                        sign2 = 1
                    elif mm < 0:
                        logx2 = nn
                        logy2 = mm
                        ch2 = 1
                        depthx2 = nn
                        depthy2 = mm + 1
                        sign2 = -1

                    if nn == 0:
                        logx1 = logx2
                        logy1 = logy2
                        ch1 = ch2
                        depthx1 = depthx2
                        depthy1 = depthy2
                        sign1 = sign2

                    if mm == 0:
                        logx2 = logx1
                        logy2 = logy1
                        ch2 = ch1
                        depthx2 = depthx1
                        depthy2 = depthy1
                        sign2 = sign1

                    intpath.append((logx1, logy1, ch1, depthx1, depthy1, sign1, logx2, logy2, ch2, depthx2, depthy2, sign2, mm, nn))
                    self.patchIntPath['scale_{}'.format(i)] = intpath

    def patchIntegration(self, depthmaplow, ang, intrinsic, scale):
        log = self.ang2log(intrinsic=intrinsic, ang=ang)
        depthmaplowl = torch.log(depthmaplow.squeeze(1))
        _, lh, lw = depthmaplowl.shape
        assert (lh == self.height / (2 ** scale)) and (lw == self.width / (2 ** scale)), print("Resolution and scale are not cooresponded")

        xx = self.xxdict["scale_{}".format(scale)]
        yy = self.yydict["scale_{}".format(scale)]
        cc = self.ccdict["scale_{}".format(scale)]
        intpaths = self.patchIntPath['scale_{}'.format(scale)]
        depthmapr = torch.zeros([self.batch_size, self.height, self.width], device="cuda")
        depthmapr[:, yy[0], xx[0]] = depthmaplowl

        for path in intpaths:
            logx1, logy1, ch1, depthx1, depthy1, sign1, logx2, logy2, ch2, depthx2, depthy2, sign2, mm, nn = path
            tmpgradnode = depthmapr.clone()
            depthmapr[:, yy + mm, xx + nn] = (tmpgradnode[:, yy + depthy1, xx + depthx1] + sign1 * log[:, cc + ch1, yy + logy1, xx + logx1] +
                                              tmpgradnode[:, yy + depthy2, xx + depthx2] + sign2 * log[:, cc + ch2, yy + logy2, xx + logx2]) / 2

        depthmapr = torch.exp(depthmapr).unsqueeze(1)
        # tensor2disp(depthmaplow, vmax=40, ind=0).show()
        # tensor2disp(depthmapr, vmax=40, ind=0).show()
        return depthmapr

    def init_patchIntPath_debug(self, depthmap, ang, intrinsic):
        log = self.ang2log(intrinsic=intrinsic, ang=ang)
        depthmaps = depthmap.squeeze(1)
        depthmapl = torch.log(depthmaps)
        depthreocver = torch.zeros([self.batch_size, self.height, self.width], device='cuda')

        for i in range(1, 4):
            xx = self.xxdict["scale_{}".format(i)]
            yy = self.yydict["scale_{}".format(i)]
            cc = self.ccdict["scale_{}".format(i)]
            for m in range(0, int(2 ** i)):
                for n in range(0, int(2 ** i)):
                    if m == 0 and n == 0:
                        depthreocver[:, yy, xx] = depthmapl[:, yy, xx]
                        continue
                    if np.mod(m, 2) == 1:
                        mm = int(-(m+1)/2)
                    else:
                        mm = int(m/2)
                    if np.mod(n, 2) == 1:
                        nn = int(-(n+1)/2)
                    else:
                        nn = int(n/2)

                    if nn < 0:
                        logx1 = nn
                        logy1 = mm
                        ch1 = 0
                        depthx1 = nn + 1
                        depthy1 = mm
                        sign1 = -1
                    elif nn > 0:
                        logx1 = nn - 1
                        logy1 = mm
                        ch1 = 0
                        depthx1 = nn - 1
                        depthy1 = mm
                        sign1 = 1

                    if mm > 0:
                        logx2 = nn
                        logy2 = mm - 1
                        ch2 = 1
                        depthx2 = nn
                        depthy2 = mm - 1
                        sign2 = 1
                    elif mm < 0:
                        logx2 = nn
                        logy2 = mm
                        ch2 = 1
                        depthx2 = nn
                        depthy2 = mm + 1
                        sign2 = -1

                    if nn == 0:
                        logx1 = logx2
                        logy1 = logy2
                        ch1 = ch2
                        depthx1 = depthx2
                        depthy1 = depthy2
                        sign1 = sign2

                    if mm == 0:
                        logx2 = logx1
                        logy2 = logy1
                        ch2 = ch1
                        depthx2 = depthx1
                        depthy2 = depthy1
                        sign2 = sign1

                    depthreocver[:, yy + mm, xx + nn] = (depthreocver[:, yy + depthy1, xx + depthx1] + sign1 * log[:, cc + ch1, yy + logy1, xx + logx1] +
                                                         depthreocver[:, yy + depthy2, xx + depthx2] + sign2 * log[:, cc + ch2, yy + logy2, xx + logx2]) / 2
                    import random
                    rckidx = np.random.randint(0, xx.shape[1])
                    rckidy = np.random.randint(0, yy.shape[1])

                    ckcc = np.random.randint(0, self.batch_size)
                    ckxx = xx[ckcc, rckidy, rckidx]
                    ckyy = yy[ckcc, rckidy, rckidx]

                    torch.log(depthmaps[ckcc, ckyy, ckxx + 1]) - torch.log(depthmaps[ckcc, ckyy, ckxx])
                    log[ckcc, 0, ckyy, ckxx]

                    targetdl = depthmapl[ckcc, ckyy + mm, ckxx + nn]

                    sourcedl1 = depthmapl[ckcc, ckyy + depthy1, ckxx + depthx1]
                    linklog1 = log[ckcc, ch1, ckyy + logy1, ckxx + logx1]
                    ck1 = torch.abs(torch.exp(targetdl) - torch.exp(sourcedl1 + linklog1 * sign1))
                    ckk1 = torch.abs(torch.log(depthmap[ckcc, 0, ckyy + mm, ckxx + nn] / depthmap[ckcc, 0, ckyy + depthy1, ckxx + depthx1]) - linklog1 * sign1)

                    sourcedl2 = depthmapl[ckcc, ckyy + depthy2, ckxx + depthx2]
                    linklog2 = log[ckcc, ch2, ckyy + logy2, ckxx + logx2]
                    ck2 = torch.abs(torch.exp(targetdl) - torch.exp(sourcedl2 + linklog2 * sign2))
                    ckk2 = torch.abs(torch.log(depthmap[ckcc, 0, ckyy + mm, ckxx + nn] / depthmap[ckcc, 0, ckyy + depthy2, ckxx + depthx2]) - linklog2 * sign2)

                    # print("Err analysis:(%f, %f)" % (float(ckk1.detach().cpu().numpy()), float(ckk2.detach().cpu().numpy())))
                    if i == 3:
                        print((logx1, logy1, ch1, depthx1, depthy1, sign1, logx2, logy2, ch2, depthx2, depthy2, sign2, mm, nn))

            err = torch.abs(depthreocver - depthmapl)
            tensor2disp(torch.exp(depthmapl).unsqueeze(1), vmax=40, ind=0).show()

    def init_integration_kernel(self):
        inth = torch.Tensor(
            [[0, 1, 0, 0, 0],
             [0, 1, 1, 0, 0],
             [1, 1, 1, 0, 0],
             [1, 1, 1, 1, 0],
            ]
        )
        gth = torch.Tensor(
            [[0, -1, 1, 0, 0],
             [0, -1, 0, 1, 0],
             [-1, 0, 0, 1, 0],
             [-1, 0, 0, 0, 1],
            ]
        )
        idh = torch.Tensor(
            [[0, 1, 1, 0, 0],
             [0, 1, 0, 1, 0],
             [1, 0, 0, 1, 0],
             [1, 0, 0, 0, 1],
            ]
        )
        self.inth = torch.nn.Conv2d(1, 4, [1, 5], padding=[0, 2], bias=False)
        self.inth.weight = torch.nn.Parameter(inth.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)
        self.gth = torch.nn.Conv2d(1, 4, [1, 5], padding=[0, 2], bias=False)
        self.gth.weight = torch.nn.Parameter(gth.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)
        self.idh = torch.nn.Conv2d(1, 4, [1, 5], padding=[0, 2], bias=False)
        self.idh.weight = torch.nn.Parameter(idh.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)


        intv = torch.Tensor(
            [[1, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0],
            ]
        )
        gtv = torch.Tensor(
            [[-1, 0, 0, 1, 0, 0, 0, 0, 0],
             [-1, 0, 0, 0, 1, 0, 0, 0, 0],
             [-1, 0, 0, 0, 0, 1, 0, 0, 0],
             [-1, 0, 0, 0, 0, 0, 1, 0, 0],
             [-1, 0, 0, 0, 0, 0, 0, 1, 0],
             [-1, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        idv = torch.Tensor(
            [[1, 0, 0, 1, 0, 0, 0, 0, 0],
             [1, 0, 0, 0, 1, 0, 0, 0, 0],
             [1, 0, 0, 0, 0, 1, 0, 0, 0],
             [1, 0, 0, 0, 0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0, 0, 0, 1, 0],
             [1, 0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        self.intv = torch.nn.Conv2d(1, 6, [9, 1], padding=[4, 0], bias=False)
        self.intv.weight = torch.nn.Parameter(intv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)
        self.gtv = torch.nn.Conv2d(1, 6, [9, 1], padding=[4, 0], bias=False)
        self.gtv.weight = torch.nn.Parameter(gtv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)
        self.idv = torch.nn.Conv2d(1, 6, [9, 1], padding=[4, 0], bias=False)
        self.idv.weight = torch.nn.Parameter(idv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)

        selfconh = torch.Tensor(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, -1]
            ]
        )
        self.selfconh = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconh.weight = torch.nn.Parameter(selfconh.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)
        selfconv = torch.Tensor(
            [[0, 0, 0],
             [0, 0, 1],
             [0, -1, 0]
            ]
        )
        self.selfconv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconv.weight = torch.nn.Parameter(selfconv.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

        selfconhIndW = torch.Tensor(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 1]
            ]
        )
        self.selfconhInd = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconhInd.weight = torch.nn.Parameter(selfconhIndW.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)
        selfconvIndW = torch.Tensor(
            [[0, 0, 0],
             [0, 0, 1],
             [0, 1, 0]
            ]
        )
        self.selfconvInd = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconvInd.weight = torch.nn.Parameter(selfconvIndW.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

    def init_gradconv(self):
        weightsx = torch.Tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        weightsx = weightsx / 4 / 2

        weightsy = torch.Tensor([[-1., -2., -1.],
                                 [0., 0., 0.],
                                 [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        weightsy = weightsy / 4 / 2
        self.diffx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.diffy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.diffx.weight = nn.Parameter(weightsx, requires_grad=False)
        self.diffy.weight = nn.Parameter(weightsy, requires_grad=False)

        weightsx = torch.Tensor([[0., 0., 0.],
                                [0., -1., 1.],
                                [0., 0., 0.]]).unsqueeze(0).unsqueeze(0)
        weightsx = weightsx

        weightsy = torch.Tensor([[0., 0., 0.],
                                 [0., -1., 0.],
                                 [0., 1., 0.]]).unsqueeze(0).unsqueeze(0)
        weightsy = weightsy
        self.diffx_sharp = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.diffy_sharp = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        self.diffx_sharp.weight = nn.Parameter(weightsx, requires_grad=False)
        self.diffy_sharp.weight = nn.Parameter(weightsy, requires_grad=False)

    def init_sparsegradconv(self, height, width, sigmah, sigmaw):
        assert (np.mod(width, 2) == 1) and (np.mod(height, 2) == 1)
        centerw = int((width - 1) / 2)
        centerh = int((height - 1) / 2)

        kernelpos = np.zeros([height, width], dtype=np.float32)
        kernelneg = np.zeros([height, width], dtype=np.float32)
        for i in range(height):
            for j in range(width):
                if j < centerw:
                    kernelneg[i, j] = -np.exp(-(((i - centerh) / sigmah) ** 2 + ((j - centerw) / sigmaw) ** 2))
                elif j > centerw:
                    kernelpos[i, j] = np.exp(-(((i - centerh) / sigmah) ** 2 + ((j - centerw) / sigmaw) ** 2))
        kernelneg = kernelneg / np.sum(np.abs(kernelneg)) / 2
        kernelpos = kernelpos / np.sum(np.abs(kernelpos)) / 2

        self.kernelnegh = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[height, width], padding=[centerh, centerw], bias=False)
        self.kernelposh = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[height, width], padding=[centerh, centerw], bias=False)

        self.kernelnegh.weight = nn.Parameter(torch.from_numpy(kernelneg).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.kernelposh.weight = nn.Parameter(torch.from_numpy(kernelpos).unsqueeze(0).unsqueeze(0), requires_grad=False)

        self.kernelnegv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[width, height], padding=[centerw, centerh], bias=False)
        self.kernelposv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[width, height], padding=[centerw, centerh], bias=False)

        self.kernelnegv.weight = nn.Parameter(torch.from_numpy(np.copy(kernelneg.T)).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.kernelposv.weight = nn.Parameter(torch.from_numpy(np.copy(kernelpos.T)).unsqueeze(0).unsqueeze(0), requires_grad=False)

    def get_depth_numgrad(self, depthMap, issharp=True):
        if issharp:
            depthMap_gradx = self.diffx_sharp(depthMap)
            depthMap_grady = self.diffy_sharp(depthMap)
        else:
            depthMap_gradx = self.diffx(depthMap)
            depthMap_grady = self.diffy(depthMap)
        return depthMap_gradx, depthMap_grady

    def depth2norm(self, depthMap, intrinsic, issharp=True):
        depthMaps = depthMap.squeeze(1)
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        if issharp:
            depthMap_gradx = self.diffx_sharp(depthMap).squeeze(1)
            depthMap_grady = self.diffy_sharp(depthMap).squeeze(1)
        else:
            depthMap_gradx = self.diffx(depthMap).squeeze(1)
            depthMap_grady = self.diffy(depthMap).squeeze(1)

        vx1 = depthMaps / fx + (self.xx - bx) / fx * depthMap_gradx
        vx2 = (self.yy - by) / fy * depthMap_gradx
        vx3 = depthMap_gradx

        vy1 = (self.xx - bx) / fx * depthMap_grady
        vy2 = depthMaps / fy + (self.yy - by) / fy * depthMap_grady
        vy3 = depthMap_grady

        vx = torch.stack([vx1, vx2, vx3], dim=1)
        vy = torch.stack([vy1, vy2, vy3], dim=1)

        surfnorm = torch.cross(vx, vy, dim=1)
        surfnorm = F.normalize(surfnorm, dim=1)
        surfnorm = torch.clamp(surfnorm, min=-1+1e-6, max=1-1e-6)

        # vind = 1
        # tensor2disp(surfnorm[:, 0:1, :, :] + 1, vmax=2, ind=vind).show()
        # tensor2disp(surfnorm[:, 1:2, :, :] + 1, vmax=2, ind=vind).show()
        # tensor2disp(surfnorm[:, 2:3, :, :] + 1, vmax=2, ind=vind).show()
        # tensor2rgb((surfnorm + 1) / 2, ind=vind).show()

        return surfnorm

    def depth2ang(self, depthMap, intrinsic, issharp=True):
        depthMaps = depthMap.squeeze(1)
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        if issharp:
            depthMap_gradx = self.diffx_sharp(depthMap).squeeze(1)
            depthMap_grady = self.diffy_sharp(depthMap).squeeze(1)
        else:
            depthMap_gradx = self.diffx(depthMap).squeeze(1)
            depthMap_grady = self.diffy(depthMap).squeeze(1)

        vx1 = depthMaps / fx + (self.xx - bx) / fx * depthMap_gradx
        vx2 = (self.yy - by) / fy * depthMap_gradx
        vx3 = depthMap_gradx

        vy1 = (self.xx - bx) / fx * depthMap_grady
        vy2 = depthMaps / fy + (self.yy - by) / fy * depthMap_grady
        vy3 = depthMap_grady

        a = (self.yy - by) / fy * vx2 + vx3
        b = -vx1

        u = vy3 + (self.xx - bx) / fx * vy1
        v = -vy2

        angh = torch.atan2(a, -b)
        angv = torch.atan2(u, -v)

        angh = angh.unsqueeze(1)
        angv = angv.unsqueeze(1)

        ang = torch.cat([angh, angv], dim=1)

        # tensor2disp(angh + np.pi, vmax=2*np.pi, ind=0).show()
        # tensor2disp(angv + np.pi, vmax=2*np.pi, ind=0).show()
        return ang

    def ang2log(self, intrinsic, ang):
        protectmin = 1e-6

        angh = ang[:, 0, :, :]
        angv = ang[:, 1, :, :]

        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        a1 = ((self.yy - by) / fy)**2 + 1
        b1 = -(self.xx - bx) / fx

        a2 = ((self.yy - by) / fy)**2 + 1
        b2 = -(self.xx + 1 - bx) / fx

        a3 = torch.sin(angh)
        b3 = -torch.cos(angh)

        u1 = ((self.xx - bx) / fx)**2 + 1
        v1 = -(self.yy - by) / fy

        u2 = ((self.xx - bx) / fx)**2 + 1
        v2 = -(self.yy + 1 - by) / fy

        u3 = torch.sin(angv)
        v3 = -torch.cos(angv)

        logh = torch.log(torch.clamp(torch.abs(a3 * b1 - a1 * b3), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3 * b2 - a2 * b3), min=protectmin))
        logh = torch.clamp(logh, min=-10, max=10)

        logv = torch.log(torch.clamp(torch.abs(u3 * v1 - u1 * v3), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3 * v2 - u2 * v3), min=protectmin))
        logv = torch.clamp(logv, min=-10, max=10)

        return torch.stack([logh, logv], dim=1)

    def ang2err_loss(self, angpred, intrinsic, depthMap, errpred):
        protectmin = 1e-6

        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        errangacth = errpred[:, 0, :, :]
        errangactv = errpred[:, 1, :, :]

        errangh_min = 0
        errangh_max = torch.atan2(fx, torch.ones_like(fx)) - protectmin

        errangh = errangacth * (errangh_max - errangh_min) + errangh_min
        errlogh = -torch.log(1 - torch.tan(errangh) / fx)
        errlogh = errlogh.unsqueeze(1)

        errangv_min = 0
        errangv_max = torch.atan2(fy, torch.ones_like(fy)) - protectmin

        errangv = errangactv * (errangv_max - errangv_min) + errangv_min
        errlogv = -torch.log(1 - torch.tan(errangv) / fy)
        errlogv = errlogv.unsqueeze(1)

        angh = angpred[:, 0, :, :]
        angv = angpred[:, 1, :, :]

        a1 = ((self.yy - by) / fy)**2 + 1
        b1 = -(self.xx - bx) / fx

        a2 = ((self.yy - by) / fy)**2 + 1
        b2 = -(self.xx + 1 - bx) / fx

        a3 = torch.sin(angh)
        b3 = -torch.cos(angh)

        u1 = ((self.xx - bx) / fx)**2 + 1
        v1 = -(self.yy - by) / fy

        u2 = ((self.xx - bx) / fx)**2 + 1
        v2 = -(self.yy + 1 - by) / fy

        u3 = torch.sin(angv)
        v3 = -torch.cos(angv)

        depthMapl = torch.log(torch.clamp(depthMap, min=protectmin))

        logh = torch.log(torch.clamp(torch.abs(a3 * b1 - a1 * b3), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3 * b2 - a2 * b3), min=protectmin))
        logh = torch.clamp(logh, min=-10, max=10)

        logv = torch.log(torch.clamp(torch.abs(u3 * v1 - u1 * v3), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3 * v2 - u2 * v3), min=protectmin))
        logv = torch.clamp(logv, min=-10, max=10)

        logh = logh.unsqueeze(1)
        logv = logv.unsqueeze(1)

        vallidarmask = (depthMap > 0).float()

        indh = (self.idh(vallidarmask) == 2).float()
        indv = (self.idv(vallidarmask) == 2).float()

        inth = self.inth(logh)
        gth = self.gth(depthMapl)
        gtherr = torch.abs(inth - gth)
        estherr = self.inth(errlogh)

        intv = self.intv(logv)
        gtv = self.gtv(depthMapl)
        gtverr = torch.abs(intv - gtv)
        estverr = self.intv(errlogv)

        loss = torch.sum(torch.abs(gtherr - estherr) * indh) / torch.sum(indh) + \
               torch.sum(torch.abs(gtverr - estverr) * indv) / torch.sum(indv)

        loss = loss / 2

        # # check
        # log = self.ang2log(intrinsic=intrinsic, ang=angpred)
        #
        # fxnp = float(intrinsic[0, 0, 0].detach().cpu().numpy())
        # actnp = np.linspace(0, 1, 10000)
        # angminnp = 0
        # angmaxnp = float(torch.atan2(fx, torch.ones_like(fx))[0, 0, 0].detach().cpu().numpy()) - protectmin
        # actthtahnp = actnp * (angmaxnp - angminnp) + angminnp
        # loghnp = -np.log(1 - np.tan(actthtahnp) / fxnp)
        #
        # loghprednp = np.abs(log[:, 0, :, :].detach().cpu().numpy().flatten())
        # hist, bin_edges = np.histogram(loghprednp, loghnp)
        # hist = hist / np.sum(hist)
        # f, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.plot(actnp, loghnp)
        # ax1.set_xlabel('Sigmoid Activation value')
        # ax1.set_ylabel('Sigmoid Activation corresponded Log value')
        # ax2.plot(actnp[:-1], hist)
        # ax2.set_xlabel('Pred log value mapped Sigmoid Activation')
        # ax2.set_ylabel('Probability Density')
        #
        # fynp = float(intrinsic[0, 1, 1].detach().cpu().numpy())
        # actnp = np.linspace(0, 1, 10000)
        # angminnp = 0
        # angmaxnp = float(torch.atan2(fy, torch.ones_like(fy))[0, 0, 0].detach().cpu().numpy()) - protectmin
        # actthtahnp = actnp * (angmaxnp - angminnp) + angminnp
        # logvnp = -np.log(1 - np.tan(actthtahnp) / fynp)
        #
        # logvprednp = np.abs(log[:, 1, :, :].detach().cpu().numpy().flatten())
        # hist, bin_edges = np.histogram(logvprednp, logvnp)
        # hist = hist / np.sum(hist)
        # f, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.plot(actnp, loghnp)
        # ax1.set_xlabel('Sigmoid Activation value')
        # ax1.set_ylabel('Sigmoid Activation corresponded Log value')
        # ax2.plot(actnp[:-1], hist)
        # ax2.set_xlabel('Pred log value mapped Sigmoid Activation')
        # ax2.set_ylabel('Probability Density')
        return loss

    def depth2err_loss(self, depthpred, intrinsic, depthMap, errpred):
        protectmin = 1e-6

        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        errangacth = errpred[:, 0, :, :]
        errangactv = errpred[:, 1, :, :]

        errangh_min = 0
        errangh_max = torch.atan2(fx, torch.ones_like(fx)) - protectmin

        errangh = errangacth * (errangh_max - errangh_min) + errangh_min
        errlogh = -torch.log(1 - torch.tan(errangh) / fx)
        errlogh = errlogh.unsqueeze(1)

        errangv_min = 0
        errangv_max = torch.atan2(fy, torch.ones_like(fy)) - protectmin

        errangv = errangactv * (errangv_max - errangv_min) + errangv_min
        errlogv = -torch.log(1 - torch.tan(errangv) / fy)
        errlogv = errlogv.unsqueeze(1)

        depthpredl = torch.log(depthpred)

        depthMapl = torch.log(torch.clamp(depthMap, min=protectmin))

        vallidarmask = (depthMap > 0).float()

        indh = (self.idh(vallidarmask) == 2).float()
        indv = (self.idv(vallidarmask) == 2).float()

        inth = self.gth(depthpredl)
        gth = self.gth(depthMapl)
        gtherr = torch.abs(inth - gth)
        estherr = self.idh(errlogh)

        intv = self.gtv(depthpredl)
        gtv = self.gtv(depthMapl)
        gtverr = torch.abs(intv - gtv)
        estverr = self.idv(errlogv)

        loss = torch.sum(torch.abs(gtherr - estherr) * indh) / torch.sum(indh) + \
               torch.sum(torch.abs(gtverr - estverr) * indv) / torch.sum(indv)

        loss = loss / 2

        return loss

    def depth2ang_log(self, depthMap, intrinsic):
        depthMaps = depthMap.squeeze(1)
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        p2dhx = (self.xx - bx) / fx * depthMaps
        p2dhy = (((self.yy - by) / fy) ** 2 + 1) * depthMaps

        p2dvx = (self.yy - by) / fy * depthMaps
        p2dvy = (((self.xx - bx) / fx) ** 2 + 1) * depthMaps

        angh = torch.atan2(self.diffx_sharp(p2dhy.unsqueeze(1)), self.diffx_sharp(p2dhx.unsqueeze(1)))
        angv = torch.atan2(self.diffy_sharp(p2dvy.unsqueeze(1)), self.diffy_sharp(p2dvx.unsqueeze(1)))

        ang = torch.cat([angh, angv], dim=1)

        # ang2 = self.depth2ang(depthMap, intrinsic, True)
        # tensor2disp(angh + np.pi, vmax=np.pi * 2, ind=0).show()
        # tensor2disp(ang2[:,0:1,:,:] + np.pi, vmax=np.pi * 2, ind=0).show()
        # tensor2disp(angv + np.pi, vmax=np.pi * 2, ind=0).show()
        # tensor2disp(ang2[:,1:2,:,:] + np.pi, vmax=np.pi * 2, ind=0).show()
        #
        # log = self.ang2log(intrinsic, ang)
        # logh = log[:, 0, :, :]
        # logv = log[:, 1, :, :]
        #
        # import random
        # ckx = random.randint(0, self.width)
        # cky = random.randint(0, self.height)
        # ckz = random.randint(0, self.batch_size - 1)
        #
        # ckhgtl = torch.log(depthMap[ckz, 0, cky, ckx + 1]) - torch.log(depthMap[ckz, 0, cky, ckx])
        # ckhestl = logh[ckz, cky, ckx]
        #
        # ckvgtl = torch.log(depthMap[ckz, 0, cky + 1, ckx]) - torch.log(depthMap[ckz, 0, cky, ckx])
        # ckvestl = logv[ckz, cky, ckx]
        return ang

    def colinearityloss(self, depthMap, w):
        gradxl = self.sharpxl(depthMap).squeeze(1)
        gradxr = self.sharpxr(depthMap).squeeze(1)
        gradyu = self.sharpyu(depthMap).squeeze(1)
        gradyd = self.sharpyd(depthMap).squeeze(1)

        colinearityloss = torch.sum((torch.abs(gradxl - gradxr) + torch.abs(gradyu - gradyd)) * w) / torch.sum(w)
        return colinearityloss

    def ang2grad(self, ang, intrinsic, depthMap):
        depthMaps = depthMap.squeeze(1)
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        angh = ang[:, 0, :, :]
        angv = ang[:, 1, :, :]

        depthMap_gradx_est = depthMaps / fx * torch.sin(angh) / ((((self.yy - by) / fy) ** 2 + 1) * torch.cos(angh) - (self.xx - bx) / fx * torch.sin(angh))
        depthMap_grady_est = depthMaps / fy * torch.sin(angv) / ((((self.xx - bx) / fx) ** 2 + 1) * torch.cos(angv) - (self.yy - by) / fy * torch.sin(angv))

        depthMap_gradx_est = depthMap_gradx_est.unsqueeze(1).clamp(min=-1e6, max=1e6)
        depthMap_grady_est = depthMap_grady_est.unsqueeze(1).clamp(min=-1e6, max=1e6)

        # Check
        # depthMap_gradx = self.diffx_sharp(depthMap)
        # depthMap_grady = self.diffy_sharp(depthMap)
        #
        # tensor2grad(depthMap_gradx_est, viewind=0, percentile=80).show()
        # tensor2grad(depthMap_gradx, viewind=0, percentile=80).show()
        #
        # tensor2grad(depthMap_grady_est, viewind=0, percentile=80).show()
        # tensor2grad(depthMap_grady, viewind=0, percentile=80).show()
        return depthMap_gradx_est, depthMap_grady_est

    def intergrationloss_ang(self, ang, intrinsic, depthMap):
        anglebound = 0.1
        protectmin = 1e-6
        vlossw = self.vlossw
        angw = self.angw
        sclw = self.sclw

        angh = ang[:, 0, :, :]
        angv = ang[:, 1, :, :]

        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        a1 = ((self.yy - by) / fy)**2 + 1
        b1 = -(self.xx - bx) / fx

        a2 = ((self.yy - by) / fy)**2 + 1
        b2 = -(self.xx + 1 - bx) / fx

        a3 = torch.sin(angh)
        b3 = -torch.cos(angh)

        u1 = ((self.xx - bx) / fx)**2 + 1
        v1 = -(self.yy - by) / fy

        u2 = ((self.xx - bx) / fx)**2 + 1
        v2 = -(self.yy + 1 - by) / fy

        u3 = torch.sin(angv)
        v3 = -torch.cos(angv)

        depthMapl = torch.log(torch.clamp(depthMap, min=protectmin))

        low_angh = torch.atan2(-a1, b1)
        high_angh = torch.atan2(a2, -b2)
        pred_angh = angh
        inboundh = ((pred_angh < (high_angh - anglebound)) * (pred_angh > (low_angh + anglebound))).float()

        low_angv = torch.atan2(-u1, v1)
        high_angv = torch.atan2(u2, -v2)
        pred_angv = angv
        inboundv = ((pred_angv < (high_angv - anglebound)) * (pred_angv > (low_angv + anglebound))).float()

        logh = torch.log(torch.clamp(torch.abs(a3 * b1 - a1 * b3), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3 * b2 - a2 * b3), min=protectmin))
        logh = torch.clamp(logh, min=-10, max=10)

        logv = torch.log(torch.clamp(torch.abs(u3 * v1 - u1 * v3), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3 * v2 - u2 * v3), min=protectmin))
        logv = torch.clamp(logv, min=-10, max=10)

        logh = logh.unsqueeze(1)
        logv = logv.unsqueeze(1)
        inboundh = inboundh.unsqueeze(1)
        inboundv = inboundv.unsqueeze(1)
        pred_angh = pred_angh.unsqueeze(1)
        pred_angv = pred_angv.unsqueeze(1)

        vallidarmask = (depthMap > 0).float()

        inth = self.inth(logh)
        gth = self.gth(depthMapl)
        indh = ((self.idh(vallidarmask) == 2) * (self.inth(1-inboundh) == 0)).float()
        # hloss = (torch.sum(torch.abs(gth - inth) * indh) + torch.sum(torch.abs(pred_angh) * (1 - inboundh) * vallidarmask) * angw) / (torch.sum(vallidarmask) + 1)
        hloss1 = torch.sum(torch.abs(gth - inth) * indh) / (torch.sum(indh) + 1)
        hloss2 = torch.sum(torch.abs(pred_angh) * (1 - inboundh)) * angw / (torch.sum(1 - inboundh) + 1)

        intv = self.intv(logv)
        gtv = self.gtv(depthMapl)
        indv = ((self.idv(vallidarmask) == 2) * (self.intv(1-inboundv) == 0)).float()
        # vloss = (torch.sum(torch.abs(gtv - intv) * indv) * vlossw + torch.sum(torch.abs(pred_angv) * (1 - inboundv) * vallidarmask) * angw) / (torch.sum(vallidarmask) + 1)
        vloss1 = torch.sum(torch.abs(gtv - intv) * indv) * vlossw / (torch.sum(indv) + 1)
        vloss2 = torch.sum(torch.abs(pred_angv) * (1 - inboundv)) * angw / (torch.sum(1 - inboundh) + 1)

        scl_pixelwise = self.selfconh(logh) + self.selfconv(logv)
        scl = torch.mean(torch.abs(scl_pixelwise))

        # loss = hloss + vloss + scl * sclw
        loss = hloss1 + hloss2 + vloss1 + vloss2 + scl * sclw

        # hloss1.backward()
        # torch.mean(torch.abs(ang.grad))
        #
        # hloss2.backward()
        # torch.mean(torch.abs(ang.grad))
        #
        # vloss1.backward()
        # torch.mean(torch.abs(ang.grad))
        #
        # vloss2.backward()
        # torch.mean(torch.abs(ang.grad))
        #
        # loss.backward()
        # torch.mean(torch.abs(ang.grad))
        #
        # import random
        # ckx = random.randint(0, self.width)
        # cky = random.randint(0, self.height)
        # ckz = random.randint(0, self.batch_size - 1)
        # ckc = random.randint(0, gtv.shape[1] - 1)
        #
        # rationum_imgagev = gtv[ckz, ckc, cky, ckx].detach().cpu().numpy()
        # logvck = intv[ckz, ckc, cky, ckx].detach().cpu().numpy()
        #
        # rationum_imgageh = gth[ckz, ckc, cky, ckx].detach().cpu().numpy()
        # loghck = inth[ckz, ckc, cky, ckx].detach().cpu().numpy()
        #
        # assert gth[indh == 1].min() > np.log(protectmin)
        # assert gtv[indv == 1].min() > np.log(protectmin)
        #
        # # Regression Experiment
        # import random
        # ckx = random.randint(0, self.width - 1)
        # cky = random.randint(0, self.height - 1)
        # ckz = random.randint(0, self.batch_size - 1)
        #
        # gthr = torch.log(depthMap[ckz, 0, cky, ckx + 1]) - torch.log(depthMap[ckz, 0, cky, ckx])
        # gtvr = torch.log(depthMap[ckz, 0, cky + 1, ckx]) - torch.log(depthMap[ckz, 0, cky, ckx])
        #
        # anggthr = angh[ckz, cky, ckx]
        # anggtvr = angv[ckz, cky, ckx]
        #
        # angrseed = torch.zeros([2], dtype=torch.float, device="cuda")
        # angrseed = angrseed
        # angrseed[0] = - torch.log(1 / (anggthr / 2 / np.pi + 0.5) - 1)
        # angrseed[1] = - torch.log(1 / (anggtvr / 2 / np.pi + 0.5) - 1)
        # angrseed.requires_grad = True
        # # angrseed.requires_grad = True
        #
        # expadam = torch.optim.SGD([angrseed], lr=1e0)
        # optnum = 1000
        # inboundw = 1/20
        #
        # a1r = a1[ckz, cky, ckx]
        # b1r = b1[ckz, cky, ckx]
        #
        # a2r = a2[ckz, cky, ckx]
        # b2r = b2[ckz, cky, ckx]
        #
        # u1r = u1[ckz, cky, ckx]
        # v1r = v1[ckz, cky, ckx]
        #
        # u2r = u2[ckz, cky, ckx]
        # v2r = v2[ckz, cky, ckx]
        #
        # angrec = list()
        # for i in range(optnum):
        #     angr = (torch.sigmoid(angrseed) - 0.5) * 2 * np.pi
        #     anghr = angr[0]
        #     angvr = angr[1]
        #
        #     a3r = torch.sin(anghr)
        #     b3r = -torch.cos(anghr)
        #
        #     u3r = torch.sin(angvr)
        #     v3r = -torch.cos(angvr)
        #
        #     loghr = torch.log(torch.clamp(torch.abs(a3r * b1r - a1r * b3r), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3r * b2r - a2r * b3r), min=protectmin))
        #     loghr = torch.clamp(loghr, min=-10, max=10)
        #
        #     logvr = torch.log(torch.abs(u2r)) - torch.log(torch.abs(u1r)) + torch.log(torch.clamp(torch.abs(u3r * v1r - u1r * v3r), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3r * v2r - u2r * v3r), min=protectmin))
        #     logvr = torch.clamp(logvr, min=-10, max=10)
        #
        #     low_anghr = torch.atan2(-a1r, b1r)
        #     high_anghr = torch.atan2(a2r, -b2r)
        #     pred_anghr = anghr
        #     inboundhr = ((pred_anghr < (high_anghr - anglebound)) * (pred_anghr > (low_anghr + anglebound))).float()
        #
        #     low_angvr = torch.atan2(-u1r, v1r)
        #     high_angvr = torch.atan2(u2r, -v2r)
        #     pred_angvr = angvr
        #     inboundvr = ((pred_angvr < (high_angvr - anglebound)) * (pred_angvr > (low_angvr + anglebound))).float()
        #
        #     loss = torch.abs(gthr - loghr) *inboundhr + torch.abs(gtvr - logvr) * inboundvr + \
        #            torch.abs(pred_anghr) * (1 - inboundhr) * inboundw + torch.abs(pred_angvr) * (1 - inboundvr) * inboundw
        #
        #     expadam.zero_grad()
        #     loss.backward()
        #     expadam.step()
        #
        #     angrec.append(angr.detach().cpu().numpy())
        #     print("Iteration: %d, Loss: %f" % (i, float(loss.detach().cpu().numpy())))
        #
        # import random
        # ckx = random.randint(0, self.width - 1)
        # cky = random.randint(0, self.height - 1)
        # ckz = random.randint(0, self.batch_size - 1)
        # anghit = torch.linspace(start=-np.pi, end=np.pi, steps=1000, device="cuda")
        # anghit.requires_grad = True
        #
        # gthrit = torch.log(depthMap[ckz, 0, cky, ckx + 1]) - torch.log(depthMap[ckz, 0, cky, ckx])
        # anggthrit = angh[ckz, cky, ckx]
        #
        # a1rit = a1[ckz, cky, ckx]
        # b1rit = b1[ckz, cky, ckx]
        #
        # a2rit = a2[ckz, cky, ckx]
        # b2rit = b2[ckz, cky, ckx]
        #
        # a3rit = torch.sin(anghit)
        # b3rit = -torch.cos(anghit)
        #
        # low_anghrit = torch.atan2(-a1rit, b1rit)
        # high_anghrit = torch.atan2(a2rit, -b2rit)
        # inboundhrit = ((anghit < (high_anghrit) - anglebound) * (anghit > (low_anghrit) + anglebound)).float()
        #
        # loghrit = torch.log(torch.clamp(torch.abs(a3rit * b1rit - a1rit * b3rit), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3rit * b2rit - a2rit * b3rit), min=protectmin))
        # loghrit = torch.clamp(loghrit, min=-10, max=10)
        #
        # lossit = torch.abs(loghrit) * inboundhrit + torch.abs(anghit) * (1 - inboundhrit) * angw
        # lossit_sum = torch.sum(lossit)
        #
        # ckhind = torch.argmin(torch.abs(loghrit - gthrit) * inboundhrit + torch.abs(anghit) * (1 - inboundhrit))
        # print("True angle: %f, Most cloest angle: %f" % (float(anggthrit.detach().cpu().numpy()), float(anghit[ckhind].detach().cpu().numpy())))
        #
        # lossit_sum.backward()
        # inboundsel = inboundhrit.detach().cpu().numpy() == 1
        #
        # plt.figure()
        # plt.stem(anghit.detach().cpu().numpy()[inboundsel], lossit.detach().cpu().numpy()[inboundsel])
        #
        # plt.figure()
        # plt.stem(anghit.detach().cpu().numpy(), np.abs(anghit.grad.detach().cpu().numpy()))
        #
        # plt.figure()
        # plt.stem(anghit.detach().cpu().numpy()[inboundsel], np.abs(anghit.grad.detach().cpu().numpy())[inboundsel])
        #
        # angvit = torch.linspace(start=-np.pi, end=np.pi, steps=1000, device="cuda")
        # angvit.requires_grad = True
        #
        # gtvrit = torch.log(depthMap[ckz, 0, cky + 1, ckx]) - torch.log(depthMap[ckz, 0, cky, ckx])
        # anggtvrit = angv[ckz, cky, ckx]
        #
        # u1rit = u1[ckz, cky, ckx]
        # v1rit = v1[ckz, cky, ckx]
        #
        # u2rit = u2[ckz, cky, ckx]
        # v2rit = v2[ckz, cky, ckx]
        #
        # u3rit = torch.sin(angvit)
        # v3rit = -torch.cos(angvit)
        #
        # low_angvrit = torch.atan2(-u1rit, v1rit)
        # high_angvrit = torch.atan2(u2rit, -v2rit)
        # inboundvrit = ((angvit < (high_angvrit) - anglebound) * (angvit > (low_angvrit) + anglebound)).float()
        #
        # logvrit = torch.log(torch.clamp(torch.abs(u3rit * v1rit - u1rit * v3rit), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3rit * v2rit - u2rit * v3rit), min=protectmin))
        # logvrit = torch.clamp(logvrit, min=-10, max=10)
        #
        # lossit = torch.abs(logvrit) * inboundvrit + torch.abs(angvit) * (1 - inboundvrit) * angw
        # lossit_sum = torch.sum(lossit)
        #
        # ckvind = torch.argmin(torch.abs(logvrit - gtvrit) * inboundvrit + torch.abs(angvit) * (1 - inboundvrit))
        # print("True angle: %f, Most cloest angle: %f" % (float(anggtvrit.detach().cpu().numpy()), float(angvit[ckvind].detach().cpu().numpy())))
        #
        # lossit_sum.backward()
        # inboundsel = inboundvrit.detach().cpu().numpy() == 1
        #
        # plt.figure()
        # plt.stem(angvit.detach().cpu().numpy()[inboundsel], lossit.detach().cpu().numpy()[inboundsel])
        #
        # plt.figure()
        # plt.stem(angvit.detach().cpu().numpy(), np.abs(angvit.grad.detach().cpu().numpy()))
        #
        # plt.figure()
        # plt.stem(angvit.detach().cpu().numpy()[inboundsel], np.abs(angvit.grad.detach().cpu().numpy())[inboundsel])
        return loss, hloss1, hloss2, vloss1, vloss2, torch.sum((1 - inboundh)), torch.sum((1 - inboundv))

    def intergrationloss_ang_validation(self, ang, intrinsic, depthMap):
        protectmin = 1e-6

        angh = ang[:, 0, :, :]
        angv = ang[:, 1, :, :]

        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        a1 = ((self.yy - by) / fy)**2 + 1
        b1 = -(self.xx - bx) / fx

        a2 = ((self.yy - by) / fy)**2 + 1
        b2 = -(self.xx + 1 - bx) / fx

        a3 = torch.sin(angh)
        b3 = -torch.cos(angh)

        u1 = ((self.xx - bx) / fx)**2 + 1
        v1 = -(self.yy - by) / fy

        u2 = ((self.xx - bx) / fx)**2 + 1
        v2 = -(self.yy + 1 - by) / fy

        u3 = torch.sin(angv)
        v3 = -torch.cos(angv)

        depthMapl = torch.log(torch.clamp(depthMap, min=protectmin))

        logh = torch.log(torch.clamp(torch.abs(a3 * b1 - a1 * b3), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3 * b2 - a2 * b3), min=protectmin))
        logh = torch.clamp(logh, min=-10, max=10)

        logv = torch.log(torch.clamp(torch.abs(u3 * v1 - u1 * v3), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3 * v2 - u2 * v3), min=protectmin))
        logv = torch.clamp(logv, min=-10, max=10)

        logh = logh.unsqueeze(1)
        logv = logv.unsqueeze(1)

        vallidarmask = (depthMap > 0).float()

        inth = self.inth(logh)
        gth = self.gth(depthMapl)
        indh = (self.idh(vallidarmask) == 2).float()
        hloss = torch.sum(torch.abs(gth - inth) * indh) / (torch.sum(indh) + 1)

        intv = self.intv(logv)
        gtv = self.gtv(depthMapl)
        indv = (self.idv(vallidarmask) == 2).float()
        vloss = torch.sum(torch.abs(gtv - intv) * indv) / (torch.sum(indv) + 1)

        loss = hloss + vloss

        return loss

    def ang2normal(self, ang, intrinsic):
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        angh = ang[:, 0, :, :]
        angv = ang[:, 1, :, :]

        normx = torch.stack([torch.cos(angh), (self.yy - by) / fy * torch.sin(angh), torch.sin(angh)], dim=1)
        normy = torch.stack([(self.xx - bx) / fx * torch.sin(angv), torch.cos(angv), torch.sin(angv)], dim=1)

        surfacenormal = torch.cross(normx, normy, dim=1)
        surfacenormal = F.normalize(surfacenormal, dim=1)
        surfacenormal = torch.clamp(surfacenormal, min=-1+1e-6, max=1-1e-6)

        # tensor2rgb((surfacenormal + 1) / 2, ind=0).show()
        # tensor2rgb((self.depth2norm(depthMap, intrinsic) + 1) / 2, ind=0).show()

        return surfacenormal

    def ang2dirs(self, ang, intrinsic):
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        angh = ang[:, 0, :, :]
        angv = ang[:, 1, :, :]

        normx = torch.stack([torch.cos(angh), (self.yy - by) / fy * torch.sin(angh), torch.sin(angh)], dim=1)
        normy = torch.stack([(self.xx - bx) / fx * torch.sin(angv), torch.cos(angv), torch.sin(angv)], dim=1)

        normx = F.normalize(normx, dim=1)
        normy = F.normalize(normy, dim=1)

        return normx, normy

    def normal2ang(self, surfnorm, intrinsic):
        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        surfnormx = torch.stack([surfnorm[:, 1, :, :] * (self.yy - by) / fy + surfnorm[:, 2, :, :], -(self.yy - by) / fy * surfnorm[:, 0, :, :], -surfnorm[:, 0, :, :]], dim=1)
        surfnormy = torch.stack([-surfnorm[:, 1, :, :] * (self.xx - bx) / fx, (self.xx - bx) / fx * surfnorm[:, 0, :, :] + surfnorm[:, 2, :, :], -surfnorm[:, 1, :, :]], dim=1)

        a3 = (self.yy - by) / fy * surfnormx[:, 1, :, :] + surfnormx[:, 2, :, :]
        b3 = -surfnormx[:, 0, :, :]

        u3 = surfnormy[:, 2, :, :] + (self.xx - bx) / fx * surfnormy[:, 0, :, :]
        v3 = -surfnormy[:, 1, :, :]

        pred_angh = torch.atan2(a3, -b3).unsqueeze(1)
        pred_angv = torch.atan2(u3, -v3).unsqueeze(1)

        predang = torch.cat([pred_angh, pred_angv], dim=1)

        return predang

    def intergrationloss_surfacenormal(self, surfnorm, intrinsic, depthMap):
        anglebound = 0
        protectmin = 1e-6
        angw = 1e0
        sclw = 1e-1

        fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        bx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])
        by = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(2).expand([-1, self.height, self.width])

        surfnormx = torch.stack([surfnorm[:, 1, :, :] * (self.yy -by) / fy + surfnorm[:, 2, :, :], -(self.yy -by) / fy * surfnorm[:, 0, :, :], -surfnorm[:, 0, :, :]], dim=1)
        surfnormy = torch.stack([-surfnorm[:, 1, :, :] * (self.xx -bx) / fx, (self.xx -bx) / fx * surfnorm[:, 0, :, :] + surfnorm[:, 2, :, :], -surfnorm[:, 1, :, :]], dim=1)

        a1 = ((self.yy - by) / fy)**2 + 1
        b1 = -(self.xx - bx) / fx

        a2 = ((self.yy - by) / fy)**2 + 1
        b2 = -(self.xx + 1 - bx) / fx

        a3 = (self.yy - by) / fy * surfnormx[:, 1, :, :] + surfnormx[:, 2, :, :]
        b3 = -surfnormx[:, 0, :, :]

        u1 = ((self.xx - bx) / fx)**2 + 1
        v1 = -(self.yy - by) / fy

        u2 = ((self.xx - bx) / fx)**2 + 1
        v2 = -(self.yy + 1 - by) / fy

        u3 = surfnormy[:, 2, :, :] + (self.xx - bx) / fx * surfnormy[:, 0, :, :]
        v3 = -surfnormy[:, 1, :, :]

        logh = torch.log(torch.clamp(torch.abs(a3 * b1 - a1 * b3), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3 * b2 - a2 * b3), min=protectmin))
        logh = torch.clamp(logh, min=-10, max=10)

        logv = torch.log(torch.abs(u2)) - torch.log(torch.abs(u1)) + torch.log(torch.clamp(torch.abs(u3 * v1 - u1 * v3), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3 * v2 - u2 * v3), min=protectmin))
        logv = torch.clamp(logv, min=-10, max=10)

        depthMapl = torch.log(depthMap)

        low_angh = torch.atan2(-a1, b1)
        high_angh = torch.atan2(a2, -b2)
        pred_angh = torch.atan2(a3, -b3)
        inboundh = ((pred_angh < (high_angh - anglebound)) * (pred_angh > (low_angh + anglebound))).float()

        low_angv = torch.atan2(-u1, v1)
        high_angv = torch.atan2(u2, -v2)
        pred_angv = torch.atan2(u3, -v3)
        inboundv = ((pred_angv < (high_angv - anglebound)) * (pred_angv > (low_angv + anglebound))).float()

        # # ====Temporarily Suppressed Loss Related Code==== #
        # gth = self.diffx_sharp(depthMapl)
        # gtv = self.diffy_sharp(depthMapl)
        #
        # logh = logh.unsqueeze(1)
        # logv = logv.unsqueeze(1)
        # inboundh = inboundh.unsqueeze(1)
        # inboundv = inboundv.unsqueeze(1)
        #
        # lossh = logh
        # lossv = logv
        #
        # scl_pixelwise = self.selfconh(logh) + self.selfconv(logv)
        # scl = torch.mean(torch.abs(scl_pixelwise))
        #
        # loss = torch.mean(torch.abs(lossh - gth) * inboundh) + \
        #        torch.mean(torch.abs(lossv - gtv) * inboundv) + \
        #        torch.sum(torch.abs((low_angh + high_angh) / 2 - pred_angh) * (1 - inboundh)) / self.height / self.width / self.batch_size * angw + \
        #        torch.sum(torch.abs((low_angv + high_angv) / 2 - pred_angv) * (1 - inboundv)) / self.height / self.width / self.batch_size * angw + \
        #        scl * sclw
        # outrangeval = torch.sum((1 - inboundh)) + torch.sum((1 - inboundv))

        # # ====Validation part code==== #
        #
        # Visualization
        # tensor2disp((1 - inboundh).unsqueeze(1), vmax=1, ind=0).show()
        # tensor2disp((1 - inboundv).unsqueeze(1), vmax=1, ind=0).show()
        #
        # # Validation for range
        # import random
        # ckx = random.randint(0, self.width)
        # cky = random.randint(0, self.height)
        # ckz = random.randint(0, self.batch_size - 1)
        #
        # lowang_ck = low_angh[ckz, cky, ckx].detach().cpu().numpy()
        # highang_ck = high_angh[ckz, cky, ckx].detach().cpu().numpy()
        # predang_ck = pred_angh[ckz, cky, ckx].detach().cpu().numpy()
        #
        # intrinsicnpck = intrinsic[ckz, :, :].cpu().numpy()
        # cknum = 10000
        # d1 = np.linspace(1, 100, cknum)
        # d2 = np.flip(np.linspace(1, 100, cknum))
        #
        # pts1 = np.stack([np.ones(cknum) * ckx * d1, np.ones(cknum) * cky * d1, d1, np.ones(cknum)], axis=0)
        # pts1 = (np.linalg.inv(intrinsicnpck) @ pts1).T
        # pts1 = pts1[:, 0:3]
        # pts2 = np.stack([np.ones(cknum) * (ckx + 1) * d2, np.ones(cknum) * cky * d2, d2, np.ones(cknum)], axis=0)
        # pts2 = (np.linalg.inv(intrinsicnpck) @ pts2).T
        # pts2 = pts2[:, 0:3]
        #
        # d1gt = float(depthMap[ckz, 0, cky, ckx].detach().cpu().numpy())
        # d2gt = float(depthMap[ckz, 0, cky, ckx + 1].detach().cpu().numpy())
        # ptsgt1 = np.array([[ckx * d1gt, cky * d1gt, d1gt, 1]]).T
        # ptsgt2 = np.array([[(ckx + 1) * d2gt, cky * d2gt, d2gt, 1]]).T
        # ptsgt1 = (np.linalg.inv(intrinsicnpck) @ ptsgt1).T
        # ptsgt2 = (np.linalg.inv(intrinsicnpck) @ ptsgt2).T
        # ptsgt1 = ptsgt1[:, 0:3]
        # ptsgt2 = ptsgt2[:, 0:3]
        #
        # xax = np.array([1, 0, 0])
        # yax = np.array([0, (cky - by[ckz, cky, ckx].detach().cpu().numpy()) / fy[ckz, cky, ckx].detach().cpu().numpy(), 1])
        #
        # pts1x = pts1 @ np.expand_dims(xax, axis=1)
        # pts1y = pts1 @ np.expand_dims(yax, axis=1)
        #
        # pts2x = pts2 @ np.expand_dims(xax, axis=1)
        # pts2y = pts2 @ np.expand_dims(yax, axis=1)
        #
        # ptsgt1x = ptsgt1 @ np.expand_dims(xax, axis=1)
        # ptsgt1y = ptsgt1 @ np.expand_dims(yax, axis=1)
        #
        # ptsgt2x = ptsgt2 @ np.expand_dims(xax, axis=1)
        # ptsgt2y = ptsgt2 @ np.expand_dims(yax, axis=1)
        #
        # ckang = np.arctan2(pts2y - pts1y, pts2x - pts1x)
        # ckgt = np.arctan2(ptsgt2y - ptsgt1y, ptsgt2x - ptsgt1x)
        #
        # loggtnum = np.log(float((depthMap[ckz, 0, cky, ckx + 1] / depthMap[ckz, 0, cky, ckx]).detach().cpu().numpy()))
        # loggththeo = np.log(ptsgt2y / ptsgt1y)
        # logsampled = np.log(pts2y / pts1y)
        # logpred = float(logh[ckz, 0, cky, ckx].detach().cpu().numpy())
        #
        # fig, axs = plt.subplots()
        # axs.scatter(np.cos(ckang), np.sin(ckang))
        # axs.scatter(np.cos(lowang_ck), np.sin(lowang_ck), c='r')
        # axs.scatter(np.cos(highang_ck), np.sin(highang_ck), c='r')
        # axs.scatter(np.cos(predang_ck), np.sin(predang_ck), c='g')
        # axs.scatter(np.cos(ckgt), np.sin(ckgt), c='k')
        # axs.set_aspect('equal')
        #
        # fig, axs = plt.subplots()
        # axs.scatter(ckang, logsampled)
        # axs.scatter(ckgt, loggththeo, c='k')
        # axs.scatter(predang_ck, logpred, c='r')
        # axs.set_aspect('equal')
        #
        # Validation for log ratio
        # depthMaps = depthMap.squeeze(1)
        # depthMap_gradx = self.diffx_sharp(depthMap).squeeze(1)
        # depthMap_grady = self.diffy_sharp(depthMap).squeeze(1)
        #
        # vx1 = depthMaps / fx + (self.xx - bx) / fx * depthMap_gradx
        # vx2 = (self.yy - by) / fy * depthMap_gradx
        # vx3 = depthMap_gradx
        #
        # vy1 = (self.xx - bx) / fx * depthMap_grady
        # vy2 = depthMaps / fy + (self.yy - by) / fy * depthMap_grady
        # vy3 = depthMap_grady
        #
        # vx = torch.stack([vx1, vx2, vx3], dim=1)
        # vy = torch.stack([vy1, vy2, vy3], dim=1)
        #
        # vxnormed = F.normalize(vx, dim=1)
        # vynormed = F.normalize(vy, dim=1)
        # surfnormxnormed = F.normalize(surfnormx, dim=1)
        # surfnormynormed = F.normalize(surfnormy, dim=1)
        #
        # diffvx = torch.abs(surfnormxnormed - vxnormed)
        # diffvy = torch.abs(surfnormynormed - vynormed)
        #
        # # X direction
        # import random
        # ckx = random.randint(0, self.width)
        # cky = random.randint(0, self.height)
        # ckz = random.randint(0, self.batch_size - 1)
        #
        # xax = np.array([1, 0, 0])
        # yax = np.array([0, (cky - by[ckz, cky, ckx].detach().cpu().numpy()) / fy[ckz, cky, ckx].detach().cpu().numpy(), 1])
        # intrinsicnpck = intrinsic[ckz, :, :].cpu().numpy()
        #
        # ptsckdnp1 = depthMap[ckz, 0, cky, ckx].detach().cpu().numpy()
        # ptscknp1 = np.linalg.inv(intrinsicnpck) @ np.array([[ckx * ptsckdnp1, cky * ptsckdnp1, ptsckdnp1, 1]]).T
        #
        # ptsckdnp1t = depthMap[ckz, 0, cky, ckx].detach().cpu().numpy() + np.random.random(1) * 100
        # ptscknp1t = np.linalg.inv(intrinsicnpck) @ np.array([[ckx * ptsckdnp1t, cky * ptsckdnp1t, ptsckdnp1t, 1]]).T
        #
        # ptsckdnp2 = depthMap[ckz, 0, cky, ckx + 1].detach().cpu().numpy()
        # ptscknp2 = np.linalg.inv(intrinsicnpck) @ np.array([[(ckx + 1) * ptsckdnp2, cky * ptsckdnp2, ptsckdnp2, 1]]).T
        #
        # ptsckdnp2t = depthMap[ckz, 0, cky, ckx + 1].detach().cpu().numpy() + np.random.random(1) * 100
        # ptscknp2t = np.linalg.inv(intrinsicnpck) @ np.array([[(ckx + 1) * ptsckdnp2t, cky * ptsckdnp2t, ptsckdnp2t, 1]]).T
        #
        # assert np.sum((ptscknp2 - ptscknp1)[0:3, 0] * np.cross(xax, yax)) < 1e-3, print("colinear check failed") # should be zero
        #
        # ptscknp1p = np.array([np.sum(ptscknp1[0:3, 0] * xax), np.sum(ptscknp1[0:3, 0] * yax)])
        # ptscknp1tp = np.array([float(np.sum(ptscknp1t[0:3, 0] * xax)), float(np.sum(ptscknp1t[0:3, 0] * yax))])
        # a1cknp = a1[ckz, cky, ckx].detach().cpu().numpy()
        # b1cknp = b1[ckz, cky, ckx].detach().cpu().numpy()
        # l1param = np.array([a1cknp, b1cknp])
        # assert np.sum(l1param * (ptscknp1tp - ptscknp1p)) < 1e-3, print("line1 param check failed") # should be zero
        #
        # ptscknp2p = np.array([np.sum(ptscknp2[0:3, 0] * xax), np.sum(ptscknp2[0:3, 0] * yax)])
        # ptscknp2tp = np.array([float(np.sum(ptscknp2t[0:3, 0] * xax)), float(np.sum(ptscknp2t[0:3, 0] * yax))])
        # a2cknp = a2[ckz, cky, ckx].detach().cpu().numpy()
        # b2cknp = b2[ckz, cky, ckx].detach().cpu().numpy()
        # l2param = np.array([a2cknp, b2cknp])
        # assert np.sum(l2param * (ptscknp2tp - ptscknp2p)) < 1e-3, print("line1 param check failed") # should be zero
        #
        # ptscknp3 = ptscknp1
        # incre = surfnormx[ckz, :, cky, ckx].detach().cpu().numpy() * np.random.random(1) * 100
        # incre = np.expand_dims(np.concatenate([incre, np.array([0])]), axis=1)
        # ptscknp3t = ptscknp3 + incre
        # ptscknp3p = np.array([np.sum(ptscknp3[0:3, 0] * xax), np.sum(ptscknp3[0:3, 0] * yax)])
        # ptscknp3tp = np.array([float(np.sum(ptscknp3t[0:3, 0] * xax)), float(np.sum(ptscknp3t[0:3, 0] * yax))])
        # a3cknp = a3[ckz, cky, ckx].detach().cpu().numpy()
        # b3cknp = b3[ckz, cky, ckx].detach().cpu().numpy()
        # l3param = np.array([a3cknp, b3cknp])
        # assert np.sum(l3param * (ptscknp3tp - ptscknp3p)) < 1e-3, print("line1 param check failed")  # should be zero
        #
        # samplenum = 100000
        # inct = np.linspace(-1, 1, samplenum)
        # surfnormck = surfnormx[ckz, :, cky, ckx].detach().cpu().numpy()
        # assert np.sum(surfnormck * np.cross(xax, yax)) < 1e-3, print("surfnormx colplane check failed")  # should be zero
        #
        # incpts = np.stack([surfnormck[0] * inct + ptscknp3[0], surfnormck[1] * inct + ptscknp3[1], surfnormck[2] * inct + ptscknp3[2], np.ones([samplenum])], axis=0)
        # incpts_projected = (intrinsicnpck @ incpts).T
        # incpts_projected[:, 0] = incpts_projected[:, 0] / incpts_projected[:, 2]
        # incpts_projected[:, 1] = incpts_projected[:, 1] / incpts_projected[:, 2]
        # incpts_projected = incpts_projected[:, 0:3]
        # indnearest = np.argmin(np.abs((incpts_projected[:, 0] - ckx - 1)))
        # depthnearest = incpts_projected[indnearest, 2]
        #
        # ratio_theo_ab = np.log(np.abs(a2cknp)) - np.log(np.abs(a1cknp)) + np.log(np.abs((a3cknp * b1cknp - a1cknp * b3cknp))) - np.log(np.abs((a3cknp * b2cknp - a2cknp * b3cknp)))
        # rationum = np.log(depthnearest) - np.log(ptsckdnp1)
        # rationum_imgage = (np.log(depthMap[ckz, 0, cky, ckx + 1].detach().cpu().numpy()) - np.log(depthMap[ckz, 0, cky, ckx].detach().cpu().numpy()))
        # loghck = logh[ckz, cky, ckx].detach().cpu().numpy()
        #
        # Y Direction
        # import random
        # ckx = random.randint(0, self.width)
        # cky = random.randint(0, self.height)
        # ckz = random.randint(0, self.batch_size - 1)
        #
        # xay = np.array([0, 1, 0])
        # yay = np.array([(ckx - bx[ckz, cky, ckx].detach().cpu().numpy()) / fx[ckz, cky, ckx].detach().cpu().numpy(), 0, 1])
        # intrinsicnpck = intrinsic[ckz, :, :].cpu().numpy()
        #
        # ptsckdnp1 = depthMap[ckz, 0, cky, ckx].detach().cpu().numpy()
        # ptscknp1 = np.linalg.inv(intrinsicnpck) @ np.array([[ckx * ptsckdnp1, cky * ptsckdnp1, ptsckdnp1, 1]]).T
        #
        # ptsckdnp1t = depthMap[ckz, 0, cky, ckx].detach().cpu().numpy() + np.random.random(1) * 100
        # ptscknp1t = np.linalg.inv(intrinsicnpck) @ np.array([[ckx * ptsckdnp1t, cky * ptsckdnp1t, ptsckdnp1t, 1]]).T
        #
        # ptsckdnp2 = depthMap[ckz, 0, cky + 1, ckx].detach().cpu().numpy()
        # ptscknp2 = np.linalg.inv(intrinsicnpck) @ np.array([[ckx * ptsckdnp2, (cky + 1) * ptsckdnp2, ptsckdnp2, 1]]).T
        #
        # ptsckdnp2t = depthMap[ckz, 0, cky, ckx + 1].detach().cpu().numpy() + np.random.random(1) * 100
        # ptscknp2t = np.linalg.inv(intrinsicnpck) @ np.array([[ckx * ptsckdnp2t, (cky + 1) * ptsckdnp2t, ptsckdnp2t, 1]]).T
        #
        # assert np.sum((ptscknp2 - ptscknp1)[0:3, 0] * np.cross(xay, yay)) < 1e-3, print("colinear check failed") # should be zero
        #
        # ptscknp1p = np.array([np.sum(ptscknp1[0:3, 0] * xay), np.sum(ptscknp1[0:3, 0] * yay)])
        # ptscknp1tp = np.array([float(np.sum(ptscknp1t[0:3, 0] * xay)), float(np.sum(ptscknp1t[0:3, 0] * yay))])
        # a1cknp = u1[ckz, cky, ckx].detach().cpu().numpy()
        # b1cknp = v1[ckz, cky, ckx].detach().cpu().numpy()
        # l1param = np.array([a1cknp, b1cknp])
        # assert np.sum(l1param * (ptscknp1tp - ptscknp1p)) < 1e-3, print("line1 param check failed") # should be zero
        #
        # ptscknp2p = np.array([np.sum(ptscknp2[0:3, 0] * xay), np.sum(ptscknp2[0:3, 0] * yay)])
        # ptscknp2tp = np.array([float(np.sum(ptscknp2t[0:3, 0] * xay)), float(np.sum(ptscknp2t[0:3, 0] * yay))])
        # a2cknp = u2[ckz, cky, ckx].detach().cpu().numpy()
        # b2cknp = v2[ckz, cky, ckx].detach().cpu().numpy()
        # l2param = np.array([a2cknp, b2cknp])
        # assert np.sum(l2param * (ptscknp2tp - ptscknp2p)) < 1e-3, print("line1 param check failed") # should be zero
        #
        # surfnormck = surfnormy[ckz, :, cky, ckx].detach().cpu().numpy()
        # ptscknp3 = ptscknp1
        # incre = surfnormy[ckz, :, cky, ckx].detach().cpu().numpy() * np.random.random(1) * 100
        # incre = np.expand_dims(np.concatenate([incre, np.array([0])]), axis=1)
        # ptscknp3t = ptscknp3 + incre
        # ptscknp3p = np.array([np.sum(ptscknp3[0:3, 0] * xay), np.sum(ptscknp3[0:3, 0] * yay)])
        # ptscknp3tp = np.array([float(np.sum(ptscknp3t[0:3, 0] * xay)), float(np.sum(ptscknp3t[0:3, 0] * yay))])
        # a3cknp = u3[ckz, cky, ckx].detach().cpu().numpy()
        # b3cknp = v3[ckz, cky, ckx].detach().cpu().numpy()
        # l3param = np.array([a3cknp, b3cknp])
        # c3cknp = -(a3cknp * ptscknp3p[0] + b3cknp * ptscknp3p[1])
        # assert np.sum(l3param * (ptscknp3tp - ptscknp3p)) < 1e-3, print("line1 param check failed")  # should be zero
        # assert a3cknp * ptscknp3tp[0] + b3cknp * ptscknp3tp[1] + c3cknp < 1e-3
        # assert np.abs((ptscknp2p[0] - ptscknp1p[0]) * l3param[0] + (ptscknp2p[1] - ptscknp1p[1]) * l3param[1]) < 1e-4
        # assert np.abs(np.sum(surfnormck * xay) * a3cknp + np.sum(surfnormck * yay) * b3cknp) < 1e-4
        #
        # dir1 = np.array([np.sum((ptscknp2[0:3, 0] - ptscknp1[0:3, 0]) * xay), np.sum((ptscknp2[0:3, 0] - ptscknp1[0:3, 0]) * yay)])
        # dir1 = dir1 / np.sqrt(np.sum(dir1 ** 2))
        # dir2 = np.array([np.sum(ptscknp2[0:3, 0] * xay) - np.sum(ptscknp1[0:3, 0] * xay), np.sum(ptscknp2[0:3, 0] * yay) - np.sum(ptscknp1[0:3, 0] * yay)])
        # dir2 = dir2 / np.sqrt(np.sum(dir2 ** 2))
        # dir3 = np.array([np.sum(surfnormck * xay), np.sum(surfnormck * yay)])
        # dir3 = dir3 / np.sqrt(np.sum(dir3 ** 2))
        # l3paramnormed = l3param / np.sqrt(np.sum(l3param ** 2))
        # mult1 = np.abs(np.sum(l3paramnormed * dir3))
        # mult2 = np.abs(np.sum(l3paramnormed * dir2))
        #
        # intercept2x = -(b2cknp * c3cknp) / (a3cknp * b2cknp - a2cknp * b3cknp)
        # intercept2y = (a2cknp * c3cknp) / (a3cknp * b2cknp - a2cknp * b3cknp)
        # intercept1x = -(b1cknp * c3cknp) / (a3cknp * b1cknp - a1cknp * b3cknp)
        # intercept1y = (a1cknp * c3cknp) / (a3cknp * b1cknp - a1cknp * b3cknp)
        # assert np.abs(intercept1x * a3cknp + intercept1y * b3cknp + c3cknp) < 1e-4
        # assert np.abs(intercept2x * a3cknp + intercept2y * b3cknp + c3cknp) < 1e-4
        # assert np.abs(intercept1x * a1cknp + intercept1y * b1cknp) < 1e-4
        # assert np.abs(intercept2x * a2cknp + intercept2y * b2cknp) < 1e-4
        # assert np.abs(ptscknp2p[0] * a3cknp + ptscknp2p[1] * b3cknp + c3cknp) < 1e-4
        # assert np.abs(intercept2x - ptscknp2p[0]) + np.abs(intercept2y - ptscknp2p[1]) < 1e-2
        # assert np.abs(intercept1x - ptscknp1p[0]) + np.abs(intercept1y - ptscknp1p[1]) < 1e-3
        #
        # samplenum = 100000
        # inct = np.linspace(-1, 1, samplenum)
        # assert np.sum(surfnormck * np.cross(xay, yay)) < 1e-3, print("surfnormx colplane check failed")  # should be zero
        # assert 1 - (np.sum(surfnormck * [ptscknp2[0:3, 0] - ptscknp1[0:3, 0]]) / np.sqrt(np.sum(surfnormck**2)) / np.sqrt(np.sum((ptscknp1[0:3, 0] - ptscknp2[0:3, 0])**2))) < 1e-3
        #
        # incpts = np.stack([surfnormck[0] * inct + ptscknp3[0], surfnormck[1] * inct + ptscknp3[1], surfnormck[2] * inct + ptscknp3[2], np.ones([samplenum])], axis=0)
        # incpts_projected = (intrinsicnpck @ incpts).T
        # incpts_projected[:, 0] = incpts_projected[:, 0] / incpts_projected[:, 2]
        # incpts_projected[:, 1] = incpts_projected[:, 1] / incpts_projected[:, 2]
        # incpts_projected = incpts_projected[:, 0:3]
        # indnearest = np.argmin(np.abs((incpts_projected[:, 1] - cky - 1)))
        # depthnearest = incpts_projected[indnearest, 2]
        #
        # rationum_imgage = (np.log(depthMap[ckz, 0, cky + 1, ckx].detach().cpu().numpy()) - np.log(depthMap[ckz, 0, cky, ckx].detach().cpu().numpy()))
        # rationump = np.log(ptscknp2p[1]) - np.log(ptscknp1p[1])
        # rationum = np.log(depthnearest) - np.log(ptsckdnp1)
        # logvnp = np.log((a2cknp * c3cknp) / (a3cknp * b2cknp - a2cknp * b3cknp)) - np.log((a1cknp * c3cknp) / (a3cknp * b1cknp - a1cknp * b3cknp))
        # logvck = logv[ckz, cky, ckx].detach().cpu().numpy()
        #
        # # Regression Experiment
        # import random
        # ckx = random.randint(0, self.width - 1)
        # cky = random.randint(0, self.height - 1)
        # ckz = random.randint(0, self.batch_size - 1)
        # fxr = intrinsic[ckz, 0, 0].detach()
        # bxr = intrinsic[ckz, 0, 2].detach()
        # fyr = intrinsic[ckz, 1, 1].detach()
        # byr = intrinsic[ckz, 1, 2].detach()
        #
        # gthr = torch.log(depthMap[ckz, 0, cky, ckx + 1]) - torch.log(depthMap[ckz, 0, cky, ckx])
        # gtvr = torch.log(depthMap[ckz, 0, cky + 1, ckx]) - torch.log(depthMap[ckz, 0, cky, ckx])
        #
        # surfnormrseed = torch.rand([3], dtype=torch.float, device="cuda")
        # surfnormrseed = surfnormrseed - 10
        # surfnormrseed.requires_grad = True
        #
        # expadam = torch.optim.Adam([surfnormrseed], lr=1e-1)
        # optnum = 10000
        # inboundw = 10000
        # surfnormgt = surfnorm[ckz, :, cky, ckx]
        #
        # normrec = list()
        # for i in range(optnum):
        #     surfnormr = (torch.sigmoid(surfnormrseed) - 0.5) * 2
        #     surfnormr = surfnormr / torch.sqrt(torch.sum(surfnormr ** 2))
        #
        #     surfnormxr = torch.stack([surfnormr[1] * (cky -byr) / fyr + surfnormr[2], -(cky -byr) / fyr * surfnormr[0], -surfnormr[0]])
        #     surfnormyr = torch.stack([-surfnormr[1] * (ckx -bxr) / fxr, (ckx -bxr) / fxr * surfnormr[0] + surfnormr[2], -surfnormr[1]])
        #
        #     a1r = ((cky - byr) / fyr)**2 + 1
        #     b1r = -(ckx - bxr) / fxr
        #
        #     a2r = ((cky - byr) / fyr)**2 + 1
        #     b2r = -(ckx + 1 - bxr) / fxr
        #
        #     a3r = (cky - byr) / fyr * surfnormxr[1] + surfnormxr[2]
        #     b3r = -surfnormxr[0]
        #
        #     u1r = ((ckx - bxr) / fxr)**2 + 1
        #     v1r = -(cky - byr) / fyr
        #
        #     u2r = ((ckx - bxr) / fxr)**2 + 1
        #     v2r = -(cky + 1 - byr) / fyr
        #
        #     u3r = surfnormyr[2] + (ckx - bxr) / fxr * surfnormyr[0]
        #     v3r = -surfnormyr[1]
        #
        #     loghr = torch.log(torch.clamp(torch.abs(a3r * b1r - a1r * b3r), min=protectmin)) - torch.log(torch.clamp(torch.abs(a3r * b2r - a2r * b3r), min=protectmin))
        #     loghr = torch.clamp(loghr, min=-10, max=10)
        #
        #     logvr = torch.log(torch.abs(u2r)) - torch.log(torch.abs(u1r)) + torch.log(torch.clamp(torch.abs(u3r * v1r - u1r * v3r), min=protectmin)) - torch.log(torch.clamp(torch.abs(u3r * v2r - u2r * v3r), min=protectmin))
        #     logvr = torch.clamp(logvr, min=-10, max=10)
        #
        #     low_anghr = torch.atan2(-a1r, b1r)
        #     high_anghr = torch.atan2(a2r, -b2r)
        #     pred_anghr = torch.atan2(a3r, -b3r)
        #     inboundhr = ((pred_anghr < (high_anghr - anglebound)) * (pred_anghr > (low_anghr + anglebound))).float()
        #
        #     low_angvr = torch.atan2(-u1r, v1r)
        #     high_angvr = torch.atan2(u2r, -v2r)
        #     pred_angvr = torch.atan2(u3r, -v3r)
        #     inboundvr = ((pred_angvr < (high_angvr - anglebound)) * (pred_angvr > (low_angvr + anglebound))).float()
        #
        #     loss = torch.abs(gthr - loghr) *inboundhr + torch.abs(gtvr - logvr) * inboundvr + \
        #            torch.abs((high_anghr + low_anghr) / 2 - pred_anghr) * (1 - inboundhr) * inboundw + torch.abs((high_angvr + low_angvr) / 2 - pred_angvr) * (1 - inboundvr) * inboundw
        #
        #     expadam.zero_grad()
        #     loss.backward()
        #     expadam.step()
        #
        #     normrec.append(surfnormr.detach().cpu().numpy())
        #     print("Iteration: %d, Loss: %f, ang: %f, vec: (%f, %f, %f)" % (i, float(loss.detach().cpu().numpy()), float(torch.sum(surfnormgt * surfnormr)), normrec[-1][0], normrec[-1][1], normrec[-1][2]))
        #
        # normrecnp = np.stack(normrec, axis=0)
        # import matlab
        # import matlab.engine
        # eng = matlab.engine.start_matlab()
        #
        # pltx = matlab.double(normrecnp[:, 0].tolist())
        # plty = matlab.double(normrecnp[:, 1].tolist())
        # pltz = matlab.double(normrecnp[:, 2].tolist())
        #
        # surfnormgtnp = surfnormgt.detach().cpu().numpy()
        # pltgtx = matlab.double(surfnormgtnp[0:1].tolist())
        # pltgty = matlab.double(surfnormgtnp[1:2].tolist())
        # pltgtz = matlab.double(surfnormgtnp[2:3].tolist())
        #
        # eng.figure()
        # eng.plot3(pltx, plty, pltz)
        # eng.eval('hold on', nargout=0)
        # eng.scatter3(pltgtx, pltgty, pltgtz, 15, 'filled', 'r', nargout=0)
        # eng.eval('axis equal', nargout=0)
        # eng.title('Optimization on single pixel for 25000 epoch')
        # eng.xlabel('Normx')
        # eng.ylabel('Normy')
        # eng.zlabel('Normz')
        # eng.eval('grid on', nargout=0)
        #
        # # Singular value analysis
        # ins = torch.isinf(logh)
        # batchind = list()
        # for i in range(self.batch_size):
        #     batchind.append(torch.ones([self.height, self.width]) * i)
        # batchind = torch.stack(batchind, dim=0).cuda()
        #
        # insbz = batchind[ins].long()[0].detach().cpu().numpy()
        # insxx = self.xx[ins].long()[0].detach().cpu().numpy()
        # insyy = self.yy[ins].long()[0].detach().cpu().numpy()
        #
        # intrinsicnpck = intrinsic[insbz, :, :].cpu().numpy()
        #
        # ptsckdnp1 = depthMap[insbz, 0, insyy, insxx].detach().cpu().numpy()
        # ptscknp1 = np.linalg.inv(intrinsicnpck) @ np.array([[insxx * ptsckdnp1, insyy * ptsckdnp1, ptsckdnp1, 1]]).T
        #
        # ptsckdnp2 = depthMap[insbz, 0, insyy, insxx + 1].detach().cpu().numpy()
        # ptscknp2 = np.linalg.inv(intrinsicnpck) @ np.array([[(insxx + 1) * ptsckdnp2, insyy * ptsckdnp2, ptsckdnp2, 1]]).T
        #
        # singularnormx = surfnormx[insbz, :, insyy, insxx]
        # singularnormx = singularnormx / torch.sqrt(torch.sum(singularnormx ** 2))
        # singularnumx = ptscknp2 - ptscknp1
        # singularnumx = (singularnumx / np.sqrt(np.sum(singularnumx ** 2)))[0:3, 0]
        # singulartheolx = np.array([(insxx + 1 - bx[insbz, insyy, insxx].detach().cpu().numpy()) / fx[insbz, insyy, insxx].detach().cpu().numpy(), (insyy - by[insbz, insyy, insxx].detach().cpu().numpy()) / fy[insbz, insyy, insxx].detach().cpu().numpy(), 1])
        # singulartheolx = singulartheolx / np.sqrt(np.sum(singulartheolx ** 2))
        #
        # pltrange = 1e1
        # plt.figure()
        # plt.hist(np.clip(logh.detach().cpu().numpy().flatten(), a_min=-pltrange, a_max=pltrange), bins=100, range=[-0.01, 0.01])
        return
