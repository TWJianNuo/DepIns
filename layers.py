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