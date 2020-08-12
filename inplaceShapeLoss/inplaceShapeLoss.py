from __future__ import absolute_import, division, print_function

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from layers import *
import cv2
import numpy as np
import torch

import inplaceShapeLoss_cuda

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


class InplaceShapeLoss(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(InplaceShapeLoss, self).__init__()

    def grad_check(self, logdepth, logratioh, logratiov, valindic, srw, srh):
        rndseeds = torch.rand_like(logdepth)

        ckc = torch.randint(0, logdepth.shape[0], [1])[0]
        ckh = torch.randint(0, logdepth.shape[2], [1])[0]
        ckw = torch.randint(0, logdepth.shape[3], [1])[0]

        logratioh[ckc, 0, ckh, ckw] = logratioh[ckc, 0, ckh, ckw] - 100

        lossrec = torch.zeros_like(logdepth)
        countsrec = torch.zeros_like(logdepth)
        inplaceShapeLoss_cuda.inplaceShapeLoss_forward(logdepth, logratioh, logratiov, valindic.int(), lossrec, countsrec, rndseeds, srw, srh)

        gradh = torch.zeros_like(logdepth)
        gradv = torch.zeros_like(logdepth)
        grad_re = torch.ones_like(logdepth)
        inplaceShapeLoss_cuda.inplaceShapeLoss_backward(logdepth, logratioh, logratiov, valindic.int(), grad_re, gradh, gradv, countsrec, rndseeds, srw, srh)

        torch.sum(lossrec[ckc, 0, ckh, ckw:ckw+2])
        torch.sum(3 / countsrec[ckc, 0, ckh, ckw:ckw+2])

        # Compute Numerical Gradient, horizontal
        dev = 5

        lossrec1 = torch.zeros_like(logdepth)
        countsrec = torch.zeros_like(logdepth)
        logratioh1 = torch.clone(logratioh)
        logratioh1[ckc,0,ckh,ckw] = logratioh1[ckc,0,ckh,ckw] + dev
        inplaceShapeLoss_cuda.inplaceShapeLoss_forward(logdepth, logratioh1, logratiov, valindic.int(), lossrec1, countsrec, rndseeds, srw, srh)
        loss1 = torch.sum(lossrec1)

        lossrec2 = torch.zeros_like(logdepth)
        countsrec = torch.zeros_like(logdepth)
        logratioh2 = torch.clone(logratioh)
        logratioh2[ckc,0,ckh,ckw] = logratioh2[ckc,0,ckh,ckw] - dev
        inplaceShapeLoss_cuda.inplaceShapeLoss_forward(logdepth, logratioh2, logratiov, valindic.int(), lossrec2, countsrec, rndseeds, srw, srh)
        loss2 = torch.sum(lossrec2)

        numgrad = (loss1 - loss2) / dev / 2
        thegrqad = gradh[ckc,0,ckh,ckw]
        print(numgrad)
        print(thegrqad)

        return lossrec


    @staticmethod
    def forward(ctx, logdepth, logratioh, logratiov, valindic, srw, srh):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        indc: indices for channel
        indx: indices for x
        """
        torch.random.manual_seed(100)
        rndseeds = torch.rand_like(logdepth)

        lossrec = torch.zeros_like(logdepth)
        countsrec = torch.zeros_like(logdepth)
        inplaceShapeLoss_cuda.inplaceShapeLoss_forward(logdepth, logratioh, logratiov, valindic.int(), lossrec, countsrec, rndseeds, srw, srh)

        ctx.save_for_backward(logdepth, logratioh, logratiov, valindic.int(), countsrec, rndseeds)
        ctx.srw = srw
        ctx.srh = srh
        return lossrec

    @staticmethod
    def backward(ctx, grad_re):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_re = grad_re.contiguous()
        logdepth, logratioh, logratiov, valindic, countsrec, rndseeds = ctx.saved_tensors
        srw = ctx.srw
        srh = ctx.srh

        gradh = torch.zeros_like(logdepth)
        gradv = torch.zeros_like(logdepth)
        inplaceShapeLoss_cuda.inplaceShapeLoss_backward(logdepth, logratioh, logratiov, valindic, grad_re, gradh, gradv, countsrec, rndseeds, srw, srh)
        return None, gradh, gradv, None, None, None



def cvt_png2depth_PreSIL(tsv_depth):
    maxM = 1000
    sMax = 255 ** 3 - 1

    tsv_depth = tsv_depth.astype(np.float)
    depthIm = (tsv_depth[:, :, 0] * 255 * 255 + tsv_depth[:, :, 1] * 255 + tsv_depth[:, :, 2]) / sMax * maxM
    return depthIm

if __name__ == "__main__":
    """Evaluates a pretrained model using a specified test set
    """
    from torch.autograd import gradcheck

    preSILroot = '/home/shengjie/Documents/Data/PreSIL_organized'
    seq = '000000'
    frameinds = [164, 174]

    intrinsic = np.array([
        [960, 0, 960],
        [0, 960, 540],
        [0, 0, 1]
    ])

    height = 448
    width = 1024
    localGeomDesp = LocalThetaDesp(height=height, width=width, batch_size=len(frameinds), intrinsic=intrinsic).cuda()

    rgb_torchs = list()
    depth_torchs = list()
    for frameind in frameinds:
        rgb = pil.open(os.path.join(preSILroot, seq, 'rgb', str(frameind).zfill(6) + '.png'))

        depth = pil.open(os.path.join(preSILroot, seq, 'depth', str(frameind).zfill(6) + '.png'))
        depth = cvt_png2depth_PreSIL(np.array(depth))

        rgb_torch = torch.from_numpy(np.array(rgb)).unsqueeze(0).permute([0,3,1,2]).float().cuda().contiguous() / 255.0
        depth_torch = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float().cuda()
        depth_torch = torch.clamp(depth_torch, min=0, max=120)

        rgb_torchs.append(rgb_torch)
        depth_torchs.append(depth_torch)
        # tensor2disp(1/depth_torch, vmax=0.5, ind=0).show()

    rgb_torchs = torch.cat(rgb_torchs, dim=0).contiguous()
    depth_torchs = torch.cat(depth_torchs, dim=0).contiguous()

    htheta, vtheta = localGeomDesp.get_theta(depth_torchs)
    ratioh, ratiohl, ratiov, ratiovl = localGeomDesp.get_ratio(htheta, vtheta)

    # gradcker = InplaceShapeLoss()
    # lossrec = gradcker.grad_check(torch.log(depth_torchs), ratiohl, ratiovl, depth_torchs > 0, 3, 5)

    htheta_act_noisy = torch.zeros_like(htheta, requires_grad=True)
    vtheta_act_noisy = torch.zeros_like(vtheta, requires_grad=True)

    inplaceSL = InplaceShapeLoss.apply
    optimizer = torch.optim.Adam([htheta_act_noisy, vtheta_act_noisy], lr=1e-2)
    for k in range(1500):
        htheta_noisy = torch.sigmoid(htheta_act_noisy) * 2 * np.pi
        vtheta_noisy = torch.sigmoid(vtheta_act_noisy) * 2 * np.pi

        inbl, outbl, scl = localGeomDesp.inplacePath_loss(depth_torchs, htheta_noisy, vtheta_noisy)
        loss = inbl + outbl / 10 + scl

        # _, ratiohl_noisy, _, ratiovl_noisy = localGeomDesp.get_ratio(htheta_noisy, vtheta_noisy)
        # lossrec = inplaceSL(torch.log(depth_torchs), ratiohl_noisy, ratiovl_noisy, depth_torchs > 0, 1, 1)
        # loss = torch.mean(lossrec)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("It:%d, inbound loss:%f, outbound loss: %f" %(k, inbl.detach().cpu().numpy(), outbl.detach().cpu().numpy()))

    tensor2disp(htheta_noisy - 1, vmax=4, ind=0).show()
    tensor2disp(vtheta_noisy - 1, vmax=4, ind=0).show()

