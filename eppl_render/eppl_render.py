import math
import matplotlib.pyplot as plt
import PIL.Image as pil
from torch import nn
from torch.autograd import Function
from utils import *
# from numba import jit
import torch
import copy
from layers import BackProj3D
from Oview_Gan.epp_render_c import *

import eppl_cuda

class EPPLFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(EPPLFunction, self).__init__()

    @staticmethod
    def forward(ctx, depthmap, inv_r_sigma, projected2d, selector, Pcombined, kws, sr):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        if selector.dtype != torch.float32:
            selector = selector.float()
        rimg, counter = eppl_cuda.eppl_forward(inv_r_sigma, projected2d, selector, kws, sr)
        ctx.save_for_backward(depthmap, inv_r_sigma, projected2d, selector, Pcombined)
        ctx.counter = counter
        ctx.kws = kws
        ctx.sr = sr
        return rimg

    @staticmethod
    def backward(ctx, grad_rimg):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        if not grad_rimg.is_contiguous():
            grad_rimg = grad_rimg.contiguous()
        depthmap, inv_r_sigma, projected2d, selector, Pcombined = ctx.saved_tensors
        counter = ctx.counter
        kws = ctx.kws
        sr = ctx.sr
        gradDepth, = eppl_cuda.eppl_backward(grad_rimg, depthmap, inv_r_sigma, projected2d, selector.float(), Pcombined, counter, kws, sr)
        return gradDepth, None, None, None, None, None, None



class EPPLFunctionExam(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(EPPLFunction, self).__init__()

    @staticmethod
    def forward(ctx, depthmap, depthmapns, inv_r_sigma, projected2d, projected2dns, selector, selectorns, Pcombined, kws, sr):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        if selector.dtype != torch.float32:
            selector = selector.float()
        if selectorns.dtype != torch.float32:
            selectorns = selectorns.float()
        assert torch.sum(torch.abs(selector - selectorns)) == 0
        rimgns, counterns = eppl_cuda.eppl_forward(inv_r_sigma, projected2dns, selectorns, kws, sr)
        rimg, counterns = eppl_cuda.eppl_forward(inv_r_sigma, projected2d, selector, kws, sr)
        ctx.save_for_backward(depthmapns, inv_r_sigma, projected2dns, selectorns, Pcombined)
        ctx.counterns = counterns
        ctx.kws = kws
        ctx.sr = sr
        ctx.rimgns = rimgns
        return torch.sum((rimgns - rimg) * selectorns)

    @staticmethod
    def backward(ctx, grad_sum):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_rimg = torch.ones_like(ctx.rimgns)
        depthmap, inv_r_sigma, projected2d, selector, Pcombined = ctx.saved_tensors
        counter = ctx.counterns
        kws = ctx.kws
        sr = ctx.sr
        gradDepth, = eppl_cuda.eppl_backward(grad_rimg, depthmap, inv_r_sigma, projected2d, selector.float(), Pcombined, counter, kws, sr)
        gradDepth = gradDepth * grad_sum
        return None, gradDepth, None, None, None, None, None, None, None, None


class EpplRender(nn.Module):
    def __init__(self, height, width, batch_size, lr = 0, sampleNum = 10):
        super(EpplRender, self).__init__()
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
        # self.sr = 39

        self.eps = 1e-6
        # self.sampleNum = 10
        self.sampleNum = sampleNum

        self.lr = lr

        self.epplf = EPPLFunction.apply

        self.nextrinsic = dict()
        self.inv_r_sigma = dict()
        self.Pcombined = dict()
        self.keys = dict()

        self.rand_dmap = dict()
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
        sampleNum = camtrail.shape[1]
        diffvec = (campos + viewbias).unsqueeze(1).expand(self.batch_size, sampleNum, 4, -1) - camtrail
        diffvec = diffvec / torch.norm(diffvec, dim=[2, 3], keepdim=True)

        diffvece = diffvec[:,:,0:3,0]
        camdire = camdir.unsqueeze(1).expand(-1, sampleNum, -1, -1)[:,:,0:3,0]

        v = torch.cross(camdire, diffvece, dim=2)
        s = torch.norm(v, dim=2, keepdim=True)
        c = torch.sum(camdire * diffvece, dim=2, keepdim=True)

        V = torch.zeros([self.batch_size, sampleNum, 3, 3], dtype=torch.float32, device=torch.device("cuda"))
        V[:, :, 0, 1] = -v[:, :, 2]
        V[:, :, 0, 2] = v[:, :, 1]
        V[:, :, 1, 0] = v[:, :, 2]
        V[:, :, 1, 2] = -v[:, :, 0]
        V[:, :, 2, 0] = -v[:, :, 1]
        V[:, :, 2, 1] = v[:, :, 0]

        ce = c.unsqueeze(3).expand(-1,-1,3,3)
        R = torch.eye(3).view(1, 1, 3, 3).expand([self.batch_size, sampleNum, -1, -1]).cuda() + V + V @ V * (1 / (1 + ce))


        r_ex = torch.transpose(R @ torch.transpose(extrinsic[:,0:3,0:3], 1, 2).unsqueeze(1).expand([-1, sampleNum, -1, -1]), 2, 3)
        t_ex = - r_ex @ camtrail[:,:,0:3,:]
        nextrinsic = torch.cat([r_ex, t_ex], dim=3)
        addr = torch.from_numpy(np.array([[[0,0,0,1]]], dtype=np.float32)).unsqueeze(2).expand(self.batch_size, sampleNum, -1, -1).cuda()
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
        sampleNum = nextrinsic.shape[1]
        intrinsice = intrinsic.unsqueeze(1).expand(-1, sampleNum, -1, -1)
        camKs = intrinsice @ nextrinsic

        camKs_e = camKs.view(self.batch_size, sampleNum, 1, 1, 4, 4).expand(-1, -1, self.height, self.width, -1, -1)
        pts3d_e = pts3d.permute([0,2,3,1]).unsqueeze(4).unsqueeze(1).expand([-1,sampleNum, -1, -1, -1, -1])
        projected3d = torch.matmul(camKs_e, pts3d_e).squeeze(5).permute(0, 1, 4, 2, 3)

        projecteddepth = projected3d[:, :, 2, :, :] + self.eps
        projected2d = torch.stack([projected3d[:, :, 0, :, :] / projecteddepth, projected3d[:, :, 1, :, :] / projecteddepth], dim=2)
        selector = (projected2d[:, :, 0, :, :] > 0) * (projected2d[:, :, 0, :, :] < self.width- 1) * (projected2d[:, :, 1, :, :] > 0) * (
                projected2d[:, :, 1, :, :] < self.height - 1) * (projecteddepth > 0)
        selector = selector.unsqueeze(2)
        projecteddepth = projecteddepth.unsqueeze(2)

        if addmask is not None:
            selector = selector * addmask.unsqueeze(1).expand([-1, sampleNum, -1, -1 , -1])
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
        for i in list(np.linspace(0,rimg.shape[1] -1 ,4).astype(np.int)):
        # for i in list(np.linspace(0,self.sampleNum -1 ,self.sampleNum).astype(np.int)):
            imgt.append(np.array(tensor2disp(rimg_t[bz], vmax=vmax, ind=i)))
        return pil.fromarray(np.concatenate(imgt, axis=0))

    def grad_check(self, depthmap, semanticmap, intrinsic, extrinsic, camIndex):
        # Compute Mask
        addmask = self.post_mask(depthmap = depthmap, semanticmap = semanticmap)
        # Generate Camera Track
        camtrail, campos, viewbias, camdir = self.get_camtrail(extrinsic, intrinsic)
        camtrail = camtrail[:, camIndex, :, :]
        nextrinsic = self.cvtCamtrail2Extrsic(camtrail, campos, viewbias, camdir, extrinsic)
        invcamK = torch.inverse(intrinsic @ extrinsic)
        epipoLine, Pcombined = self.get_eppl(intrinsic=intrinsic, extrinsic=extrinsic, nextrinsic=nextrinsic)
        r_sigma, inv_r_sigma, rotM = self.eppl2CovM(epipoLine)

        # Compute GT Mask
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)
        projected2d, projecteddepth, selector = self.proj2de(pts3d=pts3d, intrinsic=intrinsic, nextrinsic=nextrinsic, addmask = addmask)
        # rimg_gt, grad2d, counter_np, depthmapnp_grad = eppl_render(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap.cpu().numpy(), kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)
        # rimg_gt, grad2d, _, depthmapnp_grad = eppl_render_l2(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap.cpu().numpy(), rimg_gt = rimg_gt, kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

        # Grad Check
        # from torch.autograd import gradcheck
        # epplfck = EPPLFunctionExam.apply
        # depthmapns = depthmap + torch.randn(depthmap.shape, device=torch.device("cuda")) * 1e-2
        # depthmapns = nn.Parameter(depthmapns, requires_grad=True)
        # pts3dns = self.bck(predDepth=depthmapns, invcamK=invcamK)
        # projected2dns, _, selectorns = self.proj2de(pts3d=pts3dns, intrinsic=intrinsic, nextrinsic=nextrinsic, addmask=addmask)
        # input = (depthmap, depthmapns, inv_r_sigma, projected2d, projected2dns, selector, selectorns, Pcombined, self.kws, self.sr)
        # gradcheck(epplfck, input, eps=1e-3, atol=1e-3, raise_exception=True)

        epplf = EPPLFunction.apply
        rimg = epplf(depthmap, inv_r_sigma, projected2d, selector, Pcombined, self.kws, self.sr)
        ns = torch.randn(depthmap.shape, device=torch.device("cuda")) * 1e0
        ns = nn.Parameter(ns, requires_grad=True)
        opts = torch.optim.Adam([ns], lr=1e-2)
        imggt = self.show_rendered_eppl(rimg.cpu().numpy())
        lossrec = list()
        for i in range(1000):
            depthmap_ns = depthmap + ns
            pts3d_ns = self.bck(predDepth=depthmap_ns, invcamK=invcamK)
            projected2d_ns, projecteddepth_ns, selector_ns = self.proj2de(pts3d=pts3d_ns, intrinsic=intrinsic,
                                                                          nextrinsic=nextrinsic, addmask=addmask)
            rimg_ns = epplf(depthmap_ns, inv_r_sigma, projected2d_ns, selector_ns, Pcombined, self.kws, self.sr)

            loss = torch.sum(torch.abs(rimg - rimg_ns))
            opts.zero_grad()
            loss.backward()
            opts.step()
            print(loss)



        # Visualization
        epplf = EPPLFunction.apply
        rimg = epplf(depthmap, inv_r_sigma, projected2d, selector, Pcombined, self.kws, self.sr)
        depthmap_ns = depthmap + torch.randn(depthmap.shape, device=torch.device("cuda")) * 1e0
        imggt = self.show_rendered_eppl(rimg.cpu().numpy())
        lossrec = list()
        depthmap_nsg = nn.Parameter(depthmap_ns, requires_grad=True)
        opts = torch.optim.Adam([depthmap_nsg], lr=1e-2)
        for i in range(1000):
            pts3d_ns = self.bck(predDepth=depthmap_nsg, invcamK=invcamK)
            projected2d_ns, projecteddepth_ns, selector_ns = self.proj2de(pts3d=pts3d_ns, intrinsic=intrinsic,
                                                                          nextrinsic=nextrinsic, addmask=addmask)
            rimg_ns = epplf(depthmap_nsg, inv_r_sigma, projected2d_ns, selector_ns, Pcombined, self.kws, self.sr)

            loss = torch.sum(torch.abs(rimg - rimg_ns))
            opts.zero_grad()
            loss.backward()
            opts.step()
            print(loss)

            # vmax = 0.08
            # bz = 0
            # imgt = list()
            # for j in list(np.linspace(0, len(camIndex) - 1, 4).astype(np.int)):
            #     imgt.append(np.array(tensor2disp(rimg_ns[bz].unsqueeze(1), vmax=vmax, ind=j)))
            # imgt = pil.fromarray(np.concatenate(imgt, axis=0))
            # imgt = pil.fromarray(np.concatenate([np.array(imggt), np.array(imgt)], axis=1)).save(os.path.join('/media/shengjie/other/Depins/Depins/visualization/oview_iterate/it_40_torch_indexed', str(i) + '.png'))

        depthmap_ns = depthmap + torch.randn(depthmap.shape, device=torch.device("cuda")) * 1e-1
        pts3d_ns = self.bck(predDepth=depthmap_ns, invcamK=invcamK)
        projected2d_ns, projecteddepth_ns, selector_ns = self.proj2de(pts3d=pts3d_ns, intrinsic=intrinsic, nextrinsic=nextrinsic, addmask = addmask)
        rimg_ns, grad2d_ns, _, depthmapnp_grad = eppl_render_l1(inv_sigmaM=inv_r_sigma.detach().cpu().numpy(), pts2d=projected2d_ns.permute([0,1,3,4,2]).detach().cpu().numpy(), mask = selector_ns.detach().cpu().numpy(), Pcombinednp = Pcombined.cpu().numpy(), depthmapnp = depthmap_ns.cpu().numpy(), rimg_gt = rimg_gt, kws = self.kws, sr = self.sr, bs = self.batch_size, samplesz = self.sampleNum * 2, height = self.height, width = self.width)

    def forward(self, depthmap, semanticmap, intrinsic, extrinsic, camIndex):
        # Compute Mask
        addmask = self.post_mask(depthmap = depthmap, semanticmap = semanticmap)

        # Get Keys for computation wise
        keys = [str(key) + '_' + str(camIndex) for key in torch.sum(torch.abs(extrinsic @ intrinsic), dim = [1,2]).detach().cpu().numpy()]

        inPool = True
        for key in keys:
            if key not in self.keys:
                inPool = False

        if not inPool:
            # Generate Camera Track
            camtrail, campos, viewbias, camdir = self.get_camtrail(extrinsic, intrinsic)
            camtrail = camtrail[:,camIndex,:,:]
            nextrinsic = self.cvtCamtrail2Extrsic(camtrail, campos, viewbias, camdir, extrinsic)
            epipoLine, Pcombined = self.get_eppl(intrinsic=intrinsic, extrinsic=extrinsic, nextrinsic=nextrinsic)
            r_sigma, inv_r_sigma, rotM = self.eppl2CovM(epipoLine)

            # Store
            for idx, key in enumerate(keys):
                self.nextrinsic[key] = nextrinsic[idx].detach()
                self.inv_r_sigma[key] = inv_r_sigma[idx].detach()
                self.Pcombined[key] = Pcombined[idx].detach()
        else:
            nextrinsic = torch.stack([self.nextrinsic[key] for key in keys], dim=0)
            inv_r_sigma = torch.stack([self.inv_r_sigma[key] for key in keys], dim=0)
            Pcombined = torch.stack([self.Pcombined[key] for key in keys], dim=0)

        # Compute GT Mask
        invcamK = torch.inverse(intrinsic @ extrinsic)
        pts3d = self.bck(predDepth=depthmap, invcamK=invcamK)
        projected2d, projecteddepth, selector = self.proj2de(pts3d=pts3d, intrinsic=intrinsic, nextrinsic=nextrinsic, addmask = addmask)
        rimg = self.epplf(depthmap, inv_r_sigma, projected2d, selector, Pcombined, self.kws, self.sr)

        return rimg, inv_r_sigma, Pcombined, addmask

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
        sampleNum = nextrinsic.shape[1]
        if sampleNum not in self.rand_dmap:
            torch.random.manual_seed(200)
            rand_dmap = torch.rand([self.batch_size, 1, 1, self.height, self.width]).expand([-1, sampleNum, -1, -1, -1]).cuda() + 2
            self.rand_dmap[sampleNum] = rand_dmap
        else:
            rand_dmap = self.rand_dmap[sampleNum]
        intrinsic_44, added_extrinsic = self.org_intrinsic(intrinsic)
        intrinsic_44e = intrinsic_44.unsqueeze(1).expand([-1, sampleNum, -1, -1])
        added_extrinsice = added_extrinsic.unsqueeze(1).expand([-1, sampleNum, -1, -1])
        extrinsice = extrinsic.unsqueeze(1).expand([-1, sampleNum, -1, -1])
        extrinsic_old = added_extrinsice @ extrinsice
        extrinsic_new = added_extrinsice @ nextrinsic

        Pold = intrinsic_44e @ extrinsic_old
        Pnew = intrinsic_44e @ extrinsic_new
        Cold = torch.inverse(Pold) @ torch.tensor([0, 0, 0, 1]).view(1, 1, 4, 1).expand([self.batch_size, sampleNum, -1, -1]).float().cuda()
        Pcombined = Pnew @ torch.inverse(Pold)

        xxe = self.xx.unsqueeze(1).expand([-1, sampleNum, -1, -1, -1])
        yye = self.yy.unsqueeze(1).expand([-1, sampleNum, -1, -1, -1])
        onese = self.ones.unsqueeze(1).expand([-1, sampleNum, -1, -1, -1])

        randx = torch.cat([xxe * rand_dmap, yye * rand_dmap, rand_dmap, onese], dim=2)

        Cold_new = Pnew @ Cold
        Cold_new = Cold_new[:, :, 0:3, :]
        Cold_new[:, :, 0, :] = Cold_new[:, :, 0, :] / Cold_new[:, :, 2, :]
        Cold_new[:, :, 1, :] = Cold_new[:, :, 1, :] / Cold_new[:, :, 2, :]
        Cold_new[:, :, 2, :] = Cold_new[:, :, 2, :] / Cold_new[:, :, 2, :]
        Cold_newn = Cold_new / torch.norm(Cold_new, dim = [2,3], keepdim=True).expand([-1,-1,3,-1])
        Cold_newne = Cold_newn.view(self.batch_size, sampleNum, 1, 1, 3, 1).expand([-1, -1, self.height, self.width, -1, -1])

        tmpM = Pnew @ torch.inverse(Pold)
        tmpM = tmpM.view(self.batch_size, sampleNum, 1, 1, 4, 4).expand(-1, -1, self.height, self.width, -1, -1)

        randx_e = randx.permute([0,1,3,4,2]).unsqueeze(5)
        randx_new = torch.matmul(tmpM, randx_e)

        randx_new = randx_new[:,:,:,:,0:3,:]
        randx_new[:, :, :, :, 0, :] = randx_new[:, :, :, :, 0, :] / (randx_new[:, :, :, :, 2, :] + self.eps)
        randx_new[:, :, :, :, 1, :] = randx_new[:, :, :, :, 1, :] / (randx_new[:, :, :, :, 2, :] + self.eps)
        randx_new[:, :, :, :, 2, :] = randx_new[:, :, :, :, 2, :] / (randx_new[:, :, :, :, 2, :] + self.eps)
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
        sampleNum = epipoLine.shape[1]
        ln = torch.sqrt(epipoLine[:,:,0,:,:].pow(2) + epipoLine[:,:,1,:,:].pow(2))
        ldeg = torch.acos(epipoLine[:,:,1,:,:] / ln)

        rotM = torch.stack([torch.stack([torch.cos(ldeg), torch.sin(ldeg)], dim=4), torch.stack([-torch.sin(ldeg), torch.cos(ldeg)], dim=4)], dim=5)
        r_sigma = rotM @ self.sigma.unsqueeze(1).expand([-1,sampleNum,-1,-1,-1,-1]) @ rotM.transpose(dim0=4, dim1=5)
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
        # visibletype = [5]  # pole
        # visibletype = [2, 3, 4]  # building, wall, fence
        # visibletype = [11, 12, 16, 17]  # person, rider, motorcycle, bicycle
        # visibletype = [13, 14, 15, 16]  # car, truck, bus, train
        visibletype = [13, 14, 15, 16]
        addmask = torch.zeros_like(semanticmap)
        for vt in visibletype:
            addmask = addmask + (semanticmap == vt).byte()
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