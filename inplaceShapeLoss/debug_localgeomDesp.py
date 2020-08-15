import torch
import torch.nn as nn
import numpy as np

class LocalThetaDesp(nn.Module):
    def __init__(self, height, width, batch_size, intrinsic, extrinsic=None, STEREO_SCALE_FACTOR=5.4, minDepth=0.1, maxDepth=100, patchw=15, patchh=3):
        super(LocalThetaDesp, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size

        self.boundStabh = 0.02
        self.boundStabv = 0.02
        self.invIn = nn.Parameter(torch.from_numpy(np.linalg.inv(intrinsic)).float(), requires_grad=False)
        self.intrinsic = nn.Parameter(torch.from_numpy(intrinsic).float(), requires_grad=False)

        # Init grid points
        xx, yy = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.xx = nn.Parameter(torch.from_numpy(xx).float(), requires_grad=False)
        self.yy = nn.Parameter(torch.from_numpy(yy).float(), requires_grad=False)
        self.pixelLocs = nn.Parameter(torch.stack([self.xx, self.yy, torch.ones_like(self.xx)], dim=2), requires_grad=False)

        # Compute Horizontal Direciton
        hdir1 = self.invIn.unsqueeze(0).unsqueeze(0).expand([self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(3)
        hdir2 = self.invIn.unsqueeze(0).unsqueeze(0).expand([self.height, self.width, -1, -1]) @ (self.pixelLocs + torch.Tensor([1, 0, 0])).unsqueeze(3)
        hdir3 = torch.cross(hdir1, hdir2)
        hdir3 = hdir3 / torch.norm(hdir3, dim=2, keepdim=True)

        # Compute horizontal x axis
        hxd = torch.Tensor([0, 0, 1]).unsqueeze(1) - torch.sum(hdir3 * torch.Tensor([0, 0, 1]).unsqueeze(1), dim=[2, 3], keepdim=True) * hdir3
        hxd = hxd / torch.norm(hxd, dim=2, keepdim=True)
        hyd = torch.cross(hxd, hdir3)
        hM = torch.stack([hxd.squeeze(3), hyd.squeeze(3)], dim=2)
        self.hM = nn.Parameter(hM, requires_grad=False)

        hdir1p = self.hM @ (hdir1 / torch.norm(hdir1, keepdim=True, dim=2))
        hdir2p = self.hM @ (hdir2 / torch.norm(hdir2, keepdim=True, dim=2))

        lowerboundh = torch.atan2(hdir1p[:, :, 1, 0], hdir1p[:, :, 0, 0])
        lowerboundh = self.convert_htheta(lowerboundh) - float(np.pi) + self.boundStabh
        upperboundh = torch.atan2(hdir2p[:, :, 1, 0], hdir2p[:, :, 0, 0])
        upperboundh = self.convert_htheta(upperboundh) - self.boundStabh
        middeltargeth = (lowerboundh + upperboundh) / 2
        self.lowerboundh = nn.Parameter(lowerboundh.unsqueeze(0).unsqueeze(0).expand([self.batch_size, -1, -1, -1]), requires_grad=False)
        self.upperboundh = nn.Parameter(upperboundh.unsqueeze(0).unsqueeze(0).expand([self.batch_size, -1, -1, -1]), requires_grad=False)
        self.middeltargeth = nn.Parameter(middeltargeth.unsqueeze(0).unsqueeze(0).expand([self.batch_size, -1, -1, -1]), requires_grad=False)

        # Compute Vertical Direciton
        vdir1 = self.invIn.unsqueeze(0).unsqueeze(0).expand([self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(3)
        vdir2 = self.invIn.unsqueeze(0).unsqueeze(0).expand([self.height, self.width, -1, -1]) @ (self.pixelLocs + torch.Tensor([0, 1, 0])).unsqueeze(3)
        vdir3 = torch.cross(vdir1, vdir2)
        vdir3 = vdir3 / torch.norm(vdir3, dim=2, keepdim=True)

        # Compute vertical x axis
        vxd = torch.Tensor([0, 0, 1]).unsqueeze(1) - torch.sum(vdir3 * torch.Tensor([0, 0, 1]).unsqueeze(1), dim=[2, 3], keepdim=True) * vdir3
        vxd = vxd / torch.norm(vxd, dim=2, keepdim=True)
        vyd = torch.cross(vxd, vdir3)
        vM = torch.stack([vxd.squeeze(3), vyd.squeeze(3)], dim=2)
        self.vM = nn.Parameter(vM, requires_grad=False)

        vdir1p = self.vM @ (vdir1 / torch.norm(vdir1, keepdim=True, dim=2))
        vdir2p = self.vM @ (vdir2 / torch.norm(vdir2, keepdim=True, dim=2))

        lowerboundv = torch.atan2(vdir1p[:, :, 1, 0], vdir1p[:, :, 0, 0])
        lowerboundv = self.convert_vtheta(lowerboundv) - float(np.pi) + self.boundStabv
        upperboundv = torch.atan2(vdir2p[:, :, 1, 0], vdir2p[:, :, 0, 0])
        upperboundv = self.convert_vtheta(upperboundv) - self.boundStabv
        middeltargetv = (lowerboundv + upperboundv) / 2
        self.lowerboundv = nn.Parameter(lowerboundv.unsqueeze(0).unsqueeze(0).expand([self.batch_size, -1, -1, -1]), requires_grad=False)
        self.upperboundv = nn.Parameter(upperboundv.unsqueeze(0).unsqueeze(0).expand([self.batch_size, -1, -1, -1]), requires_grad=False)
        self.middeltargetv = nn.Parameter(middeltargetv.unsqueeze(0).unsqueeze(0).expand([self.batch_size, -1, -1, -1]), requires_grad=False)

        weightl = torch.Tensor(
            [[0, 0, 0],
             [0, -1, 1],
             [0, 0, 0]]
        )
        self.hdiffConv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.hdiffConv.weight = torch.nn.Parameter(weightl.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

        weightv = torch.Tensor(
            [[0, 0, 0],
             [0, -1, 0],
             [0, 1, 0]]
        )
        self.vdiffConv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.vdiffConv.weight = torch.nn.Parameter(weightv.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

        copyl = torch.Tensor(
            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]]
        )
        self.copylConv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.copylConv.weight = torch.nn.Parameter(copyl.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

        copyv = torch.Tensor(
            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]
        )
        self.copyvConv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.copyvConv.weight = torch.nn.Parameter(copyv.unsqueeze(0).unsqueeze(0).float(), requires_grad=False)

        self.mink = -150
        self.maxk = 150

        npts3d = (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(0).unsqueeze(4).expand([self.batch_size, -1, -1, -1, -1]))

        npts3d_shifted_h = (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ (self.pixelLocs + torch.Tensor([1, 0, 0])).unsqueeze(0).unsqueeze(4).expand([self.batch_size, -1, -1, -1, -1]))
        npts3d_p_h = self.hM.unsqueeze(0).expand([self.batch_size, -1, -1, -1, -1]) @ npts3d
        npts3d_p_shifted_h = self.hM.unsqueeze(0).expand([self.batch_size, -1, -1, -1, -1]) @ npts3d_shifted_h
        self.npts3d_p_h = nn.Parameter(npts3d_p_h, requires_grad=False)
        self.npts3d_p_shifted_h = nn.Parameter(npts3d_p_shifted_h, requires_grad=False)

        npts3d_shifted_v = (self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand([self.batch_size, self.height, self.width, -1, -1]) @ (self.pixelLocs + torch.Tensor([0, 1, 0])).unsqueeze(0).unsqueeze(4).expand([self.batch_size, -1, -1, -1, -1]))
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
            [[0, -1, 0, 1, 0],
             [-1, 0, 0, 1, 0],
             [-1, 0, 0, 0, 1],
             ]
        )
        idh = torch.Tensor(
            [[0, 1, 0, 1, 0],
             [1, 0, 0, 1, 0],
             [1, 0, 0, 0, 1],
             ]
        )
        self.lossh = torch.nn.Conv2d(1, 3, [1, 5], padding=[0, 2], bias=False)
        self.lossh.weight = torch.nn.Parameter(lossh.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)
        self.gth = torch.nn.Conv2d(1, 3, [1, 5], padding=[0, 2], bias=False)
        self.gth.weight = torch.nn.Parameter(gth.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)
        self.idh = torch.nn.Conv2d(1, 3, [1, 5], padding=[0, 2], bias=False)
        self.idh.weight = torch.nn.Parameter(idh.unsqueeze(1).unsqueeze(1).float(), requires_grad=False)

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
        self.lossv = torch.nn.Conv2d(1, 6, [9, 1], padding=[4, 0], bias=False)
        self.lossv.weight = torch.nn.Parameter(lossv.unsqueeze(1).unsqueeze(3).float(), requires_grad=False)
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
        self.selfconhInd.weight = torch.nn.Parameter(selfconhIndW.unsqueeze(0).unsqueeze(0).float(),
                                                     requires_grad=False)
        selfconvIndW = torch.Tensor(
            [[0, 0, 0],
             [0, 0, 1],
             [0, 1, 0]
             ]
        )
        self.selfconvInd = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.selfconvInd.weight = torch.nn.Parameter(selfconvIndW.unsqueeze(0).unsqueeze(0).float(),
                                                     requires_grad=False)

        from inplaceShapeLoss import InplaceShapeLoss
        self.inplaceSL = InplaceShapeLoss.apply

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
        pts3d = depthmap.squeeze(1).unsqueeze(3).unsqueeze(4).expand([-1, -1, -1, 3, -1]) * (
                    self.invIn.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(
                        [self.batch_size, self.height, self.width, -1, -1]) @ self.pixelLocs.unsqueeze(0).unsqueeze(
                4).expand([self.batch_size, -1, -1, -1, -1]))

        hcord = self.hM.unsqueeze(0).expand([self.batch_size, -1, -1, -1, -1]) @ pts3d
        hdify = self.hdiffConv(hcord[:, :, :, 1, 0].unsqueeze(1))
        hdifx = self.hdiffConv(hcord[:, :, :, 0, 0].unsqueeze(1))

        htheta = torch.atan2(hdify, hdifx)
        htheta = self.convert_htheta(htheta)
        htheta = torch.clamp(htheta, min=1e-3, max=float(np.pi) * 2 - 1e-3)
        vcord = self.vM.unsqueeze(0).expand([self.batch_size, -1, -1, -1, -1]) @ pts3d
        vdify = self.vdiffConv(vcord[:, :, :, 1, 0].unsqueeze(1))
        vdifx = self.vdiffConv(vcord[:, :, :, 0, 0].unsqueeze(1))

        vtheta = torch.atan2(vdify, vdifx)
        vtheta = self.convert_vtheta(vtheta)
        vtheta = torch.clamp(vtheta, min=1e-3, max=float(np.pi) * 2 - 1e-3)
        return htheta, vtheta

    def cleaned_path_loss(self, depthmap, htheta, vtheta):
        depthmapl = torch.log(torch.clamp(depthmap, min=1e-3))
        inboundh = (htheta < self.upperboundh) * (htheta > self.lowerboundh)
        inboundh = inboundh.float()
        outboundh = 1 - inboundh

        bk_htheta = self.backconvert_htheta(htheta)
        npts3d_pdiff_uph = torch.cos(bk_htheta) * self.npts3d_p_h[:, :, :, 1, 0].unsqueeze(1) - torch.sin(bk_htheta) * self.npts3d_p_h[:, :, :, 0, 0].unsqueeze(1)
        npts3d_pdiff_downh = torch.cos(bk_htheta) * self.npts3d_p_shifted_h[:, :, :, 1, 0].unsqueeze(1) - torch.sin(bk_htheta) * self.npts3d_p_shifted_h[:, :, :, 0, 0].unsqueeze(1)
        ratiohl = torch.log(torch.clamp(torch.abs(npts3d_pdiff_uph), min=1e-4)) - torch.log(torch.clamp(torch.abs(npts3d_pdiff_downh), min=1e-4))

        lossh = self.lossh(ratiohl)
        gth = self.gth(depthmapl)
        indh = (self.idh((depthmap > 0).float()) == 2).float() * (self.lossh(outboundh) == 0).float()
        hloss = (torch.sum(torch.abs(gth - lossh) * inboundh * indh) + torch.sum(torch.abs(self.middeltargeth - htheta) * outboundh * indh) / 20) / (torch.sum(indh) + 1)

        inboundv = (vtheta < self.upperboundv) * (vtheta > self.lowerboundv)
        inboundv = inboundv.float()
        outboundv = 1 - inboundv

        bk_vtheta = self.backconvert_vtheta(vtheta)
        npts3d_pdiff_upv = torch.cos(bk_vtheta) * self.npts3d_p_v[:, :, :, 1, 0].unsqueeze(1) - torch.sin(bk_vtheta) * self.npts3d_p_v[:, :, :, 0, 0].unsqueeze(1)
        npts3d_pdiff_downv = torch.cos(bk_vtheta) * self.npts3d_p_shifted_v[:, :, :, 1, 0].unsqueeze(1) - torch.sin(bk_vtheta) * self.npts3d_p_shifted_v[:, :, :, 0, 0].unsqueeze(1)
        ratiovl = torch.log(torch.clamp(torch.abs(npts3d_pdiff_upv), min=1e-4)) - torch.log(torch.clamp(torch.abs(npts3d_pdiff_downv), min=1e-4))

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

        npts3d_pdiff_uph = torch.cos(bk_htheta).squeeze(1) * self.npts3d_p_h[:, :, :, 1, 0] - torch.sin(
            bk_htheta).squeeze(1) * self.npts3d_p_h[:, :, :, 0, 0]
        npts3d_pdiff_downh = torch.cos(bk_htheta).squeeze(1) * self.npts3d_p_shifted_h[:, :, :, 1, 0] - torch.sin(
            bk_htheta).squeeze(1) * self.npts3d_p_shifted_h[:, :, :, 0, 0]
        ratioh = npts3d_pdiff_uph / npts3d_pdiff_downh
        ratioh = torch.clamp(ratioh, min=1e-3)
        ratiohl = torch.log(ratioh)

        bk_vtheta = self.backconvert_vtheta(vtheta)

        npts3d_pdiff_upv = torch.cos(bk_vtheta).squeeze(1) * self.npts3d_p_v[:, :, :, 1, 0] - torch.sin(
            bk_vtheta).squeeze(1) * self.npts3d_p_v[:, :, :, 0, 0]
        npts3d_pdiff_downv = torch.cos(bk_vtheta).squeeze(1) * self.npts3d_p_shifted_v[:, :, :, 1, 0] - torch.sin(
            bk_vtheta).squeeze(1) * self.npts3d_p_shifted_v[:, :, :, 0, 0]
        ratiov = npts3d_pdiff_upv / npts3d_pdiff_downv
        ratiov = torch.clamp(ratiov, min=1e-3)
        ratiovl = torch.log(ratiov)

        ratioh = ratioh.unsqueeze(1)
        ratiohl = ratiohl.unsqueeze(1)
        ratiov = ratiov.unsqueeze(1)
        ratiovl = ratiovl.unsqueeze(1)
        return ratioh, ratiohl, ratiov, ratiovl


    def inplacePath_loss(self, depthmap, htheta, vtheta, balancew=10, isExcludehw=False):
        srw = 5
        srh = 15
        srwh = 5

        depthmapl = torch.log(torch.clamp(depthmap, min=1e-3))

        inboundh = (htheta < self.upperboundh) * (htheta > self.lowerboundh)
        inboundh = inboundh.float()
        outboundh = 1 - inboundh

        inboundv = (vtheta < self.upperboundv) * (vtheta > self.lowerboundv)
        inboundv = inboundv.float()
        outboundv = 1 - inboundv

        bk_htheta = self.backconvert_htheta(htheta)
        npts3d_pdiff_uph = torch.cos(bk_htheta) * self.npts3d_p_h[:, :, :, 1, 0].unsqueeze(1) - torch.sin(
            bk_htheta) * self.npts3d_p_h[:, :, :, 0, 0].unsqueeze(1)
        npts3d_pdiff_downh = torch.cos(bk_htheta) * self.npts3d_p_shifted_h[:, :, :, 1, 0].unsqueeze(1) - torch.sin(
            bk_htheta) * self.npts3d_p_shifted_h[:, :, :, 0, 0].unsqueeze(1)
        ratiohl = torch.log(torch.clamp(torch.abs(npts3d_pdiff_uph), min=1e-4)) - torch.log(
            torch.clamp(torch.abs(npts3d_pdiff_downh), min=1e-4))

        bk_vtheta = self.backconvert_vtheta(vtheta)
        npts3d_pdiff_upv = torch.cos(bk_vtheta) * self.npts3d_p_v[:, :, :, 1, 0].unsqueeze(1) - torch.sin(
            bk_vtheta) * self.npts3d_p_v[:, :, :, 0, 0].unsqueeze(1)
        npts3d_pdiff_downv = torch.cos(bk_vtheta) * self.npts3d_p_shifted_v[:, :, :, 1, 0].unsqueeze(1) - torch.sin(
            bk_vtheta) * self.npts3d_p_shifted_v[:, :, :, 0, 0].unsqueeze(1)
        ratiovl = torch.log(torch.clamp(torch.abs(npts3d_pdiff_upv), min=1e-4)) - torch.log(
            torch.clamp(torch.abs(npts3d_pdiff_downv), min=1e-4))

        lossrech = self.inplaceSL(depthmapl, ratiohl, ratiovl, depthmap > 0, srw, 0)
        lossrechi = (lossrech > 0).float() * inboundh

        lossrecv = self.inplaceSL(depthmapl, ratiohl, ratiovl, depthmap > 0, 0, srh)
        lossrecvi = (lossrecv > 0).float() * inboundv


        lossrecvh = self.inplaceSL(depthmapl, ratiohl, ratiovl, depthmap > 0, srwh, srwh)
        lossrecvhi = (lossrecvh > 0).float() * inboundv * inboundh


        outbl = torch.sum(torch.abs(self.middeltargeth - htheta) * outboundh) / self.height / self.width + \
                torch.sum(torch.abs(self.middeltargetv - vtheta) * outboundv) / self.height / self.width

        if not isExcludehw:
            inbl = (torch.sum(lossrech * lossrechi) / (torch.sum(lossrechi) + 1) +
                    torch.sum(lossrecv * lossrecvi) / (torch.sum(lossrecvi) + 1) / balancew +
                    torch.sum(lossrecvh * lossrecvhi) / (torch.sum(lossrecvhi) + 1) / balancew) / 3
        else:
            inbl = (torch.sum(lossrech * lossrechi) / (torch.sum(lossrechi) + 1) +
                    torch.sum(lossrecv * lossrecvi) / (torch.sum(lossrecvi) + 1) / balancew) / 2

        scl_pixelwise = self.selfconh(ratiohl) + self.selfconv(ratiovl)
        scl_mask = (self.selfconvInd(inboundv) == 2).float() * (self.selfconhInd(inboundh) == 2).float()
        scl = torch.sum(torch.abs(scl_pixelwise) * scl_mask) / (torch.sum(scl_mask) + 1)

        return inbl, outbl, scl
