import torch
import shapeintegration_cuda
import torch.nn as nn

class IntegrationCRFFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(IntegrationCRFFunction, self).__init__()

    @staticmethod
    def forward(ctx, pred_log, semantics, mask, variance, depthin, clipvariance, maxrange):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        pred_log = pred_log.contiguous().float()
        semantics = semantics.contiguous().int()
        mask = mask.contiguous().int()
        variance = variance.contiguous().float()
        depthin = depthin.contiguous().float()

        bz, _, h, w = depthin.shape
        clipvariance = float(clipvariance)
        maxrange = float(maxrange)

        depthout = torch.zeros_like(depthin)
        summedconfidence = torch.zeros_like(depthin)

        shapeintegration_cuda.shapeIntegration_crf_variance_forward(pred_log, semantics, mask, variance, depthin, depthout, summedconfidence, h, w, bz, clipvariance, maxrange)

        ctx.save_for_backward(pred_log, semantics, mask, variance, depthin, depthout, summedconfidence)
        ctx.h = h
        ctx.w = w
        ctx.bz = bz
        ctx.clipvariance = clipvariance
        ctx.maxrange = maxrange
        return depthout

    @staticmethod
    def backward(ctx, grad_depthin):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_depthin = grad_depthin.contiguous().float()

        pred_log, semantics, mask, variance, depthin, depthout, summedconfidence = ctx.saved_tensors
        h = ctx.h
        w = ctx.w
        bz = ctx.bz
        clipvariance = ctx.clipvariance
        maxrange = ctx.maxrange

        grad_varianceout = torch.zeros_like(variance)
        grad_depthout = torch.zeros_like(depthin)

        shapeintegration_cuda.shapeIntegration_crf_variance_backward(pred_log, semantics, mask, variance, depthin, depthout, summedconfidence, grad_depthin, grad_varianceout, grad_depthout, h, w, bz, clipvariance, maxrange)
        return None, None, None, grad_varianceout, grad_depthout, None, None

class CRFIntegrationModule(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, clipvariance, maxrange, lam):
        super(CRFIntegrationModule, self).__init__()

        self.intfunc = IntegrationCRFFunction.apply
        self.clipvariance = clipvariance
        self.maxrange = maxrange
        self.lam = lam

    def forward(self, pred_log, semantics, mask, variance, depthin, times=1):
        depthout = depthin.clone()
        for k in range(times):
            lateralre = self.intfunc(pred_log, semantics, mask, variance, depthout, self.clipvariance, self.maxrange)
            optselector = (lateralre > 0).float()
            depthout = (1 - optselector) * depthin + optselector * (depthin * self.lam + lateralre * (1 - self.lam))
        return depthout

class IntegrationConstrainFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(IntegrationConstrainFunction, self).__init__()

    @staticmethod
    def forward(ctx, pred_log, instancepred, mask, pred_depth, h, w, bz):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        pred_log = pred_log.contiguous().float()
        instancepred = instancepred.contiguous().int()
        mask = mask.contiguous().int()
        pred_depth = pred_depth.contiguous().float()

        constrainout = torch.zeros_like(pred_depth)
        counts = torch.zeros_like(pred_depth)
        shapeintegration_cuda.shapeIntegration_crf_constrain_forward(pred_log, instancepred, mask, pred_depth, constrainout, counts, h, w, bz)

        ctx.save_for_backward(pred_log, instancepred, mask, pred_depth, counts)
        ctx.h = h
        ctx.w = w
        ctx.bz = bz
        return constrainout

    @staticmethod
    def backward(ctx, grad_constrain):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_constrain = grad_constrain.contiguous()

        pred_log, instancepred, mask, pred_depth, counts = ctx.saved_tensors
        h = ctx.h
        w = ctx.w
        bz = ctx.bz

        gradout_depth = torch.zeros_like(grad_constrain)

        shapeintegration_cuda.shapeIntegration_crf_constrain_backward(pred_log, instancepred, mask, pred_depth, counts, grad_constrain, gradout_depth, h, w, bz)
        return None, None, None, gradout_depth, None, None, None