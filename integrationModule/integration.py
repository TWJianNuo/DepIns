import torch
import shapeintegration_cuda


class IntegrationFunction(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(IntegrationFunction, self).__init__()

    @staticmethod
    def forward(ctx, pred_ang, pred_log, confidence, semantics, mask, pred_depth, variancebar, h, w, bz):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        pred_ang = pred_ang.contiguous().float()
        pred_log = pred_log.contiguous().float()
        pred_depth = pred_depth.contiguous().float()
        confidence = confidence.contiguous().float()
        semantics = semantics.contiguous().int()
        mask = mask.contiguous().int()

        integrated_depth = torch.zeros_like(pred_depth)
        summedConfidence = torch.zeros_like(pred_depth)
        shapeintegration_cuda.shapeIntegration_forward(pred_ang, pred_log, confidence, semantics, mask, pred_depth, integrated_depth, summedConfidence, variancebar, h, w, bz)

        ctx.save_for_backward(pred_ang, pred_log, confidence, semantics, mask, pred_depth, integrated_depth, summedConfidence, variancebar)
        ctx.h = h
        ctx.w = w
        ctx.bz = bz
        return integrated_depth

    @staticmethod
    def backward(ctx, grad_depth):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        grad_depth = grad_depth.contiguous()

        pred_ang, pred_log, confidence, semantics, mask, pred_depth, integrated_depth, summedConfidence, variancebar = ctx.saved_tensors
        h = ctx.h
        w = ctx.w
        bz = ctx.bz

        gradout_depth = torch.zeros_like(grad_depth)
        gradout_confidence = torch.zeros_like(grad_depth)

        shapeintegration_cuda.shapeIntegration_backward(pred_ang, pred_log, confidence, semantics, mask, pred_depth, integrated_depth, summedConfidence, grad_depth, gradout_depth, gradout_confidence, variancebar, h, w, bz)

        return None, None, gradout_confidence, None, None, gradout_depth, None, None, None, None

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