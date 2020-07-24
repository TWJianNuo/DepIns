import math
import matplotlib.pyplot as plt
import PIL.Image as pil
from torch import nn
from torch.autograd import Function
from utils import *
import torch
import copy
import sparsMMul_cuda
from layers import BackProj3D


class SparsMMul(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __int__(self):
        super(SparsMMul, self).__init__()

    @staticmethod
    def forward(ctx, indx, indy, sizeM, mtxvals, vecvals, ist):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        indc: indices for channel
        indx: indices for x
        """

        if not ist:
            re = torch.zeros([sizeM[0]], device="cuda", dtype=torch.float)
        else:
            re = torch.zeros([sizeM[1]], device="cuda", dtype=torch.float)

        sparsMMul_cuda.sparsMMul_forward(indx, indy, mtxvals, vecvals, ist, re)

        ctx.save_for_backward(indx, indy, mtxvals, vecvals)
        ctx.ist = ist
        return re

    @staticmethod
    def backward(ctx, grad_re):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        indx, indy, mtxvals, vecvals = ctx.saved_tensors
        ist = ctx.ist

        grad_mtxvals = torch.zeros_like(mtxvals)
        grad_vecvals = torch.zeros_like(vecvals)
        sparsMMul_cuda.sparsMMul_backward(indx, indy, mtxvals, vecvals, ist, grad_re, grad_mtxvals, grad_vecvals)
        return None, None, None, grad_mtxvals, grad_vecvals, None


'''
# Following is some naive test code for correctness
height = 320
width = 2048

spsheight = height * width * 2
spswidth = height * width

spsSize = 100000

sizeM = torch.Tensor([spsheight, spswidth]).int().cuda()
indx = torch.randperm(spswidth, device="cuda", dtype=torch.long)[0:spsSize]
indy = torch.randperm(spsheight, device="cuda", dtype=torch.long)[0:spsSize]
mtxvals = torch.rand([spsSize], device="cuda", dtype=torch.float)

sparsmmul = SparsMMul.apply

ist = False
vecvals = torch.rand([spswidth], device="cuda", dtype=torch.float)
re = sparsmmul(indx, indy, sizeM, mtxvals, vecvals, ist)

# Check for correctness
spsMtorch = torch.sparse.FloatTensor(torch.stack([indy, indx], dim=0), mtxvals, torch.Size(sizeM))
reck = torch.sparse.mm(spsMtorch, vecvals.unsqueeze(1)).squeeze(1)
print(torch.abs(re - reck).max())

ist = True
vecvals = torch.rand([spsheight], device="cuda", dtype=torch.float)
re = sparsmmul(indx, indy, sizeM, mtxvals, vecvals, ist)

# Check for correctness
spsMtorch = torch.sparse.FloatTensor(torch.stack([indy, indx], dim=0), mtxvals, torch.Size(sizeM))
reck = torch.sparse.mm(spsMtorch.transpose(dim0=0, dim1=1), vecvals.unsqueeze(1)).squeeze(1)
print(torch.abs(re - reck).max())



height = 30
width = 60

spsheight = height * width * 2
spswidth = height * width

spsSize = 1000

sizeM = torch.Tensor([spsheight, spswidth]).int().cuda()
indx = torch.randperm(spswidth, device="cuda", dtype=torch.long)[0:spsSize]
indy = torch.randperm(spsheight, device="cuda", dtype=torch.long)[0:spsSize]

mtxvals = torch.rand([spsSize], device="cuda", dtype=torch.float, requires_grad=True)
vecvals = torch.rand([spswidth], device="cuda", dtype=torch.float, requires_grad=True)

optimizer = torch.optim.Adam([mtxvals, vecvals], lr=1e-3)

ist = False
re = sparsmmul(indx, indy, sizeM, mtxvals, vecvals, ist)
optimizer.zero_grad()
torch.sum(re ** 2).backward()
grad_mtxvals = mtxvals.grad.clone()
grad_vecvals = vecvals.grad.clone()

# Check for correctness
spsMtorch = torch.sparse.FloatTensor(torch.stack([indy, indx], dim=0), mtxvals, torch.Size(sizeM))
reck = torch.sparse.mm(spsMtorch, vecvals.unsqueeze(1)).squeeze(1)
optimizer.zero_grad()
torch.sum(reck ** 2).backward()
gradck_mtxvals = mtxvals.grad.clone()
gradck_vecvals = vecvals.grad.clone()

print(torch.abs(grad_mtxvals - gradck_mtxvals).max())
print(torch.abs(grad_vecvals - gradck_vecvals).max())


vecvals = torch.rand([spsheight], device="cuda", dtype=torch.float, requires_grad=True)
optimizer = torch.optim.Adam([mtxvals, vecvals], lr=1e-3)
ist = True
re = sparsmmul(indx, indy, sizeM, mtxvals, vecvals, ist)
optimizer.zero_grad()
torch.sum(re ** 2).backward()
grad_mtxvals = mtxvals.grad.clone()
grad_vecvals = vecvals.grad.clone()

# Check for correctness
spsMtorch = torch.sparse.FloatTensor(torch.stack([indy, indx], dim=0), mtxvals, torch.Size(sizeM))
reck = torch.sparse.mm(spsMtorch.transpose(dim0=0, dim1=1), vecvals.unsqueeze(1)).squeeze(1)
optimizer.zero_grad()
torch.sum(reck ** 2).backward()
gradck_mtxvals = mtxvals.grad.clone()
gradck_vecvals = vecvals.grad.clone()

print(torch.abs(grad_mtxvals - gradck_mtxvals).max())
print(torch.abs(grad_vecvals - gradck_vecvals).max())


# ========================================== #

import torch.optim as optim
import matplotlib.pyplot as plt
th = 300
tw = 100
tW = torch.rand([th, tw], requires_grad=True)
ta = torch.rand([th, 1], requires_grad=True)

ada_opter = optim.Adam([tW], lr=1e-2)
err = list()

for iter in range(20):
    x0 = torch.zeros([tw, 1])
    d0 = ta
    r0 = tW.t() @ ta
    p0 = tW.t() @ ta
    t0 = tW @ p0
    for i in range(20):
        alpha = torch.sum(r0 * r0) / torch.sum(t0 * t0)
        x0 = x0 + alpha * p0
        d0 = d0 - alpha * t0
        r1 = tW.t() @ d0
        beta = torch.sum(r1 * r1) / torch.sum(r0 * r0)
        p0 = r1 + beta * p0
        t0 = tW @ p0
        r0 = r1
    loss = torch.sum((tW @ x0 - ta) ** 2)
    ada_opter.zero_grad()
    loss.backward()
    ada_opter.step()
    err.append(loss.detach().cpu().numpy())

plt.figure()
plt.stem(err)


import numpy as np
sparsmmul = SparsMMul.apply
xx, yy = np.meshgrid(range(tw), range(th), indexing='xy')
xx = torch.from_numpy(xx.flatten()).long().cuda()
yy = torch.from_numpy(yy.flatten()).long().cuda()

tW = torch.rand([th * tw], requires_grad=True, device="cuda")
ta = torch.rand([th], requires_grad=True, device="cuda")

sizeM = torch.Tensor([th, tw]).int().cuda()
ada_opter = optim.Adam([tW], lr=1e-2)
err = list()
for iter in range(20):
    x0 = torch.zeros([tw], device="cuda", dtype=torch.float)
    d0 = ta
    r0 = sparsmmul(xx, yy, sizeM, tW, ta, True)
    p0 = sparsmmul(xx, yy, sizeM, tW, ta, True)
    t0 = sparsmmul(xx, yy, sizeM, tW, p0, False)
    for i in range(20):
        alpha = torch.sum(r0 * r0) / torch.sum(t0 * t0)
        x0 = x0 + alpha * p0
        d0 = d0 - alpha * t0
        r1 = sparsmmul(xx, yy, sizeM, tW, d0, True)
        beta = torch.sum(r1 * r1) / torch.sum(r0 * r0)
        p0 = r1 + beta * p0
        t0 = sparsmmul(xx, yy, sizeM, tW, p0, False)
        r0 = r1
    loss = torch.sum((sparsmmul(xx, yy, sizeM, tW, x0, False) - ta) ** 2)
    ada_opter.zero_grad()
    loss.backward()
    ada_opter.step()
    err.append(loss.detach().cpu().numpy())

plt.figure()
plt.stem(err)
'''

