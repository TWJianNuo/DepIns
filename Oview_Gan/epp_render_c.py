import numpy as np
import numba
from numba import jit

@jit(nopython=True, parallel=True)
def eppl_render(inv_sigmaM, pts2d, mask, Pcombinednp, depthmapnp, kws, sr, bs, samplesz, height, width):
    eps = 1e-6
    srhalf = int((sr - 1) / 2)
    rimg = np.zeros((bs, samplesz, height, width), dtype=np.float32)
    counter = np.zeros((bs, samplesz, height, width), dtype=np.float32)
    grad2d = np.zeros((bs, samplesz, 2, height, width), dtype=np.float32)
    depthmapnp_grad = np.zeros((bs, 1, height, width), dtype=np.float32)
    for c in range(bs):
        for sz in range(samplesz):
            for m in range(height):
                for n in range(width):
                    if mask[c, sz, 0, m, n] > eps:
                        ctx = int(np.round_(pts2d[c, sz, m, n, 0]))
                        cty = int(np.round_(pts2d[c, sz, m, n, 1]))
                        for i in range(ctx - srhalf, ctx + srhalf + 1):
                            for j in range(cty - srhalf, cty + srhalf + 1):
                                if i >= 0 and i < width and j >= 0 and j < height:
                                    fi = float(i)
                                    fj = float(j)
                                    cx = (pts2d[c, sz, m, n, 0] - fi) / kws
                                    cy = (pts2d[c, sz, m, n, 1] - fj) / kws
                                    expx = inv_sigmaM[c,sz,m,n,0,0] * cx * cx + inv_sigmaM[c,sz,m,n,1,0] * cx * cy + inv_sigmaM[c,sz,m,n,0,1] * cx * cy + inv_sigmaM[c,sz,m,n,1,1] * cy * cy
                                    expv = np.exp(-expx / 2)
                                    rimg[c, sz, j, i] = rimg[c, sz, j, i] + expv / 2 / np.pi
                                    counter[c, sz, j, i] = counter[c, sz, j, i] + 1

    for c in range(bs):
        for sz in range(samplesz):
            for m in range(height):
                for n in range(width):
                    rimg[c, sz, m, n] = rimg[c, sz, m, n] / (counter[c, sz, m, n] + eps)

    # Backward
    for c in range(bs):
        for sz in range(samplesz):
            m11 = Pcombinednp[c, sz, 0, 0]
            m12 = Pcombinednp[c, sz, 0, 1]
            m13 = Pcombinednp[c, sz, 0, 2]
            m14 = Pcombinednp[c, sz, 0, 3]

            m21 = Pcombinednp[c, sz, 1, 0]
            m22 = Pcombinednp[c, sz, 1, 1]
            m23 = Pcombinednp[c, sz, 1, 2]
            m24 = Pcombinednp[c, sz, 1, 3]

            m31 = Pcombinednp[c, sz, 2, 0]
            m32 = Pcombinednp[c, sz, 2, 1]
            m33 = Pcombinednp[c, sz, 2, 2]
            m34 = Pcombinednp[c, sz, 2, 3]
            for m in range(height):
                for n in range(width):
                    D = depthmapnp[c, 0, m, n]
                    y = float(m)
                    x = float(n)
                    gradPxDep = (m11 * x + m12 * y + m13) / (m31 * x * D + m32 * y * D + m33 * D + m34) \
                                -(m11 * x * D + m12 * y * D + m13 * D + m14) / ((m31 * x * D + m32 * y * D + m33 * D + m34)**2) * (m31 * x + m32 * y + m33)
                    gradPyDep = (m21 * x + m22 * y + m23) / (m31 * x * D + m32 * y * D + m33 * D + m34) \
                                -(m21 * x * D + m22 * y * D + m23 * D + m24) / ((m31 * x * D + m32 * y * D + m33 * D + m34)**2) * (m31 * x + m32 * y + m33)
                    if mask[c, sz, 0, m, n] > eps:
                        ctx = int(np.round_(pts2d[c, sz, m, n, 0]))
                        cty = int(np.round_(pts2d[c, sz, m, n, 1]))
                        for i in range(ctx - srhalf, ctx + srhalf + 1):
                            for j in range(cty - srhalf, cty + srhalf + 1):
                                if i >= 0 and i < width and j >= 0 and j < height:
                                    fi = float(i)
                                    fj = float(j)
                                    cx = (pts2d[c, sz, m, n, 0] - fi) / kws
                                    cy = (pts2d[c, sz, m, n, 1] - fj) / kws
                                    expx = inv_sigmaM[c,sz,m,n,0,0] * cx * cx + inv_sigmaM[c,sz,m,n,1,0] * cx * cy + inv_sigmaM[c,sz,m,n,0,1] * cx * cy + inv_sigmaM[c,sz,m,n,1,1] * cy * cy
                                    expv = np.exp(-expx / 2) / 2 / np.pi

                                    tmpk = expv / (-2) / (counter[c, sz, j, i] + eps)
                                    tmpx = tmpk * (2 * inv_sigmaM[c,sz,m,n,0,0] * cx + inv_sigmaM[c,sz,m,n,1,0] * cy + inv_sigmaM[c,sz,m,n,0,1] * cy)
                                    grad2d[c, sz, 0, m, n] = grad2d[c, sz, 0, m, n] + tmpx / kws
                                    tmpy = tmpk * (2 * inv_sigmaM[c,sz,m,n,1,1] * cy + inv_sigmaM[c,sz,m,n,1,0] * cx + inv_sigmaM[c,sz,m,n,0,1] * cx)
                                    grad2d[c, sz, 1, m, n] = grad2d[c, sz, 1, m, n] + tmpy / kws

                                    depthmapnp_grad[c, 0, m, n] = depthmapnp_grad[c, 0, m, n] + \
                                                                  tmpk / kws * ((2 * inv_sigmaM[c,sz,m,n,0,0] * cx + inv_sigmaM[c,sz,m,n,1,0] * cy + inv_sigmaM[c,sz,m,n,0,1] * cy) * gradPxDep + (2 * inv_sigmaM[c,sz,m,n,1,1] * cy + inv_sigmaM[c,sz,m,n,1,0] * cx + inv_sigmaM[c,sz,m,n,0,1] * cx) * gradPyDep)

    return rimg, grad2d, counter, depthmapnp_grad



@jit(nopython=True, parallel=True)
def eppl_render_l2(inv_sigmaM, pts2d, mask, Pcombinednp, depthmapnp, rimg_gt, kws, sr, bs, samplesz, height, width):
    eps = 1e-6
    srhalf = int((sr - 1) / 2)
    rimg = np.zeros((bs, samplesz, height, width), dtype=np.float32)
    counter = np.zeros((bs, samplesz, height, width), dtype=np.float32)
    grad2d = np.zeros((bs, samplesz, 2, height, width), dtype=np.float32)
    depthmapnp_grad = np.zeros((bs, 1, height, width), dtype=np.float32)
    for c in range(bs):
        for sz in range(samplesz):
            for m in range(height):
                for n in range(width):
                    if mask[c, sz, 0, m, n] > eps:
                        ctx = int(np.round_(pts2d[c, sz, m, n, 0]))
                        cty = int(np.round_(pts2d[c, sz, m, n, 1]))
                        for i in range(ctx - srhalf, ctx + srhalf + 1):
                            for j in range(cty - srhalf, cty + srhalf + 1):
                                if i >= 0 and i < width and j >= 0 and j < height:
                                    fi = float(i)
                                    fj = float(j)
                                    cx = (pts2d[c, sz, m, n, 0] - fi) / kws
                                    cy = (pts2d[c, sz, m, n, 1] - fj) / kws
                                    expx = inv_sigmaM[c,sz,m,n,0,0] * cx * cx + inv_sigmaM[c,sz,m,n,1,0] * cx * cy + inv_sigmaM[c,sz,m,n,0,1] * cx * cy + inv_sigmaM[c,sz,m,n,1,1] * cy * cy
                                    expv = np.exp(-expx / 2)
                                    rimg[c, sz, j, i] = rimg[c, sz, j, i] + expv / 2 / np.pi
                                    counter[c, sz, j, i] = counter[c, sz, j, i] + 1

    for c in range(bs):
        for sz in range(samplesz):
            for m in range(height):
                for n in range(width):
                    rimg[c, sz, m, n] = rimg[c, sz, m, n] / (counter[c, sz, m, n] + eps)

    # Backward
    for c in range(bs):
        for sz in range(samplesz):
            m11 = Pcombinednp[c, sz, 0, 0]
            m12 = Pcombinednp[c, sz, 0, 1]
            m13 = Pcombinednp[c, sz, 0, 2]
            m14 = Pcombinednp[c, sz, 0, 3]

            m21 = Pcombinednp[c, sz, 1, 0]
            m22 = Pcombinednp[c, sz, 1, 1]
            m23 = Pcombinednp[c, sz, 1, 2]
            m24 = Pcombinednp[c, sz, 1, 3]

            m31 = Pcombinednp[c, sz, 2, 0]
            m32 = Pcombinednp[c, sz, 2, 1]
            m33 = Pcombinednp[c, sz, 2, 2]
            m34 = Pcombinednp[c, sz, 2, 3]
            for m in range(height):
                for n in range(width):
                    D = depthmapnp[c, 0, m, n]
                    y = float(m)
                    x = float(n)
                    gradPxDep = (m11 * x + m12 * y + m13) / (m31 * x * D + m32 * y * D + m33 * D + m34) \
                                -(m11 * x * D + m12 * y * D + m13 * D + m14) / ((m31 * x * D + m32 * y * D + m33 * D + m34)**2) * (m31 * x + m32 * y + m33)
                    gradPyDep = (m21 * x + m22 * y + m23) / (m31 * x * D + m32 * y * D + m33 * D + m34) \
                                -(m21 * x * D + m22 * y * D + m23 * D + m24) / ((m31 * x * D + m32 * y * D + m33 * D + m34)**2) * (m31 * x + m32 * y + m33)
                    if mask[c, sz, 0, m, n] > eps:
                        ctx = int(np.round_(pts2d[c, sz, m, n, 0]))
                        cty = int(np.round_(pts2d[c, sz, m, n, 1]))
                        for i in range(ctx - srhalf, ctx + srhalf + 1):
                            for j in range(cty - srhalf, cty + srhalf + 1):
                                if i >= 0 and i < width and j >= 0 and j < height:
                                    fi = float(i)
                                    fj = float(j)
                                    cx = (pts2d[c, sz, m, n, 0] - fi) / kws
                                    cy = (pts2d[c, sz, m, n, 1] - fj) / kws
                                    expx = inv_sigmaM[c,sz,m,n,0,0] * cx * cx + inv_sigmaM[c,sz,m,n,1,0] * cx * cy + inv_sigmaM[c,sz,m,n,0,1] * cx * cy + inv_sigmaM[c,sz,m,n,1,1] * cy * cy
                                    expv = np.exp(-expx / 2) / 2 / np.pi / (counter[c, sz, j, i] + eps)

                                    tmpk = expv / (-2) * (2 * (rimg[c, sz, j, i] - rimg_gt[c, sz, j, i]))
                                    # tmpk = expv / (-2) * (2 * rimg[c, sz, j, i])
                                    # tmpk = expv / (-2) / (rimg[c, sz, j, i] + 10)
                                    # tmpk = expv / (-2)
                                    tmpx = tmpk * (2 * inv_sigmaM[c,sz,m,n,0,0] * cx + inv_sigmaM[c,sz,m,n,1,0] * cy + inv_sigmaM[c,sz,m,n,0,1] * cy)
                                    grad2d[c, sz, 0, m, n] = grad2d[c, sz, 0, m, n] + tmpx / kws
                                    tmpy = tmpk * (2 * inv_sigmaM[c,sz,m,n,1,1] * cy + inv_sigmaM[c,sz,m,n,1,0] * cx + inv_sigmaM[c,sz,m,n,0,1] * cx)
                                    grad2d[c, sz, 1, m, n] = grad2d[c, sz, 1, m, n] + tmpy / kws

                                    depthmapnp_grad[c, 0, m, n] = depthmapnp_grad[c, 0, m, n] + \
                                                                  tmpk / kws * ((2 * inv_sigmaM[c,sz,m,n,0,0] * cx + inv_sigmaM[c,sz,m,n,1,0] * cy + inv_sigmaM[c,sz,m,n,0,1] * cy) * gradPxDep + (2 * inv_sigmaM[c,sz,m,n,1,1] * cy + inv_sigmaM[c,sz,m,n,1,0] * cx + inv_sigmaM[c,sz,m,n,0,1] * cx) * gradPyDep)

    return rimg, grad2d, counter, depthmapnp_grad




@jit(nopython=True, parallel=True)
def eppl_render_l1(inv_sigmaM, pts2d, mask, Pcombinednp, depthmapnp, rimg_gt, kws, sr, bs, samplesz, height, width):
    eps = 1e-6
    srhalf = int((sr - 1) / 2)
    rimg = np.zeros((bs, samplesz, height, width), dtype=np.float32)
    counter = np.zeros((bs, samplesz, height, width), dtype=np.float32)
    grad2d = np.zeros((bs, samplesz, 2, height, width), dtype=np.float32)
    depthmapnp_grad = np.zeros((bs, 1, height, width), dtype=np.float32)
    for c in range(bs):
        for sz in range(samplesz):
            for m in range(height):
                for n in range(width):
                    if mask[c, sz, 0, m, n] > eps:
                        ctx = int(np.round_(pts2d[c, sz, m, n, 0]))
                        cty = int(np.round_(pts2d[c, sz, m, n, 1]))
                        for i in range(ctx - srhalf, ctx + srhalf + 1):
                            for j in range(cty - srhalf, cty + srhalf + 1):
                                if i >= 0 and i < width and j >= 0 and j < height:
                                    fi = float(i)
                                    fj = float(j)
                                    cx = (pts2d[c, sz, m, n, 0] - fi) / kws
                                    cy = (pts2d[c, sz, m, n, 1] - fj) / kws
                                    expx = inv_sigmaM[c,sz,m,n,0,0] * cx * cx + inv_sigmaM[c,sz,m,n,1,0] * cx * cy + inv_sigmaM[c,sz,m,n,0,1] * cx * cy + inv_sigmaM[c,sz,m,n,1,1] * cy * cy
                                    expv = np.exp(-expx / 2)
                                    rimg[c, sz, j, i] = rimg[c, sz, j, i] + expv / 2 / np.pi
                                    counter[c, sz, j, i] = counter[c, sz, j, i] + 1

    for c in range(bs):
        for sz in range(samplesz):
            for m in range(height):
                for n in range(width):
                    rimg[c, sz, m, n] = rimg[c, sz, m, n] / (counter[c, sz, m, n] + eps)

    # Backward
    for c in range(bs):
        for sz in range(samplesz):
            m11 = Pcombinednp[c, sz, 0, 0]
            m12 = Pcombinednp[c, sz, 0, 1]
            m13 = Pcombinednp[c, sz, 0, 2]
            m14 = Pcombinednp[c, sz, 0, 3]

            m21 = Pcombinednp[c, sz, 1, 0]
            m22 = Pcombinednp[c, sz, 1, 1]
            m23 = Pcombinednp[c, sz, 1, 2]
            m24 = Pcombinednp[c, sz, 1, 3]

            m31 = Pcombinednp[c, sz, 2, 0]
            m32 = Pcombinednp[c, sz, 2, 1]
            m33 = Pcombinednp[c, sz, 2, 2]
            m34 = Pcombinednp[c, sz, 2, 3]
            for m in range(height):
                for n in range(width):
                    D = depthmapnp[c, 0, m, n]
                    y = float(m)
                    x = float(n)
                    gradPxDep = (m11 * x + m12 * y + m13) / (m31 * x * D + m32 * y * D + m33 * D + m34) \
                                -(m11 * x * D + m12 * y * D + m13 * D + m14) / ((m31 * x * D + m32 * y * D + m33 * D + m34)**2) * (m31 * x + m32 * y + m33)
                    gradPyDep = (m21 * x + m22 * y + m23) / (m31 * x * D + m32 * y * D + m33 * D + m34) \
                                -(m21 * x * D + m22 * y * D + m23 * D + m24) / ((m31 * x * D + m32 * y * D + m33 * D + m34)**2) * (m31 * x + m32 * y + m33)
                    if mask[c, sz, 0, m, n] > eps:
                        ctx = int(np.round_(pts2d[c, sz, m, n, 0]))
                        cty = int(np.round_(pts2d[c, sz, m, n, 1]))
                        for i in range(ctx - srhalf, ctx + srhalf + 1):
                            for j in range(cty - srhalf, cty + srhalf + 1):
                                if i >= 0 and i < width and j >= 0 and j < height:
                                    fi = float(i)
                                    fj = float(j)
                                    cx = (pts2d[c, sz, m, n, 0] - fi) / kws
                                    cy = (pts2d[c, sz, m, n, 1] - fj) / kws
                                    expx = inv_sigmaM[c,sz,m,n,0,0] * cx * cx + inv_sigmaM[c,sz,m,n,1,0] * cx * cy + inv_sigmaM[c,sz,m,n,0,1] * cx * cy + inv_sigmaM[c,sz,m,n,1,1] * cy * cy
                                    expv = np.exp(-expx / 2) / 2 / np.pi / (counter[c, sz, j, i] + eps)

                                    if (rimg[c, sz, j, i] - rimg_gt[c, sz, j, i]) > 0:
                                        tmpk = expv / (-2) * 1
                                    else:
                                        tmpk = expv / (-2) * (-1)
                                    # tmpk = expv / (-2) * (2 * rimg[c, sz, j, i])
                                    # tmpk = expv / (-2) / (rimg[c, sz, j, i] + 10)
                                    # tmpk = expv / (-2)
                                    tmpx = tmpk * (2 * inv_sigmaM[c,sz,m,n,0,0] * cx + inv_sigmaM[c,sz,m,n,1,0] * cy + inv_sigmaM[c,sz,m,n,0,1] * cy)
                                    grad2d[c, sz, 0, m, n] = grad2d[c, sz, 0, m, n] + tmpx / kws
                                    tmpy = tmpk * (2 * inv_sigmaM[c,sz,m,n,1,1] * cy + inv_sigmaM[c,sz,m,n,1,0] * cx + inv_sigmaM[c,sz,m,n,0,1] * cx)
                                    grad2d[c, sz, 1, m, n] = grad2d[c, sz, 1, m, n] + tmpy / kws

                                    depthmapnp_grad[c, 0, m, n] = depthmapnp_grad[c, 0, m, n] + \
                                                                  tmpk / kws * ((2 * inv_sigmaM[c,sz,m,n,0,0] * cx + inv_sigmaM[c,sz,m,n,1,0] * cy + inv_sigmaM[c,sz,m,n,0,1] * cy) * gradPxDep + (2 * inv_sigmaM[c,sz,m,n,1,1] * cy + inv_sigmaM[c,sz,m,n,1,0] * cx + inv_sigmaM[c,sz,m,n,0,1] * cx) * gradPyDep)

    return rimg, grad2d, counter, depthmapnp_grad




@jit(nopython=True, parallel=True)
def eppl_render_l1_sfgrad(inv_sigmaM, pts2d, mask, Pcombinednp, depthmapnp, rimg_gt, kws, sr, bs, samplesz, height, width):
    eps = 1e-6
    srhalf = int((sr - 1) / 2)
    rimg = np.zeros((bs, samplesz, height, width), dtype=np.float32)
    counter = np.zeros((bs, samplesz, height, width), dtype=np.float32)
    grad2d = np.zeros((bs, samplesz, 2, height, width), dtype=np.float32)
    depthmapnp_grad = np.zeros((bs, 1, height, width), dtype=np.float32)
    for c in range(bs):
        for sz in range(samplesz):
            for m in range(height):
                for n in range(width):
                    if mask[c, sz, 0, m, n] > eps:
                        ctx = int(np.round_(pts2d[c, sz, m, n, 0]))
                        cty = int(np.round_(pts2d[c, sz, m, n, 1]))
                        for i in range(ctx - srhalf, ctx + srhalf + 1):
                            for j in range(cty - srhalf, cty + srhalf + 1):
                                if i >= 0 and i < width and j >= 0 and j < height:
                                    fi = float(i)
                                    fj = float(j)
                                    cx = (pts2d[c, sz, m, n, 0] - fi) / kws
                                    cy = (pts2d[c, sz, m, n, 1] - fj) / kws
                                    expx = inv_sigmaM[c,sz,m,n,0,0] * cx * cx + inv_sigmaM[c,sz,m,n,1,0] * cx * cy + inv_sigmaM[c,sz,m,n,0,1] * cx * cy + inv_sigmaM[c,sz,m,n,1,1] * cy * cy
                                    expv = np.exp(-expx / 2)
                                    rimg[c, sz, j, i] = rimg[c, sz, j, i] + expv / 2 / np.pi
                                    counter[c, sz, j, i] = counter[c, sz, j, i] + 1

    for c in range(bs):
        for sz in range(samplesz):
            for m in range(height):
                for n in range(width):
                    rimg[c, sz, m, n] = rimg[c, sz, m, n] / (counter[c, sz, m, n] + eps)

    # Backward
    for c in range(bs):
        for sz in range(samplesz):
            m11 = Pcombinednp[c, sz, 0, 0]
            m12 = Pcombinednp[c, sz, 0, 1]
            m13 = Pcombinednp[c, sz, 0, 2]
            m14 = Pcombinednp[c, sz, 0, 3]

            m21 = Pcombinednp[c, sz, 1, 0]
            m22 = Pcombinednp[c, sz, 1, 1]
            m23 = Pcombinednp[c, sz, 1, 2]
            m24 = Pcombinednp[c, sz, 1, 3]

            m31 = Pcombinednp[c, sz, 2, 0]
            m32 = Pcombinednp[c, sz, 2, 1]
            m33 = Pcombinednp[c, sz, 2, 2]
            m34 = Pcombinednp[c, sz, 2, 3]
            for m in range(height):
                for n in range(width):
                    D = depthmapnp[c, 0, m, n]
                    y = float(m)
                    x = float(n)
                    gradPxDep = (m11 * x + m12 * y + m13) / (m31 * x * D + m32 * y * D + m33 * D + m34) \
                                -(m11 * x * D + m12 * y * D + m13 * D + m14) / ((m31 * x * D + m32 * y * D + m33 * D + m34)**2) * (m31 * x + m32 * y + m33)
                    gradPyDep = (m21 * x + m22 * y + m23) / (m31 * x * D + m32 * y * D + m33 * D + m34) \
                                -(m21 * x * D + m22 * y * D + m23 * D + m24) / ((m31 * x * D + m32 * y * D + m33 * D + m34)**2) * (m31 * x + m32 * y + m33)
                    if mask[c, sz, 0, m, n] > eps:
                        ctx = int(np.round_(pts2d[c, sz, m, n, 0]))
                        cty = int(np.round_(pts2d[c, sz, m, n, 1]))
                        for i in range(ctx - srhalf, ctx + srhalf + 1):
                            for j in range(cty - srhalf, cty + srhalf + 1):
                                if i >= 0 and i < width and j >= 0 and j < height:
                                    if np.abs(rimg_gt[c, sz, j, i]) > eps:
                                        fi = float(i)
                                        fj = float(j)
                                        cx = (pts2d[c, sz, m, n, 0] - fi) / kws
                                        cy = (pts2d[c, sz, m, n, 1] - fj) / kws
                                        expx = inv_sigmaM[c,sz,m,n,0,0] * cx * cx + inv_sigmaM[c,sz,m,n,1,0] * cx * cy + inv_sigmaM[c,sz,m,n,0,1] * cx * cy + inv_sigmaM[c,sz,m,n,1,1] * cy * cy
                                        expv = np.exp(-expx / 2) / 2 / np.pi / (counter[c, sz, j, i] + eps)

                                        if (rimg[c, sz, j, i] - rimg_gt[c, sz, j, i]) > 0:
                                            tmpk = expv / (-2) * 1
                                        else:
                                            tmpk = expv / (-2) * (-1)
                                        # tmpk = expv / (-2) * (2 * rimg[c, sz, j, i])
                                        # tmpk = expv / (-2) / (rimg[c, sz, j, i] + 10)
                                        # tmpk = expv / (-2)
                                        tmpx = tmpk * (2 * inv_sigmaM[c,sz,m,n,0,0] * cx + inv_sigmaM[c,sz,m,n,1,0] * cy + inv_sigmaM[c,sz,m,n,0,1] * cy)
                                        grad2d[c, sz, 0, m, n] = grad2d[c, sz, 0, m, n] + tmpx / kws
                                        tmpy = tmpk * (2 * inv_sigmaM[c,sz,m,n,1,1] * cy + inv_sigmaM[c,sz,m,n,1,0] * cx + inv_sigmaM[c,sz,m,n,0,1] * cx)
                                        grad2d[c, sz, 1, m, n] = grad2d[c, sz, 1, m, n] + tmpy / kws

                                        depthmapnp_grad[c, 0, m, n] = depthmapnp_grad[c, 0, m, n] + \
                                                                      tmpk / kws * ((2 * inv_sigmaM[c,sz,m,n,0,0] * cx + inv_sigmaM[c,sz,m,n,1,0] * cy + inv_sigmaM[c,sz,m,n,0,1] * cy) * gradPxDep + (2 * inv_sigmaM[c,sz,m,n,1,1] * cy + inv_sigmaM[c,sz,m,n,1,0] * cx + inv_sigmaM[c,sz,m,n,0,1] * cx) * gradPyDep)

    return rimg, grad2d, counter, depthmapnp_grad