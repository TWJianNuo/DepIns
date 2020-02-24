import numpy as np
import numba
from numba import jit

@jit(nopython=True, parallel=True)
def eppl_render(inv_sigmaM, pts2d, mask, kws, sr, bs, samplesz, height, width):
    eps = 1e-6
    srhalf = int((sr - 1) / 2)
    rimg = np.zeros((bs, samplesz, height, width), dtype=np.float32)
    counter = np.zeros((bs, samplesz, height, width), dtype=np.float32)
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
    return rimg