from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from torch.utils.data import DataLoader

import torch
from Exp_PreSIL.dataloader_kitti import KittiDataset

import networks

import time
import json

from layers import *
from networks import *

import argparse

default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",                  type=str,                               help="path to dataset")
parser.add_argument("--gt_path",                    type=str,                               help="path to kitti gt file")
parser.add_argument("--predang_path",               type=str,                               help="path to kitti gt file")
parser.add_argument("--semanticspred_path",         type=str,   default='None',             help="path to kitti semantics prediction file")
parser.add_argument("--val_gt_path",                type=str,                               help="path to validation gt file")
parser.add_argument("--split",                      type=str,                               help="train/val split to use")
parser.add_argument("--num_layers",                 type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                     type=int,   default=320,                help="input image height")
parser.add_argument("--width",                      type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                       type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                       type=int,   default=1220,               help="cropped image width")
parser.add_argument("--min_depth",                  type=float, default=0.1,                help="minimum depth")
parser.add_argument("--max_depth",                  type=float, default=100.0,              help="maximum depth")
parser.add_argument("--load_angweights_folder",     type=str,                               help="path to kitti gt file")
parser.add_argument("--load_depthweights_folder",   type=str,                               help="path to kitti gt file")
parser.add_argument("--load_angErr_folder",     type=str,                               help="path to kitti gt file")
parser.add_argument("--load_depthErr_folder",   type=str,                               help="path to kitti gt file")

# OPTIMIZATION options
parser.add_argument("--batch_size",                 type=int,   default=12,                 help="batch size")
parser.add_argument("--num_workers",                type=int,   default=6,                  help="number of dataloader workers")

@numba.jit(nopython=True, parallel=True)
def dynamicReceptiveField(height, width, err_shapenph, err_shapenpv, err_depthnph, err_depthnpv, sw, sh):
    hrecord = np.zeros((height, width))
    vrecord = np.zeros((height, width))

    for m in range(height):
        for n in range(width):
            hgood = 0
            vgood = 0

            intl = 0
            for lx in range(sw):
                ckx = n - (lx + 1)
                if ckx >= 0:
                    intl = intl + err_shapenph[m, ckx]
                    refl = err_depthnph[m, n] + err_depthnph[m, n - (lx + 1)]
                    if intl <= refl:
                        hgood = hgood + 1
                    else:
                        break

            intr = 0
            for rx in range(sw):
                ckx = n + (rx + 1)
                if ckx < width:
                    intr = intr + err_shapenph[m, ckx - 1]
                    refr = err_depthnph[m, n] + err_depthnph[m, n + (rx + 1)]
                    if intr <= refr:
                        hgood = hgood + 1
                    else:
                        break

            intu = 0
            for ru in range(sh):
                cky = m - (ru + 1)
                if cky >= 0:
                    intu = intu + err_shapenpv[cky, n]
                    refu = err_depthnpv[m, n] + err_depthnpv[m - (ru + 1), n]
                    if intu <= refu:
                        vgood = vgood + 1
                    else:
                        break

            intd = 0
            for rd in range(sh):
                cky = m + (rd + 1)
                if cky < height:
                    intd = intd + err_shapenpv[cky - 1, n]
                    refd = err_depthnpv[m, n] + err_depthnpv[m + (rd + 1), n]
                    if intd <= refd:
                        vgood = vgood + 1
                    else:
                        break

            hrecord[m,n] = hgood
            vrecord[m,n] = vgood
    return hrecord, vrecord

def dynamicReceptiveFieldFromShape(anghnp, angvnp, w, h, m, n, barh, barv):
    counth = 1
    sumh = anghnp[m, n]
    sumsquareh = anghnp[m, n] ** 2
    ptsh = [np.array([n, m])]
    for i in range(1, 2 * w):
        if np.mod(i, 2) == 1:
            incn = -int((i + 1) / 2)
        else:
            incn = int(i / 2)

        searchn = n + incn
        if searchn >= 0 and searchn < w:
            sumh = sumh + anghnp[m, searchn]
            sumsquareh = sumsquareh + anghnp[m, searchn] ** 2
            counth = counth + 1
            varh = sumsquareh - sumh ** 2 / counth
            if varh < barh:
                ptsh.append(np.array([searchn, m]))
            else:
                break

    countv = 1
    sumv = angvnp[m, n]
    sumsquarev = angvnp[m, n] ** 2
    ptsv = [np.array([n, m])]
    for j in range(1, 2 * h):
        if np.mod(j, 2) == 1:
            incm = -int((j + 1) / 2)
        else:
            incm = int(j / 2)

        searchm = m + incm
        if searchm >= 0 and searchm < h:
            sumv = sumv + angvnp[searchm, n]
            sumsquarev = sumsquarev + angvnp[searchm, n] ** 2
            countv = countv + 1
            varv = sumsquarev - sumv ** 2 / countv
            if varv < barv:
                ptsv.append(np.array([n, searchm]))

    ptsh = np.array(ptsh)
    ptsv = np.array(ptsv)
    return ptsh, ptsv

@numba.jit(nopython=True, parallel=True)
def evaluateReceptiveFieldSemantic(height, width, logh, logv, depthpredl, depthgtl, mask, semantics, srh, srv):
    shapeErrRech_tot = np.zeros((srh))
    depthErrRech_tot = np.zeros((srh))
    errRech_num = np.zeros((srh))
    shapeErrRecv_tot = np.zeros((srv))
    depthErrRecv_tot = np.zeros((srv))
    errRecv_num = np.zeros((srv))
    for m in range(height):
        for n in range(width):
            if mask[m, n]:
                refs = semantics[m, n]
                inth = 0
                for sx in range(n, width-1):
                    if sx - n >= srh:
                        break
                    if semantics[m, sx + 1] != refs:
                        break
                    inth = inth + logh[m, sx]
                    if mask[m, sx + 1]:
                        depthh = depthpredl[m, sx + 1] - depthpredl[m, n]
                        gth = depthgtl[m, sx + 1] - depthgtl[m, n]
                        shapeErrRech_tot[sx - n] = shapeErrRech_tot[sx - n] + np.abs(inth - gth)
                        depthErrRech_tot[sx - n] = depthErrRech_tot[sx - n] + np.abs(depthh - gth)
                        errRech_num[sx - n] = errRech_num[sx - n] + 1

                intv = 0
                for sy in range(m, height-1):
                    if sy - m >= srv:
                        break
                    if semantics[sy + 1, n] != refs:
                        break
                    intv = intv + logv[sy, n]
                    if mask[sy + 1, n]:
                        depthv = depthpredl[sy + 1, n] - depthpredl[m, n]
                        gtv = depthgtl[sy + 1, n] - depthgtl[m, n]
                        shapeErrRecv_tot[sy - m] = shapeErrRecv_tot[sy - m] + np.abs(intv - gtv)
                        depthErrRecv_tot[sy - m] = depthErrRecv_tot[sy - m] + np.abs(depthv - gtv)
                        errRecv_num[sy - m] = errRecv_num[sy - m] + 1

    return shapeErrRech_tot, depthErrRech_tot, errRech_num, shapeErrRecv_tot, depthErrRecv_tot, errRecv_num

@numba.jit(nopython=True, parallel=True)
def evaluateReceptiveFieldVariance(height, width, logh, logv, depthpredl, depthgtl, shapeh, shapev, mask, semanticmask, recnum, maxvar):
    shapeErrh = np.zeros(recnum)
    depthErrh = np.zeros(recnum)
    rangeh = np.zeros(recnum)
    numh = np.zeros(recnum)

    shapeErrv = np.zeros(recnum)
    depthErrv = np.zeros(recnum)
    rangev = np.zeros(recnum)
    numv = np.zeros(recnum)

    for m in range(height):
        for n in range(width):
            if mask[m, n]:
                inthr = 0
                inthl = 0
                counth = 1

                squaresumh = shapeh[m, n] ** 2
                sumh = shapeh[m, n]

                breakl = False
                breakr = False
                for ih in range(width * 2):
                    if ih % 2 == 0:
                        if breakr:
                            continue
                        sn = n + int(ih / 2) + 1

                        if sn >= width:
                            breakr = True
                            continue
                        elif not semanticmask[m, sn]:
                            breakr = True
                            continue
                        else:
                            inthr = inthr + logh[m, sn-1]
                            squaresumh = squaresumh + shapeh[m, sn-1]**2
                            sumh = sumh + shapeh[m, sn-1]
                            counth = counth + 1

                            if mask[m, sn]:
                                # varh = (squaresumh / counth - (sumh / counth)**2)
                                varh = (squaresumh - sumh ** 2 / counth)
                                recind = int(varh / maxvar * recnum)
                                recind = min(recind, recnum-1)

                                predh = depthpredl[m, sn] - depthpredl[m, n]
                                gth = depthgtl[m, sn] - depthgtl[m, n]

                                shapeErrh[recind] = shapeErrh[recind] + np.abs(inthr - gth)
                                depthErrh[recind] = depthErrh[recind] + np.abs(predh - gth)
                                rangeh[recind] = rangeh[recind] + counth
                                numh[recind] = numh[recind] + 1
                    else:
                        if breakl:
                            continue
                        sn = n - int(ih / 2) - 1
                        if sn < 0:
                            breakl = True
                            continue
                        elif not semanticmask[m, sn]:
                            breakl = True
                            continue
                        else:
                            inthl = inthl - logh[m, sn]
                            squaresumh = squaresumh + shapeh[m, sn]**2
                            sumh = sumh + shapeh[m, sn]
                            counth = counth + 1

                            if mask[m, sn]:
                                varh = (squaresumh - sumh ** 2 / counth)
                                recind = int(varh / maxvar * recnum)
                                recind = min(recind, recnum-1)

                                predh = depthpredl[m, sn] - depthpredl[m, n]
                                gth = depthgtl[m, sn] - depthgtl[m, n]

                                shapeErrh[recind] = shapeErrh[recind] + np.abs(inthl - gth)
                                depthErrh[recind] = depthErrh[recind] + np.abs(predh - gth)
                                rangeh[recind] = rangeh[recind] + counth
                                numh[recind] = numh[recind] + 1

                intvu = 0
                intvd = 0
                countv = 1

                squaresumv = shapev[m, n] ** 2
                sumv = shapev[m, n]

                breaku = False
                breakd = False
                for iv in range(height * 2):
                    if iv % 2 == 0:
                        if breakd:
                            continue
                        sm = m + int(iv / 2) + 1
                        if sm >= height:
                            breakd = True
                            continue
                        elif not semanticmask[sm, n]:
                            breakd = True
                            continue
                        else:
                            intvd = intvd + logv[sm-1, n]
                            squaresumv = squaresumv + shapev[sm-1, n]**2
                            sumv = sumv + shapev[sm-1, n]
                            countv = countv + 1

                            if mask[sm, n]:
                                varv = (squaresumv - sumv ** 2 / countv)
                                recind = int(varv / maxvar * recnum)
                                recind = min(recind, recnum-1)

                                predv = depthpredl[sm, n] - depthpredl[m, n]
                                gtv = depthgtl[sm, n] - depthgtl[m, n]

                                shapeErrv[recind] = shapeErrv[recind] + np.abs(intvd - gtv)
                                depthErrv[recind] = depthErrv[recind] + np.abs(predv - gtv)
                                rangev[recind] = rangev[recind] + countv
                                numv[recind] = numv[recind] + 1
                    else:
                        if breaku:
                            continue
                        sm = m - int(iv / 2) - 1
                        if sm < 0:
                            breaku = True
                            continue
                        elif not semanticmask[sm, n]:
                            breaku = True
                            continue
                        else:
                            intvu = intvu - logv[sm, n]
                            squaresumv = squaresumv + shapev[sm, n]**2
                            sumv = sumv + shapev[sm, n]
                            countv = countv + 1

                            if mask[sm, n]:
                                varv = (squaresumv - sumv ** 2 / countv)
                                recind = int(varv / maxvar * recnum)
                                recind = min(recind, recnum-1)

                                predv = depthpredl[sm, n] - depthpredl[m, n]
                                gtv = depthgtl[sm, n] - depthgtl[m, n]

                                shapeErrv[recind] = shapeErrv[recind] + np.abs(intvu - gtv)
                                depthErrv[recind] = depthErrv[recind] + np.abs(predv - gtv)
                                rangev[recind] = rangev[recind] + countv
                                numv[recind] = numv[recind] + 1
    return shapeErrh, depthErrh, rangeh, numh, shapeErrv, depthErrv, rangev, numv

@numba.jit(nopython=True, parallel=True)
def evaluateReceptiveFieldTrueVariance(height, width, logh, logv, depthpredl, depthgtl, shapeh, shapev, mask, semanticmask, recnum, maxvar):
    shapeErrh = np.zeros(recnum)
    depthErrh = np.zeros(recnum)
    rangeh = np.zeros(recnum)
    numh = np.zeros(recnum)

    shapeErrv = np.zeros(recnum)
    depthErrv = np.zeros(recnum)
    rangev = np.zeros(recnum)
    numv = np.zeros(recnum)

    for m in range(height):
        for n in range(width):
            if mask[m, n]:
                inthr = 0
                inthl = 0
                counth = 1

                squaresumh = shapeh[m, n] ** 2
                sumh = shapeh[m, n]

                breakl = False
                breakr = False
                for ih in range(width * 2):
                    if ih % 2 == 0:
                        if breakr:
                            continue
                        sn = n + int(ih / 2) + 1
                        if sn >= width:
                            breakr = True
                            continue
                        elif not semanticmask[m, sn]:
                            breakr = True
                            continue
                        else:
                            inthr = inthr + logh[m, sn-1]
                            squaresumh = squaresumh + shapeh[m, sn]**2
                            sumh = sumh + shapeh[m, sn]
                            counth = counth + 1

                            if mask[m, sn]:
                                varh = (squaresumh / counth - (sumh / counth) ** 2)
                                recind = int(varh / maxvar * recnum)
                                recind = min(recind, recnum-1)

                                predh = depthpredl[m, sn] - depthpredl[m, n]
                                gth = depthgtl[m, sn] - depthgtl[m, n]

                                shapeErrh[recind] = shapeErrh[recind] + np.abs(inthr - gth)
                                depthErrh[recind] = depthErrh[recind] + np.abs(predh - gth)
                                rangeh[recind] = rangeh[recind] + counth
                                numh[recind] = numh[recind] + 1
                    else:
                        if breakl:
                            continue
                        sn = n - int(ih / 2) - 1
                        if sn < 0:
                            breakl = True
                            continue
                        elif not semanticmask[m, sn]:
                            breakl = True
                            continue
                        else:
                            inthl = inthl - logh[m, sn]
                            squaresumh = squaresumh + shapeh[m, sn]**2
                            sumh = sumh + shapeh[m, sn]
                            counth = counth + 1

                            if mask[m, sn]:
                                varh = (squaresumh / counth - (sumh / counth) ** 2)
                                recind = int(varh / maxvar * recnum)
                                recind = min(recind, recnum-1)

                                predh = depthpredl[m, sn] - depthpredl[m, n]
                                gth = depthgtl[m, sn] - depthgtl[m, n]

                                shapeErrh[recind] = shapeErrh[recind] + np.abs(inthl - gth)
                                depthErrh[recind] = depthErrh[recind] + np.abs(predh - gth)
                                rangeh[recind] = rangeh[recind] + counth
                                numh[recind] = numh[recind] + 1

                intvu = 0
                intvd = 0
                countv = 1

                squaresumv = shapev[m, n] ** 2
                sumv = shapev[m, n]

                breaku = False
                breakd = False
                for iv in range(height * 2):
                    if iv % 2 == 0:
                        if breakd:
                            continue
                        sm = m + int(iv / 2) + 1
                        if sm >= height:
                            breakd = True
                            continue
                        elif not semanticmask[sm, n]:
                            breakd = True
                            continue
                        else:
                            intvd = intvd + logv[sm-1, n]
                            squaresumv = squaresumv + shapev[sm, n]**2
                            sumv = sumv + shapev[sm, n]
                            countv = countv + 1

                            if mask[sm, n]:
                                varv = (squaresumv / countv - (sumv / countv) ** 2)
                                recind = int(varv / maxvar * recnum)
                                recind = min(recind, recnum-1)

                                predv = depthpredl[sm, n] - depthpredl[m, n]
                                gtv = depthgtl[sm, n] - depthgtl[m, n]

                                shapeErrv[recind] = shapeErrv[recind] + np.abs(intvd - gtv)
                                depthErrv[recind] = depthErrv[recind] + np.abs(predv - gtv)
                                rangev[recind] = rangev[recind] + countv
                                numv[recind] = numv[recind] + 1
                    else:
                        if breaku:
                            continue
                        sm = m - int(iv / 2) - 1
                        if sm < 0:
                            breaku = True
                            continue
                        elif not semanticmask[sm, n]:
                            breaku = True
                            continue
                        else:
                            intvu = intvu - logv[sm, n]
                            squaresumv = squaresumv + shapev[sm, n]**2
                            sumv = sumv + shapev[sm, n]
                            countv = countv + 1

                            if mask[sm, n]:
                                varv = (squaresumv / countv - (sumv / countv) ** 2)
                                recind = int(varv / maxvar * recnum)
                                recind = min(recind, recnum-1)

                                predv = depthpredl[sm, n] - depthpredl[m, n]
                                gtv = depthgtl[sm, n] - depthgtl[m, n]

                                shapeErrv[recind] = shapeErrv[recind] + np.abs(intvu - gtv)
                                depthErrv[recind] = depthErrv[recind] + np.abs(predv - gtv)
                                rangev[recind] = rangev[recind] + countv
                                numv[recind] = numv[recind] + 1
    return shapeErrh, depthErrh, rangeh, numh, shapeErrv, depthErrv, rangev, numv

def get_dynamicRange(height, width, shapeh, shapev, semanticmask, barh, barv, m, n):
    pts = list()

    counth = 1
    squaresumh = shapeh[m, n] ** 2
    sumh = shapeh[m, n]

    breakl = False
    breakr = False
    for ih in range(width * 2):
        if ih % 2 == 0:
            if breakr:
                continue
            sn = n + int(ih / 2) + 1

            if sn >= width:
                breakr = True
                continue
            elif not semanticmask[m, sn]:
                breakr = True
                continue
            else:
                squaresumh = squaresumh + shapeh[m, sn-1]**2
                sumh = sumh + shapeh[m, sn-1]
                counth = counth + 1
                varh = (squaresumh / counth - (sumh / counth) ** 2)
                if varh < barh:
                    pts.append(np.array([sn, m]))
                else:
                    breakr = True
        else:
            if breakl:
                continue
            sn = n - int(ih / 2) - 1
            if sn < 0:
                breakl = True
                continue
            elif not semanticmask[m, sn]:
                breakl = True
                continue
            else:
                squaresumh = squaresumh + shapeh[m, sn]**2
                sumh = sumh + shapeh[m, sn]
                counth = counth + 1
                varh = (squaresumh / counth - (sumh / counth) ** 2)
                if varh < barh:
                    pts.append(np.array([sn, m]))
                else:
                    breakl = True

    countv = 1
    squaresumv = shapev[m, n] ** 2
    sumv = shapev[m, n]

    breaku = False
    breakd = False
    for iv in range(height * 2):
        if iv % 2 == 0:
            if breakd:
                continue
            sm = m + int(iv / 2) + 1
            if sm >= height:
                breakd = True
                continue
            elif not semanticmask[sm, n]:
                breakd = True
                continue
            else:
                squaresumv = squaresumv + shapev[sm-1, n]**2
                sumv = sumv + shapev[sm-1, n]
                countv = countv + 1
                varv = (squaresumv / countv - (sumv / countv) ** 2)
                if varv < barv:
                    pts.append(np.array([n, sm]))
                else:
                    breakd = True
        else:
            if breaku:
                continue
            sm = m - int(iv / 2) - 1
            if sm < 0:
                breaku = True
                continue
            elif not semanticmask[sm, n]:
                breaku = True
                continue
            else:
                squaresumv = squaresumv + shapev[sm, n]**2
                sumv = sumv + shapev[sm, n]
                countv = countv + 1
                varv = (squaresumv / countv - (sumv / countv) ** 2)
                if varv < barv:
                    pts.append(np.array([n, sm]))
                else:
                    breaku = True
    return pts

def progressBar(current, total, catname, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('%s progress: [%s%s] %d %%' % (catname, arrow, spaces, percent), end='\r')

class Trainer:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.device = "cuda"

        self.depthmodels = {}
        self.depthmodels["depthencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=False, num_input_channels=3)
        self.depthmodels["depthdecoder"] = DepthDecoder(self.depthmodels["depthencoder"].num_ch_enc, num_output_channels=1)
        self.depthmodels["depthencoder"].to(self.device)
        self.depthmodels["depthdecoder"].to(self.device)
        self.load_model(weightFolder=self.opt.load_depthweights_folder, encoderName='depthencoder',
                        decoderName='depthdecoder', encoder=self.depthmodels["depthencoder"],
                        decoder=self.depthmodels["depthdecoder"])
        for m in self.depthmodels.values():
            m.eval()

        self.angmodels = {}
        self.angmodels["angencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=False)
        self.angmodels["angdecoder"] = DepthDecoder(self.angmodels["angencoder"].num_ch_enc, num_output_channels=2)
        self.angmodels["angencoder"].to(self.device)
        self.angmodels["angdecoder"].to(self.device)
        self.load_model(weightFolder=self.opt.load_angweights_folder, encoderName='angencoder',
                        decoderName='angdecoder', encoder=self.angmodels["angencoder"],
                        decoder=self.angmodels["angdecoder"])
        for m in self.angmodels.values():
            m.eval()

        self.depthErrmodels = {}
        self.depthErrmodels["dErrencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=False, num_input_channels=4)
        self.depthErrmodels["dErrdecoder"] = DepthDecoder(self.depthErrmodels["dErrencoder"].num_ch_enc, num_output_channels=2)
        self.depthErrmodels["dErrencoder"].to(self.device)
        self.depthErrmodels["dErrdecoder"].to(self.device)
        self.load_model(weightFolder=self.opt.load_depthErr_folder, encoderName='dErrencoder',
                        decoderName='dErrdecoder', encoder=self.depthErrmodels["dErrencoder"],
                        decoder=self.depthErrmodels["dErrdecoder"])
        for m in self.depthErrmodels.values():
            m.eval()

        self.angErrmodels = {}
        self.angErrmodels["aErrencoder"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=False, num_input_channels=5)
        self.angErrmodels["aErrdecoder"] = DepthDecoder(self.angErrmodels["aErrencoder"].num_ch_enc, num_output_channels=2)
        self.angErrmodels["aErrencoder"].to(self.device)
        self.angErrmodels["aErrdecoder"].to(self.device)
        self.load_model(weightFolder=self.opt.load_angErr_folder, encoderName='aErrencoder',
                        decoderName='aErrdecoder', encoder=self.angErrmodels["aErrencoder"],
                        decoder=self.angErrmodels["aErrdecoder"])
        for m in self.angErrmodels.values():
            m.eval()

        print("Training is using:\t", self.device)

        self.set_dataset()

        self.MIN_DEPTH = 1e-3
        self.MAX_DEPTH = 80

        self.STEREO_SCALE_FACTOR = 5.4

        self.sfnormOptimizer = SurfaceNormalOptimizer(height=self.opt.crph, width=self.opt.crpw, batch_size=self.opt.batch_size).cuda()

    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "{}_files.txt")
        test_fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "splits", self.opt.split, "val_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(test_fpath)

        self.train_dataset = KittiDataset(
            self.opt.data_path, self.opt.gt_path, train_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=True, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path
        )

        self.val_dataset = KittiDataset(
            self.opt.data_path, self.opt.val_gt_path, val_filenames, self.opt.height, self.opt.width,
            crph=self.opt.crph, crpw=self.opt.crpw, is_train=False, predang_path=self.opt.predang_path,
            semanticspred_path=self.opt.semanticspred_path
        )

        self.train_loader = DataLoader(
            self.train_dataset, self.opt.batch_size, shuffle=True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            self.val_dataset, self.opt.batch_size, shuffle=False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        srh = 30
        srv = 30
        with torch.no_grad():
            for batch_idx, inputs in enumerate(self.val_loader):
                for key, ipt in inputs.items():
                    if not key == 'tag':
                        inputs[key] = ipt.to(self.device)

                outputs_depth = self.depthmodels['depthdecoder'](self.depthmodels['depthencoder'](inputs['color']))
                outputs_ang = self.angmodels['angdecoder'](self.angmodels['angencoder'](inputs['color']))
                outputs_dErr = self.depthErrmodels["dErrdecoder"](self.depthErrmodels["dErrencoder"](torch.cat([inputs['color'], outputs_depth['disp', 0]], dim=1)))
                outputs_aErr = self.angErrmodels["aErrdecoder"](self.angErrmodels["aErrencoder"](torch.cat([inputs['color'], outputs_ang['disp', 0]], dim=1)))

                pred_depth = F.interpolate(outputs_depth['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                _, pred_depth = disp_to_depth(pred_depth, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
                pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

                err_depth_act = F.interpolate(outputs_dErr['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                err_shape_act = F.interpolate(outputs_aErr['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                rgb_resized = F.interpolate(inputs['color'], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)

                pred_ang = F.interpolate(outputs_ang['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                pred_ang = (pred_ang - 0.5) * 2 * np.pi
                pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang)

                err_depth = self.sfnormOptimizer.act2err(errpred=err_depth_act, intrinsic=inputs['K'])
                err_shape = self.sfnormOptimizer.act2err(errpred=err_shape_act, intrinsic=inputs['K'])

                err_depthnph = err_depth[0, 0, :, :].cpu().numpy()
                err_depthnpv = err_depth[0, 1, :, :].cpu().numpy()
                err_shapenph = err_shape[0, 0, :, :].cpu().numpy()
                err_shapenpv = err_shape[0, 1, :, :].cpu().numpy()

                logh = pred_log[0, 0, :, :].detach().cpu().numpy()
                logv = pred_log[0, 1, :, :].detach().cpu().numpy()
                pred_depthnp = pred_depth[0, 0, :, :].detach().cpu().numpy()

                depthgtnp = inputs['depthgt'][0, 0, :, :].detach().cpu().numpy()
                nn, mm = np.meshgrid(range(self.opt.crpw), range(self.opt.crph), indexing='xy')
                mask = np.zeros([self.opt.crph, self.opt.crpw])
                mask[int(0.40810811 * self.opt.crph):int(0.99189189 * self.opt.crph), int(0.03594771 * self.opt.crpw):int(0.96405229 * self.opt.crpw)] = 1
                mask = mask == 1
                mask = mask * (depthgtnp > 0)
                mask = mask * (inputs['semanticspred'][0,0,:,:].cpu().numpy() != 0)
                nn = nn[mask]
                mm = mm[mask]

                semanticsnp = inputs['semanticspred'][0,0,:,:].cpu().numpy()

                shapeErrRech_totc, depthErrRech_totc, errRech_numc, shapeErrRecv_totc, depthErrRecv_totc, errRecv_numc = evaluateReceptiveFieldSemantic(height=self.opt.crph, width=self.opt.crpw, logh=logh, logv=logv, depthpredl=np.log(pred_depthnp), depthgtl=np.log(depthgtnp), mask=depthgtnp > 0, srh=srh, srv=srv)
                shapeErrRech_tot = shapeErrRech_tot + shapeErrRech_totc
                depthErrRech_tot = depthErrRech_tot + depthErrRech_totc
                errRech_num = errRech_num + errRech_numc
                shapeErrRecv_tot = shapeErrRecv_tot + shapeErrRecv_totc
                depthErrRecv_tot = depthErrRecv_tot + depthErrRecv_totc
                errRecv_num = errRecv_num + errRecv_numc

                ptshtot = list()
                ptsvtot = list()
                for kk in range(6):
                    anghnp = pred_ang[0,0,:,:].detach().cpu().numpy()
                    angvnp = pred_ang[0,1,:,:].detach().cpu().numpy()
                    rndindex = np.random.randint(low=0, high=nn.shape[0])
                    n = nn[rndindex]
                    m = mm[rndindex]
                    ptsh, ptsv = dynamicReceptiveFieldFromShape(anghnp=anghnp, angvnp=angvnp, w=self.opt.crpw, h=self.opt.crph, m=m, n=n, barh=0.1, barv=0.1)
                    ptshtot.append(ptsh)
                    ptsvtot.append(ptsv)
                ptshtot = np.concatenate(ptshtot, axis=0)
                ptsvtot = np.concatenate(ptsvtot, axis=0)
                plt.figure(figsize=(16, 8))
                plt.imshow(tensor2rgb(rgb_resized, ind=0))
                plt.scatter(ptshtot[:, 0], ptshtot[:, 1], s=0.1, c='r')
                plt.scatter(ptsvtot[:, 0], ptsvtot[:, 1], s=0.1, c='r')
                plt.scatter([n], [m], s=5, c='b')
                plt.savefig(os.path.join(
                    '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/analysis_dynamicReceptiveField2',
                    '{}.png'.format(batch_idx)))
                plt.close()

                for kk in range(6):
                    sw = 100
                    sh = 50
                    rndindex = np.random.randint(low=0, high=nn.shape[0])
                    n = nn[rndindex]
                    m = mm[rndindex]
                    goodpts, badpts, goodptsl, badptsl = vls_dynamicReceptiveField(err_shapenph=err_shapenph,
                                                                                   err_shapenpv=err_shapenpv,
                                                                                   err_depthnph=err_depthnph,
                                                                                   err_depthnpv=err_depthnpv,
                                                                                   logh=logh, logv=logv,
                                                                                   pred_depthnp=pred_depthnp,
                                                                                   depthgtnp=depthgtnp, mask=mask,
                                                                                   w=self.opt.crpw, h=self.opt.crph, sw=sw,
                                                                                   sh=sh, m=m, n=n)
                    fig, axs = plt.subplots(2, figsize=(16, 8))
                    axs[0].imshow(tensor2rgb(rgb_resized, ind=0))
                    axs[0].scatter(goodpts[:, 0], goodpts[:, 1], s=0.1, c='r')
                    axs[0].scatter(badpts[:, 0], badpts[:, 1], s=0.1, c='b')
                    axs[1].imshow(tensor2rgb(rgb_resized, ind=0))
                    axs[1].scatter(goodpts[:, 0], goodpts[:, 1], s=0.1, c='c')
                    axs[1].scatter(badpts[:, 0], badpts[:, 1], s=0.1, c='c')
                    axs[1].scatter(badptsl[:, 0], badptsl[:, 1], s=0.7, c='b')
                    axs[1].scatter(goodptsl[:, 0], goodptsl[:, 1], s=0.7, c='r')
                    plt.savefig(os.path.join(
                        '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/analysis_dynamicReceptiveField',
                        '{}.png'.format(batch_idx)))
                    plt.close()

                minang = - np.pi / 3 * 2
                maxang = 2 * np.pi - np.pi / 3 * 2
                tensor2disp(pred_ang[:, 0:1, :, :] - minang, vmax=maxang, ind=0).show()
                tensor2disp(pred_ang[:, 1:2, :, :] - minang, vmax=maxang, ind=0).show()
                tensor2disp(outputs_depth['disp', 0], vmax=0.1, ind=0).show()

                tensor2disp(err_depth[:, 0:1, :, :], vmax=np.pi / 2, ind=0).show()
                tensor2disp(err_shape[:, 0:1, :, :], vmax=np.pi / 2, ind=0).show()

                hgood, vgood = dynamicReceptiveField(height=self.opt.height, width=self.opt.width, err_shapenph=err_shapenph, err_shapenpv=err_shapenpv, err_depthnph=err_depthnph, err_depthnpv=err_depthnpv, sw=self.opt.width, sh=self.opt.height)
                plt.figure()
                plt.imshow(tensor2disp(torch.from_numpy(hgood).unsqueeze(0).unsqueeze(0), vmax=50, ind=0))
                plt.colorbar()
                plt.title('Horizontal Direction Receptive field Range')
                plt.figure()
                plt.imshow(tensor2disp(torch.from_numpy(vgood).unsqueeze(0).unsqueeze(0), vmax=50, ind=0))
                plt.colorbar()
                plt.title('Vertical Direction Receptive field Range')

    def val_semantic(self):
        self.set_eval()
        srh = 100
        srv = 100
        from kitti_utils import read_calib_file, trainId2label
        interested_labelComb = [['road'], ['sidewalk'], ['building'], ['wall'], ['fence'], ['pole'], ['vegetation'], ['terrain'], ['person', 'rider', 'motorcycle', 'bicycle'], ['car', 'truck', 'bus', 'train'], ['traffic light'], ['traffic sign']]

        for labelComb in interested_labelComb:
            shapeErrRech_tot = np.zeros((srh))
            depthErrRech_tot = np.zeros((srh))
            errRech_num = np.zeros((srh))
            shapeErrRecv_tot = np.zeros((srv))
            depthErrRecv_tot = np.zeros((srv))
            errRecv_num = np.zeros((srv))
            with torch.no_grad():
                for batch_idx, inputs in enumerate(self.val_loader):
                    for key, ipt in inputs.items():
                        if not key == 'tag':
                            inputs[key] = ipt.to(self.device)

                    outputs_depth = self.depthmodels['depthdecoder'](self.depthmodels['depthencoder'](inputs['color']))
                    outputs_ang = self.angmodels['angdecoder'](self.angmodels['angencoder'](inputs['color']))

                    pred_depth = F.interpolate(outputs_depth['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                    _, pred_depth = disp_to_depth(pred_depth, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
                    pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

                    pred_ang = F.interpolate(outputs_ang['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                    pred_ang = (pred_ang - 0.5) * 2 * np.pi
                    pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang)

                    logh = pred_log[0, 0, :, :].detach().cpu().numpy()
                    logv = pred_log[0, 1, :, :].detach().cpu().numpy()
                    pred_depthnp = pred_depth[0, 0, :, :].detach().cpu().numpy()

                    depthgtnp = inputs['depthgt'][0, 0, :, :].detach().cpu().numpy()

                    mask = (inputs['depthgt'] > 0)
                    semanticmask = torch.zeros_like(mask)
                    for labelname in labelComb:
                        for lind in trainId2label:
                            if trainId2label[lind].name == labelname:
                                semanticmask = semanticmask + (inputs['semanticspred'] == trainId2label[lind].trainId)
                    mask = mask * semanticmask > 0
                    mask = mask[0,0,:,:].cpu().numpy()

                    semanticsnp = inputs['semanticspred'][0,0,:,:].cpu().numpy()

                    if np.sum(mask) > 0:
                        shapeErrRech_totc, depthErrRech_totc, errRech_numc, shapeErrRecv_totc, depthErrRecv_totc, errRecv_numc = evaluateReceptiveFieldSemantic(height=self.opt.crph, width=self.opt.crpw, logh=logh, logv=logv, depthpredl=np.log(pred_depthnp), depthgtl=np.log(depthgtnp), mask=mask, semantics=semanticsnp,  srh=srh, srv=srv)
                        shapeErrRech_tot = shapeErrRech_tot + shapeErrRech_totc
                        depthErrRech_tot = depthErrRech_tot + depthErrRech_totc
                        errRech_num = errRech_num + errRech_numc
                        shapeErrRecv_tot = shapeErrRecv_tot + shapeErrRecv_totc
                        depthErrRecv_tot = depthErrRecv_tot + depthErrRecv_totc
                        errRecv_num = errRecv_num + errRecv_numc

                    print("Batch %d finished" % batch_idx)

            figname = ''
            for labelname in labelComb:
                if len(figname) > 0:
                    figname = figname + '_'
                figname = figname + labelname

            shapeErrRech_tot = shapeErrRech_tot / errRech_num
            depthErrRech_tot = depthErrRech_tot / errRech_num
            shapeErrRecv_tot = shapeErrRecv_tot / errRecv_num
            depthErrRecv_tot = depthErrRecv_tot / errRecv_num

            fig, axs = plt.subplots(2, figsize=(16, 8))
            axs[0].scatter(list(range(srh)), shapeErrRech_tot)
            axs[0].scatter(list(range(srh)), depthErrRech_tot)
            axs[0].legend(['shapeErr h', 'depthErr h'])
            axs[1].scatter(list(range(srv)), shapeErrRecv_tot)
            axs[1].scatter(list(range(srv)), depthErrRecv_tot)
            axs[1].legend(['shapeErr v', 'depthErr v'])
            fig.suptitle('Shape Error Analysis for semantic type %s' % figname)
            plt.savefig(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/analysis_semanticShapeErr', "{}.png".format(figname)), bbox_inches='tight')
            plt.close()

    def val_semantic_VarMDepthErr(self):
        self.set_eval()
        from kitti_utils import trainId2label
        interested_labelComb = [['road'], ['sidewalk'], ['terrain'], ['building'], ['wall'], ['fence'], ['pole'], ['traffic light'], ['traffic sign'], ['vegetation'], ['person', 'rider', 'motorcycle', 'bicycle'], ['car'], ['truck'], ['bus'], ['train']]
        # interested_labelComb = [['pole'], ['person', 'rider', 'motorcycle', 'bicycle']]

        maxpixels = 2e7
        maxVar = 0.2
        recnum = 100
        for labelComb in interested_labelComb:
            countedpixels = 0
            shapeErrVarh_tot = np.zeros(recnum)
            depthErrVarh_tot = np.zeros(recnum)
            rangeVarh_tot = np.zeros(recnum)
            errVarNum_h = np.zeros(recnum)

            shapeErrVarv_tot = np.zeros(recnum)
            depthErrVarv_tot = np.zeros(recnum)
            rangeVarv_tot = np.zeros(recnum)
            errVarNum_v = np.zeros(recnum)

            figname = ''
            for labelname in labelComb:
                if len(figname) > 0:
                    figname = figname + '_'
                figname = figname + labelname

            with torch.no_grad():
                for batch_idx, inputs in enumerate(self.val_loader):
                    mask = (inputs['depthgt'] > 0)
                    semanticmask = torch.zeros_like(mask)
                    for labelname in labelComb:
                        for lind in trainId2label:
                            if trainId2label[lind].name == labelname:
                                semanticmask = semanticmask + (inputs['semanticspred'] == trainId2label[lind].trainId)
                    mask = mask * semanticmask > 0
                    mask = mask[0, 0, :, :].cpu().numpy()
                    semanticmask = semanticmask[0, 0, :, :].cpu().numpy() > 0

                    if np.sum(mask) > 0:
                        for key, ipt in inputs.items():
                            if not key == 'tag':
                                inputs[key] = ipt.to(self.device)

                        outputs_depth = self.depthmodels['depthdecoder'](self.depthmodels['depthencoder'](inputs['color']))
                        outputs_ang = self.angmodels['angdecoder'](self.angmodels['angencoder'](inputs['color']))

                        pred_depth = F.interpolate(outputs_depth['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                        _, pred_depth = disp_to_depth(pred_depth, min_depth=self.opt.min_depth, max_depth=self.opt.max_depth)
                        pred_depth = pred_depth * self.STEREO_SCALE_FACTOR

                        pred_ang = F.interpolate(outputs_ang['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
                        pred_ang = (pred_ang - 0.5) * 2 * np.pi
                        pred_log = self.sfnormOptimizer.ang2log(intrinsic=inputs['K'], ang=pred_ang)

                        edge = self.sfnormOptimizer.ang2edge(pred_ang, inputs['K'])
                        edgenp = edge[0, 0].cpu().numpy()

                        semanticmask = semanticmask * (1 - edgenp)
                        mask = mask * (1 - edgenp)

                        mask = mask == 1
                        semanticmask = semanticmask == 1
                        # tensor2disp(torch.from_numpy(mask).unsqueeze(0).unsqueeze(0), vmax=1, ind=0).show()
                        # tensor2disp(torch.from_numpy(semanticmask).unsqueeze(0).unsqueeze(0), vmax=1, ind=0).show()
                        # tensor2disp(torch.from_numpy(1 - edgenp).unsqueeze(0).unsqueeze(0), vmax=1, ind=0).show()

                        logh = pred_log[0, 0, :, :].detach().cpu().numpy()
                        logv = pred_log[0, 1, :, :].detach().cpu().numpy()
                        angh = pred_ang[0, 0, :, :].detach().cpu().numpy()
                        angv = pred_ang[0, 1, :, :].detach().cpu().numpy()
                        pred_depthnp = pred_depth[0, 0, :, :].detach().cpu().numpy()
                        depthgtnp = inputs['depthgt'][0, 0, :, :].detach().cpu().numpy()

                        shapeErrVarh_totc, depthErrVarh_totc, rangeVarh_totc, errVarNum_hc, \
                        shapeErrVarv_totc, depthErrVarv_totc, rangeVarv_totc, errVarNum_vc = \
                            evaluateReceptiveFieldTrueVariance(height=self.opt.crph, width=self.opt.crpw, logh=logh, logv=logv,
                                                           depthpredl=np.log(pred_depthnp), depthgtl=np.log(depthgtnp), shapeh=angh, shapev=angv,
                                                           mask=mask, semanticmask=semanticmask, recnum=recnum, maxvar=maxVar)

                        countedpixels = countedpixels + np.sum(mask)

                        shapeErrVarh_tot = shapeErrVarh_tot + shapeErrVarh_totc
                        depthErrVarh_tot = depthErrVarh_tot + depthErrVarh_totc
                        rangeVarh_tot = rangeVarh_tot + rangeVarh_totc
                        errVarNum_h = errVarNum_h + errVarNum_hc

                        shapeErrVarv_tot = shapeErrVarv_tot + shapeErrVarv_totc
                        depthErrVarv_tot = depthErrVarv_tot + depthErrVarv_totc
                        rangeVarv_tot = rangeVarv_tot + rangeVarv_totc
                        errVarNum_v = errVarNum_v + errVarNum_vc

                        # progressBar(batch_idx, (self.val_loader.__len__()), figname, barLength=20)
                        progressBar(countedpixels, maxpixels, figname, barLength=20)
                        if countedpixels > maxpixels:
                            break

            selectorh = errVarNum_h > 0
            shapeErrVarh_tot = shapeErrVarh_tot[selectorh] / (errVarNum_h[selectorh] + 1)
            depthErrVarh_tot = depthErrVarh_tot[selectorh] / (errVarNum_h[selectorh] + 1)
            rangeVarh_tot = rangeVarh_tot[selectorh] / (errVarNum_h[selectorh] + 1)

            selectorv = errVarNum_v > 0
            shapeErrVarv_tot = shapeErrVarv_tot[selectorv] / (errVarNum_v[selectorv] + 1)
            depthErrVarv_tot = depthErrVarv_tot[selectorv] / (errVarNum_v[selectorv] + 1)
            rangeVarv_tot = rangeVarv_tot[selectorv] / (errVarNum_v[selectorv] + 1)

            errVarBin_hvls = np.linspace(0, maxVar, num=recnum)
            errVarBin_hvls = errVarBin_hvls[selectorh]
            errVarBin_vvls = np.linspace(0, maxVar, num=recnum)
            errVarBin_vvls = errVarBin_vvls[selectorv]

            fig, axs = plt.subplots(3, figsize=(16, 12))
            axs[0].scatter(errVarBin_hvls[:-1], shapeErrVarh_tot[:-1], s=25)
            axs[0].scatter(errVarBin_hvls[:-1], depthErrVarh_tot[:-1], s=25)
            axs[0].legend(['shapeErr h', 'depthErr h'])
            axs[0].set(xlabel='Variance', ylabel='Log Error')

            axs[1].scatter(errVarBin_vvls[:-1], shapeErrVarv_tot[:-1], s=25)
            axs[1].scatter(errVarBin_vvls[:-1], depthErrVarv_tot[:-1], s=25)
            axs[1].legend(['shapeErr v', 'depthErr v'])
            axs[1].set(xlabel='Variance', ylabel='Log Error')

            axs[2].scatter(errVarBin_hvls[:-1], rangeVarh_tot[:-1], s=25)
            axs[2].scatter(errVarBin_vvls[:-1], rangeVarv_tot[:-1], s=25)
            axs[2].legend(['range h', 'range v'])
            axs[2].set(xlabel='Variance', ylabel='Range')

            fig.suptitle('Metric Error Analysis for semantic type %s' % figname)
            plt.savefig(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/analysis_semanticMetric/valset_variance', "{}.png".format(figname)), bbox_inches='tight')
            plt.close()

    def vls_shaperange(self):
        self.set_eval()
        from kitti_utils import shapecats, translateTrainIdSemantics, variancebar

        gargmask = np.zeros([self.opt.crph, self.opt.crpw], dtype=np.bool)
        gargmask[int(0.40810811 * self.opt.crph):int(0.99189189 * self.opt.crph), int(0.03594771 * self.opt.crpw):int(0.96405229 * self.opt.crpw)] = 1

        xx, yy = np.meshgrid(range(self.opt.crpw), range(self.opt.crph), indexing='xy')
        for batch_idx in range(self.val_dataset.__len__()):
            inputs = self.val_dataset.__getitem__(batch_idx)
            for key, ipt in inputs.items():
                if not key == 'tag':
                    inputs[key] = ipt.to(self.device).unsqueeze(0)
            semantics = translateTrainIdSemantics(inputs['semanticspred'][0, 0, :, :].cpu().numpy())
            with torch.no_grad():
                outputs_ang = self.angmodels['angdecoder'](self.angmodels['angencoder'](inputs['color'].cuda()))
            pred_ang = F.interpolate(outputs_ang['disp', 0], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
            rgb = F.interpolate(inputs['color'], (self.opt.crph, self.opt.crpw), mode='bilinear', align_corners=True)
            pred_ang = (pred_ang - 0.5) * 2 * np.pi

            angh = pred_ang[0, 0, :, :].detach().cpu().numpy()
            angv = pred_ang[0, 1, :, :].detach().cpu().numpy()

            edge = self.sfnormOptimizer.ang2edge(pred_ang, inputs['K'])
            edgenp = edge[0, 0].cpu().numpy()

            selectedpts = list()
            centerpts = list()
            for shapecat in shapecats:
                barh, barv = variancebar[shapecat.categoryId, :]
                if (barh != -1 or barv != -1):
                    semanticmask = (semantics == shapecat.categoryId) * gargmask * (1 - edgenp)
                    semanticmask = semanticmask == 1
                    if np.sum(semanticmask) > 10:
                        xxs = xx[semanticmask]
                        yys = yy[semanticmask]
                        rndind = np.random.randint(0, np.sum(semanticmask) - 1)

                        n = xxs[rndind]
                        m = yys[rndind]
                        centerpts = centerpts + [np.array([n, m])]

                        rngpts = get_dynamicRange(height=self.opt.crph, width=self.opt.crpw, shapeh=angh, shapev=angv, semanticmask=semanticmask, barh=barh, barv=barv, m=m, n=n)
                        selectedpts = selectedpts + rngpts

            if len(centerpts) > 0 and len(selectedpts) > 0:
                figname = inputs['tag'].split(' ')[0].split('/')[1] + '_' + inputs['tag'].split(' ')[1]
                centerptsv = np.stack(centerpts, axis=0)
                selectedptsv = np.stack(selectedpts, axis=0)

                fig, axs = plt.subplots(2, figsize=(16, 8))
                axs[0].imshow(tensor2rgb(rgb, ind=0))
                axs[0].scatter(centerptsv[:, 0], centerptsv[:, 1], c='r', s=1)
                axs[0].scatter(selectedptsv[:, 0], selectedptsv[:, 1], c='g', s=1)

                axs[1].imshow(tensor2semantic(inputs['semanticspred'], ind=0))
                axs[1].scatter(selectedptsv[:, 0], selectedptsv[:, 1], c='g', s=1)
                axs[1].scatter(centerptsv[:, 0], centerptsv[:, 1], c='r', s=1)

                plt.savefig(os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/analysis_semanticMetric/rangevls', "{}.png".format(figname)), bbox_inches='tight')
                plt.close()

                # minang = - np.pi / 3 * 2
                # maxang = 2 * np.pi - np.pi / 3 * 2
                # viewind = 0
                # tensor2disp(pred_ang[:, 0:1, :, :] - minang, vmax=maxang, ind=viewind).show()
                # tensor2disp(pred_ang[:, 1:2, :, :] - minang, vmax=maxang, ind=viewind).show()
                # tensor2disp(edge, vmax=1, ind=0).show()

    def load_model(self, weightFolder, encoderName, decoderName, encoder, decoder):
        """Load model(s) from disk
        """
        assert os.path.isdir(weightFolder), "Cannot find folder {}".format(weightFolder)
        print("loading model from folder {}".format(weightFolder))

        path = os.path.join(weightFolder, "encoder.pth")
        print("Loading {} weights...".format(encoderName))
        model_dict = encoder.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        encoder.load_state_dict(model_dict)

        path = os.path.join(weightFolder, "depth.pth")
        print("Loading {} weights...".format(decoderName))
        model_dict = decoder.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        decoder.load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val_semantic_VarMDepthErr()