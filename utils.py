# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
from six.moves import urllib
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
from kitti_utils import name2label, trainId2label
from collections import Counter
from numba import njit, prange
import numba
import scipy.stats as st
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))


# ------ Added -------- #

# Visualization Related:
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def visualize_semantic(img_inds):
    # please input numpy array
    size = [img_inds.shape[1], img_inds.shape[0]]
    background = name2label['unlabeled'].color
    labelImg = np.array(pil.new("RGB", size, background))
    for id in trainId2label.keys():
        if id >= 0:
            label = trainId2label[id].name
        else:
            label = 'unlabeled'
        color = name2label[label].color
        mask = img_inds == id
        labelImg[mask, :] = color
    return pil.fromarray(labelImg)

def tensor2rgb(tensor, ind):
    slice = (tensor[ind, :, :, :].permute(1,2,0).detach().contiguous().cpu().numpy() * 255).astype(np.uint8)
    return pil.fromarray(slice)

def tensor2flatrgb(tensor):
    batchSize, channels, height, width = tensor.shape
    slice = (tensor.permute(0,2,3,1).contiguous().view(-1, width, channels).detach().contiguous().cpu().numpy() * 255).astype(np.uint8)
    return pil.fromarray(slice)

def tensor2disp(tensor, ind, vmax = None, percentile = None):
    slice = tensor[ind, 0, :, :].detach().cpu().numpy()
    if percentile is None:
        percentile = 90
    if vmax is None:
        vmax = np.percentile(slice, percentile)
    slice = slice / vmax
    cm = plt.get_cmap('magma')
    slice = (cm(slice) * 255).astype(np.uint8)
    return pil.fromarray(slice[:,:,0:3])

def tensor2disp_flat(tensor, vmax = None, percentile = None):
    batchSize, channels, height, width = tensor.shape
    slice = tensor.permute(0,2,3,1).contiguous().view(-1, width, channels).squeeze(2).detach().cpu().numpy()
    if percentile is None:
        percentile = 90
    if vmax is None:
        vmax = np.percentile(slice, percentile)
    slice = slice / vmax
    cm = plt.get_cmap('magma')
    slice = (cm(slice) * 255).astype(np.uint8)
    return pil.fromarray(slice[:,:,0:3])

def tensor2semantic(tensor, ind, isGt = False):
    slice = tensor[ind, :, :, :]
    slice = slice[0,:,:].detach().cpu().numpy()
    # visualize_semantic(slice).show()
    return visualize_semantic(slice)

def tensor2grad(gradtensor, percentile = 95, viewind = 0):
    cm = plt.get_cmap('bwr')
    gradnumpy = gradtensor.detach().cpu().numpy()[viewind, 0, :, :]

    selector_pos = gradnumpy > 0
    if np.sum(selector_pos) > 1:
        pos_bar = np.percentile(gradnumpy[selector_pos], percentile)
        gradnumpy[selector_pos] = gradnumpy[selector_pos] / pos_bar / 2

    selector_neg = gradnumpy < 0
    if np.sum(selector_neg) > 1:
        neg_bar = -np.percentile(-gradnumpy[selector_neg], percentile)
        gradnumpy[selector_neg] = -gradnumpy[selector_neg] / neg_bar / 2

    disp_grad_numpy = gradnumpy + 0.5
    colorMap = cm(disp_grad_numpy)[:,:,0:3]
    return pil.fromarray((colorMap * 255).astype(np.uint8))

def draw_detection(data, detection_res, ind):
    data_entry = data[ind, :, :, :].unsqueeze(0)
    detection_res_entry = detection_res[ind, :, :]
    detection_res_entry = detection_res_entry[detection_res_entry[:, 0] > 0, :]

    fig, ax = plt.subplots(1)
    # Display the image
    im = tensor2rgb(data_entry, ind=0)
    ax.imshow(im)

    for k in range(len(detection_res_entry)):
        # Read Label
        sx = detection_res_entry[k][0]
        sy = detection_res_entry[k][1]
        rw = detection_res_entry[k][2] - sx
        rh = detection_res_entry[k][3] - sy

        # Create a Rectangle patch
        rect = patches.Rectangle((sx, sy), rw, rh, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

def sampleImgs(imgs, pts2d, mode = 'bilinear'):
    batchSize, channels, height, width = imgs.shape
    if len(pts2d.shape) == 3:
        _, _, nums = pts2d.shape
        pts2dNormed = (pts2d / torch.Tensor([width - 1, height - 1]).cuda().unsqueeze(0).unsqueeze(2).expand(batchSize, -1, nums) - 0.5) * 2
        pts2dNormed = pts2dNormed.permute(0,2,1).unsqueeze(2)
        sampledColor = F.grid_sample(imgs, pts2dNormed, mode = mode, padding_mode='border')
        sampledColor = sampledColor.squeeze(3)
        return sampledColor
    else:
        _, _, heights, widths = pts2d.shape
        pts2dNormed = (pts2d / torch.Tensor([width - 1, height - 1]).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(batchSize, -1, heights, widths) - 0.5) * 2
        pts2dNormed = pts2dNormed.permute(0,2,3,1)
        sampledColor = F.grid_sample(imgs, pts2dNormed, mode = mode, padding_mode='border')
        return sampledColor


# Projection Function Related:
def project_3dptsTo2dpts(pts3d, camKs, imgHeight = 320, imgWidth = 1024):
    format_indicator = 0
    if len(pts3d.shape) == 4:
        batchSize, channels, height, width = pts3d.shape
        pts3dT = pts3d.view(batchSize, channels, -1)
        format_indicator = 1
    else:
        pts3dT = pts3d.permute(0, 2, 1)
    projected3d = torch.matmul(camKs, pts3dT)
    projecDepth = projected3d[:, 2, :].unsqueeze(1)
    projected2d = torch.stack(
        [projected3d[:, 0, :] / projected3d[:, 2, :], projected3d[:, 1, :] / projected3d[:, 2, :]], dim=1)
    selector = (projected2d[:, 0, :] > 0) * (projected2d[:, 0, :] < imgWidth - 1) * (projected2d[:, 1, :] > 0) * (
                projected2d[:, 1, :] < imgHeight - 1) * (projecDepth[:, 0, :] > 0)
    selector = selector.unsqueeze(1)

    if format_indicator == 1:
        projected2d = projected2d.view(batchSize, 2, height, width)
        projecDepth = projecDepth.view(batchSize, 1, height, width)
    return projected2d, projecDepth, selector

def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def filter_duplicated_depth(imgSize, xx, yy, dpethvals):
    xx = np.round(xx)
    yy = np.round(yy)
    inds = sub2ind(imgSize, yy, xx)
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    selector = np.ones([xx.shape[0]]) > 0
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        selector[pts] = 0
        max_local_ind = np.argmax(dpethvals[pts])
        selector[pts[max_local_ind]] = 1
    return selector

def backProjTo3d(pixelLocs, depths, invcamK):
    batchSize, channels, height, width = depths.shape
    tmpcx = pixelLocs[:,0,:,:] * depths[:, 0, :, :]
    tmpcy = pixelLocs[:,1,:,:] * depths[:, 0, :, :]
    tmpcd = depths[:, 0, :, :]
    tmpc1 = torch.ones([batchSize, height, width], device=torch.device('cuda'), dtype=torch.float32)
    tmpPts3d = torch.stack([tmpcx, tmpcy, tmpcd, tmpc1], dim=1)
    tmpPts3d = tmpPts3d.view(batchSize, 4, -1)
    pts3d = torch.matmul(invcamK, tmpPts3d)
    pts3d = pts3d.view(batchSize, 4, height, width)
    return pts3d

# Other
def latlonToMercator(lat, lon, scale):
    er = 6378137
    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))
    return mx, my

def latToScale(lat):
    scale = np.cos(lat * np.pi / 180.0)
    return scale

def cvtPNG2Arr(png):
    sr = 10000
    png = np.array(png)
    h = png[:,:,0]
    e = png[:,:,1]
    l = png[:,:,2]

    arrs = h.astype(np.float32) * 256 * 256 + e.astype(np.float32) * 256 + l.astype(np.float32)
    arr = arrs / sr
    return arr

def norm_tensor(depth, pSIL_insMask_shrinked):
    batch_size, _, prsil_ch, prsil_cw = depth.shape
    mu = torch.sum(pSIL_insMask_shrinked * depth, dim=[1, 2, 3]) / torch.sum(pSIL_insMask_shrinked, dim=[1, 2, 3])
    mu_ex = mu.view(batch_size, 1, 1, 1).expand([-1, 1, prsil_ch, prsil_cw])
    scale = torch.sqrt(torch.sum(pSIL_insMask_shrinked * ((depth - mu_ex) ** 2), dim=[1, 2, 3])) / torch.sum(
        pSIL_insMask_shrinked, dim=[1, 2, 3])
    scale_ex = scale.view(batch_size, 1, 1, 1).expand([-1, 1, prsil_ch, prsil_cw])

    torch.sqrt(torch.sum(((depth - mu_ex) / scale_ex) ** 2 * pSIL_insMask_shrinked, dim=[1, 2, 3])) / torch.sum(
        pSIL_insMask_shrinked, dim=[1, 2, 3])

    return (depth - mu_ex) / scale_ex

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

@numba.jit(nopython=True, parallel=False)
def recover_depth_from_theta(debias_ratiohl, depthmap, recovered_depth, optimize_mask, height, width):
    for i in range(height):
        recovered_depth[i, 0] = depthmap[i, 0]
    for i in range(height):
        for j in range(1,width):
            if optimize_mask[i, j] > 0:
                recovered_depth[i, j] = recovered_depth[i,j-1] * np.exp(debias_ratiohl[i,j-1])
            else:
                recovered_depth[i, j] = depthmap[i,j]

def conjugate_gradient():
    # For sparse Conjugate Gradient
    a = 1