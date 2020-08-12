from __future__ import absolute_import, division, print_function
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from torch.utils.data import DataLoader
from layers import *
from utils import readlines
from options import MonodepthOptions

import networks

import cv2
import numpy as np
import torch
from scipy import fftpack

import torchvision
from kitti_utils import read_calib_file

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "..", "splits")

STEREO_SCALE_FACTOR = 5.4

def read_data(framecount, eval_split):
    isck = False
    totensor = torchvision.transforms.ToTensor()

    disparity_gt_root = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/data_scene_flow/training'
    dirMap_disp = {'l': 'disp_noc_0', 'r': 'disp_noc_1'}
    dirMap_rgb = {'l': 'image_2', 'r': 'image_3'}
    dirMap_filterlida = {'l': 'image_02', 'r': 'image_03'}
    dirMap_intrinsic = {'l': 'P_rect_02', 'r': 'P_rect_03'}

    if np.random.random() > 0.5:
        dir = 'l'
        dirreverse = 'r'
    else:
        dir = 'r'
        dirreverse = 'l'

    rgb_path = os.path.join(disparity_gt_root, dirMap_rgb[dir], "{}_{}.png".format(str(framecount).zfill(6), '10'))
    rgb = totensor(pil.open(rgb_path)).unsqueeze(0).cuda()

    disparity_path = os.path.join(disparity_gt_root, dirMap_disp[dir], "{}_{}.png".format(str(framecount).zfill(6), '10'))
    dispgt = np.array(pil.open(disparity_path)).astype(np.float32) / 256.0
    dispgt = torch.from_numpy(dispgt).unsqueeze(0).unsqueeze(0).cuda()

    cam2cam = read_calib_file(os.path.join(disparity_gt_root, 'calib_cam_to_cam', "{}.txt".format(str(framecount).zfill(6))))
    intrinsic = np.resize(cam2cam[dirMap_intrinsic[dir]], [3,4])
    fl = intrinsic[0, 0]
    bs = 0.54
    depthgt = fl * bs / dispgt
    depthgt[dispgt == 0] = 0

    data = dict()
    data['depthgt'] = depthgt
    data['rgb'] = rgb

    # Check Correctness of the data
    if eval_split[framecount] != "" and isck:
        filter_lidar_root = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'
        date, seq, frame = eval_split[framecount].split(' ')
        filterlidar_gtpath = os.path.join(filter_lidar_root, date, seq, dirMap_filterlida[dir], "{}.png".format(frame))
        filterlidar_gt = np.array(pil.open(filterlidar_gtpath)).astype(np.float32) / 256.0
        filterlidar_gt = torch.from_numpy(filterlidar_gt).unsqueeze(0).unsqueeze(0).cuda()

        assert filterlidar_gt.shape == depthgt.shape, print("Image Shape Inconsistent")

        # Visualization for correctness
        cm = plt.get_cmap('plasma')
        w, h = (cam2cam[dirMap_intrinsic[dir].replace('P', 'S')]).astype(np.int)
        xx, yy = np.meshgrid(range(w), range(h), indexing='xy')

        rgb_stereo = pil.open(os.path.join(disparity_gt_root, dirMap_rgb[dirreverse], "{}_{}.png".format(str(framecount).zfill(6), '10')))

        depthgtnp = depthgt[0,0,:,:].cpu().numpy()
        dispgtnp = dispgt[0,0,:,:].cpu().numpy()

        vlssel = depthgtnp > 0
        depthscattercolor = cm(1.0 / depthgtnp[vlssel] * 10)
        xx_shifted = xx[vlssel] - dispgtnp[vlssel]
        yy_shifted = yy[vlssel]

        fig, ax = plt.subplots(4)

        ax[0].imshow(tensor2rgb(rgb, ind=0))
        ax[0].scatter(xx[vlssel], yy[vlssel], 0.1, c=depthscattercolor)
        ax[1].imshow(rgb_stereo)
        ax[1].scatter(xx_shifted, yy_shifted, 0.1, c=depthscattercolor)


        filterlidar_gtnp = filterlidar_gt[0,0,:,:].cpu().numpy()
        vlssel = filterlidar_gtnp > 0
        filterlidar_gtdispnp = fl * bs / filterlidar_gtnp[vlssel]
        depthscattercolor = cm(1.0 / filterlidar_gtnp[vlssel] * 10)
        xx_shifted = xx[vlssel] - filterlidar_gtdispnp
        yy_shifted = yy[vlssel]

        ax[2].imshow(tensor2rgb(rgb, ind=0))
        ax[2].scatter(xx[vlssel], yy[vlssel], 0.5, c=depthscattercolor)
        ax[3].imshow(rgb_stereo)
        ax[3].scatter(xx_shifted, yy_shifted, 0.5, c=depthscattercolor)

        ax[0].set_xlim(0, w)
        ax[1].set_xlim(0, w)
        ax[2].set_xlim(0, w)
        ax[3].set_xlim(0, w)

    return data
def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    splitroot = '/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/data_scene_flow/train_mapping.txt'
    eval_split = readlines(splitroot)

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)
    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, num_output_channels=3)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    numbins = 100
    binedge = list(np.linspace(start=0, stop=80, num=numbins - 1))

    lossRec_predDepth = np.zeros([numbins])
    lossRec_predShape = np.zeros([numbins])
    numRec = np.zeros([numbins])

    for framecount in range(len(eval_split)):

        data = read_data(framecount, eval_split)
        rgb = data['rgb']

        with torch.no_grad():
            rgb_networkSz = F.interpolate(rgb, [encoder_dict['height'], encoder_dict['width']], mode='bilinear', align_corners=True)
            output = depth_decoder(encoder(rgb_networkSz))

            htheta = output[("disp", 0)][:,0:1,:,:] * 2 * np.pi
            vtheta = output[("disp", 0)][:,1:2,:,:] * 2 * np.pi
            _, preddepth = disp_to_depth(output[("disp", 0)][:,2:3,:,:], opt.min_depth, opt.max_depth)
            preddepth = preddepth * STEREO_SCALE_FACTOR

            gtheight, gtwidth = data['depthgt'].shape[2:4]

            gtscale_intrinsic = np.array([
                [0.58 * gtwidth, 0, 0.5 * gtwidth],
                [0, 1.92 * gtheight, 0.5 * gtheight],
                [0, 0, 1]], dtype=np.float32)
            depcriptor = LocalThetaDesp(height=gtheight, width=gtwidth, batch_size=1, intrinsic=gtscale_intrinsic).cuda()

            preddepth_gtsize = F.interpolate(preddepth, [gtheight, gtwidth], mode='bilinear', align_corners=True)
            htheta_gtsize = F.interpolate(htheta, [gtheight, gtwidth], mode='bilinear', align_corners=True)
            vtheta_gtsize = F.interpolate(vtheta, [gtheight, gtwidth], mode='bilinear', align_corners=True)

            sobelxw = np.array([[-0.5, 0, 0.5]])
            sobelx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[1, 3], padding=[0, 1], bias=False)
            sobelx.weight = nn.Parameter(torch.from_numpy(sobelxw).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            sobelx = sobelx.cuda()

            sobelxiw = np.array([[1, 1, 1]])
            sobelxi = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=[1, 3], padding=[0, 1], bias=False)
            sobelxi.weight = nn.Parameter(torch.from_numpy(sobelxiw).float().unsqueeze(0).unsqueeze(0), requires_grad=False)
            sobelxi = sobelxi.cuda()

            gradGt = sobelx(data['depthgt'])
            gradGti = sobelxi((data['depthgt'] > 0).float())
            gradGt[gradGti != 3] = 0

            gradPredDepth = sobelx(preddepth_gtsize)

            _, gradPredShape, _, _ = depcriptor.depth_localgeom_consistency(preddepth_gtsize, htheta_gtsize, vtheta_gtsize, True)

            depthval = data['depthgt'][gradGt != 0].cpu().numpy()
            loss_predepth = (torch.abs(gradPredDepth - gradGt)[gradGt != 0]).cpu().numpy()
            loss_predShape = (torch.abs(gradPredShape - gradGt)[gradGt != 0]).cpu().numpy()

            histind = np.digitize(depthval, binedge)

            for ind in histind:
                lossRec_predDepth[ind] = lossRec_predDepth[ind] + loss_predepth[ind]
                lossRec_predShape[ind] = lossRec_predShape[ind] + loss_predShape[ind]
                numRec[ind] = numRec[ind] + 1

            print("Frame %d finished" % framecount)
            # tensor2disp(gradGt, ind=0, vmax=0.05).show()
            # tensor2disp(torch.abs(gradGt), ind=0, vmax=0.05).show()
            # tensor2disp(torch.abs(gradPredDepth), ind=0, vmax=0.05).show()
            # tensor2disp(torch.abs(gradPredShape), ind=0, vmax=0.05).show()
    loss_predDepth = lossRec_predDepth / (numRec + 1)
    loss_predShape = lossRec_predShape / (numRec + 1)

    plt.figure()
    plt.plot(binedge, loss_predDepth[0:99])
    plt.plot(binedge, loss_predShape[0:99])
    plt.legend(['Gradient Loss Depth', 'Gradient Loss Shape'])
    plt.xlabel('Depth')
    plt.ylabel('Absolute Difference of Gradient')

if __name__ == "__main__":
    options = MonodepthOptions()
    args = options.parse()
    evaluate(args)

