from __future__ import absolute_import, division, print_function
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from torch.utils.data import DataLoader
from layers import *
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

import glob
import torch.optim as optim
import cv2
import numpy as np
import torch

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "..", "splits")

STEREO_SCALE_FACTOR = 5.4

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def get_indwith_isntancelabel(mapping):
    wins_ind = list()
    for idx, m in enumerate(mapping):
        if len(m) > 1:
            wins_ind.append(idx)
    return wins_ind

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines('/home/shengjie/Documents/Project_SemanticDepth/splits/kitti_seman_mapped2depth//train_files.txt')

    mapping = readlines(os.path.join('/home/shengjie/Documents/Project_SemanticDepth/splits', 'training_mapping.txt'))
    wins_ind = get_indwith_isntancelabel(mapping)

    gt_lidar_root = '/home/shengjie/Documents/Data/Kitti/filtered_lidar'

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'], [0], 4, is_train=False, theta_gt_path=opt.theta_gt_path, load_seman=True)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, num_output_channels=3)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    depth_decoder.cuda()
    encoder.train()
    depth_decoder.train()
    # depth_decoder.eval()
    # encoder.eval()

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))

    predw = encoder_dict['width']
    predh = encoder_dict['height']

    import matlab.engine
    eng = matlab.engine.start_matlab()

    count = 0

    comps = filenames[count].split(' ')

    semidense_path = os.path.join('/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Data/kitti/semidense_gt', comps[0], 'image_02', comps[1].zfill(10)+'.png')
    semidense_depth = pil.open(semidense_path)
    semidense_depth = np.array(semidense_depth).astype(np.float32) / 256
    semidense_depth_torch = torch.from_numpy(semidense_depth).unsqueeze(0).unsqueeze(0).cuda()

    gt_depth_path = os.path.join(gt_lidar_root, comps[0], 'image_02', comps[1].zfill(10) + '.png')
    gt_depth = pil.open(gt_depth_path)
    gt_depth = np.array(gt_depth).astype(np.float32) / 256
    gtheight, gtwidth = gt_depth.shape

    instance_semantic_gt = pil.open(os.path.join('/home/shengjie/Documents/Data/Kitti/kitti_semantics/training/instance', str(wins_ind[count]).zfill(6) + '_10.png'))
    instance_semantic_gt = np.array(instance_semantic_gt).astype(np.uint16)
    instance_gt = instance_semantic_gt % 256

    gtscale_intrinsic = np.array([
        [0.58 * gtwidth, 0, 0.5 * gtwidth],
        [0, 1.92 * gtheight, 0.5 * gtheight],
        [0, 0, 1]], dtype=np.float32)
    descriptor = LocalThetaDesp(height=gtheight, width=gtwidth, batch_size=1, intrinsic=gtscale_intrinsic).cuda()

    data = dataset.__getitem__(count)
    htheta = data['htheta'].unsqueeze(0).cuda()
    vtheta = data['vtheta'].unsqueeze(0).cuda()

    htheta_gtsize = F.interpolate(htheta, [gtheight, gtwidth], mode='bilinear', align_corners=True)
    vtheta_gtsize = F.interpolate(vtheta, [gtheight, gtwidth], mode='bilinear', align_corners=True)

    input_color = data[("color", 0, 0)].unsqueeze(0).cuda()
    input_color_gtsize = F.interpolate(input_color, [gtheight, gtwidth], mode='bilinear', align_corners=True)

    htheta_gtsize = htheta_gtsize / 2 / np.pi
    vtheta_gtsize = vtheta_gtsize / 2 / np.pi
    htheta_gtsize_act = torch.log(htheta_gtsize / (1 - htheta_gtsize))
    vtheta_gtsize_act = torch.log(vtheta_gtsize / (1 - vtheta_gtsize))

    htheta_gtsize_act.requires_grad = True
    vtheta_gtsize_act.requires_grad = True

    optimizer = optim.Adam([htheta_gtsize_act] + [vtheta_gtsize_act], lr=1e-3)

    loss_rec = list()
    for kk in range(200000):
        htheta_gtsize = torch.sigmoid(htheta_gtsize_act) * 2 * np.pi
        vtheta_gtsize = torch.sigmoid(vtheta_gtsize_act) * 2 * np.pi

        optimizedDepth_torch = descriptor.vls_geompred(torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0).cuda().float(), htheta_gtsize, vtheta_gtsize, input_color_gtsize, gt_depth, eng=eng, instancemap=instance_gt)

        sel = (semidense_depth_torch > 0).float()
        loss = torch.sum(torch.abs(semidense_depth_torch - optimizedDepth_torch) * sel) / torch.sum(sel)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        loss_rec.append(float(loss.detach().cpu().numpy()))

        print("Iteration %d, loss: %f" % (kk, loss.detach().cpu().numpy()))

    plt.figure()
    plt.plot(list(range(len(loss_rec))), loss_rec)
    plt.xlabel('Iteration Number')
    plt.ylabel('Mean absolute difference')
    plt.title('Optimization curve')

    hthetaopted, vthetaopted = descriptor.get_theta(optimizedDepth_torch)
    tensor2disp(1 / optimizedDepth_torch, vmax=0.2, ind=0).show()
    tensor2disp(htheta_gtsize - 1, vmax=4, ind=0).show()
    tensor2disp(htheta - 1, vmax=4, ind=0).show()
    tensor2disp(vtheta_gtsize - 1, vmax=4, ind=0).show()
    tensor2disp(hthetaopted - 1, vmax=4, ind=0).show()
if __name__ == "__main__":
    options = MonodepthOptions()
    args = options.parse()
    evaluate(args)

