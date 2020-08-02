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

import cv2
import numpy as np
import torch
from scipy import fftpack

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "..", "splits")

STEREO_SCALE_FACTOR = 5.4

def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    plt.imshow(np.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

def inverse_mapping(mappings):
    inversemap = dict()
    for map in mappings:
        seqvrkitti, seqrawkitti = map.split('\t')
        inversemap[seqrawkitti[1::]] = seqvrkitti[0:-1]
    return inversemap

def imgsHasVrkittiMapping(filenames, mappings):
    keylist = list(mappings.keys())
    valframe = list()
    for fn in filenames:
        seq, frame, dir = fn.split(' ')
        if seq in keylist and dir == 'l':
            valframe.append(fn)
    return valframe

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    vrkitti_root = '/home/shengjie/Documents/Data/virtual_kitti_organized'

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, 'eigen_full', "train_files.txt"))
    mappings = readlines('/home/shengjie/Documents/Data/virtual_kitti_organized/mapping.txt')
    mappings = inverse_mapping(mappings)
    valframe = imgsHasVrkittiMapping(filenames, mappings)

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = datasets.KITTIRAWDataset(opt.data_path, valframe, encoder_dict['height'], encoder_dict['width'], [0], 4,
                                       is_train=False, theta_gt_path=opt.theta_gt_path, load_seman=True)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, num_output_channels=3)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> Computing predictions with size {}x{}".format(encoder_dict['width'], encoder_dict['height']))

    predw = encoder_dict['width']
    predh = encoder_dict['height']
    thetalossmap = torch.zeros([1, 1, predh, predw]).expand([opt.batch_size, -1, -1, -1]).cuda()
    thetalossmap[:,:,110::,:] = 1

    # import matlab.engine
    # eng = matlab.engine.start_matlab()

    count = np.random.randint(0, len(valframe))
    # count = 344

    data = dataset.__getitem__(count)
    input_color = data[("color", 0, 0)].unsqueeze(0).cuda()

    output = depth_decoder(encoder(input_color))

    htheta = data['htheta'].unsqueeze(0).cuda()
    vtheta = data['vtheta'].unsqueeze(0).cuda()
    _, preddepth = disp_to_depth(output[("disp", 0)][:,2:3,:,:], opt.min_depth, opt.max_depth)
    preddepth = preddepth * STEREO_SCALE_FACTOR

    semantics = torch.from_numpy(data['semanLabel']).unsqueeze(0).float()

    gt_depth = gt_depths[count]
    gtheight, gtwidth = gt_depth.shape

    gtscale_intrinsic = np.array([
        [0.58 * gtwidth, 0, 0.5 * gtwidth],
        [0, 1.92 * gtheight, 0.5 * gtheight],
        [0, 0, 1]], dtype=np.float32)
    depcriptor = LocalThetaDesp(height=gtheight, width=gtwidth, batch_size=1, intrinsic=gtscale_intrinsic).cuda()

    preddepth_gtsize = F.interpolate(preddepth, [gtheight, gtwidth], mode='bilinear', align_corners=True)
    htheta_gtsize = F.interpolate(htheta, [gtheight, gtwidth], mode='bilinear', align_corners=True)
    vtheta_gtsize = F.interpolate(vtheta, [gtheight, gtwidth], mode='bilinear', align_corners=True)
    input_color_gtsize = F.interpolate(input_color, [gtheight, gtwidth], mode='bilinear', align_corners=True)
    semantics_gtsize = F.interpolate(semantics, [gtheight, gtwidth], mode='nearest')

    hthetaFromPred, _ = depcriptor.get_theta(preddepth_gtsize)

    htehta_np = htheta.detach().cpu().numpy()[0, 0, :, :]
    hthetaFromPred_np = hthetaFromPred.detach().cpu().numpy()[0, 0, :, :]

    htehta_np_cropped = htehta_np[int(0.40810811 * gtheight):int(0.99189189 * gtheight), int(0.03594771 * gtwidth):int(0.96405229 * gtwidth)]
    hthetaFromPred_np_cropped = hthetaFromPred_np[int(0.40810811 * gtheight):int(0.99189189 * gtheight), int(0.03594771 * gtwidth):int(0.96405229 * gtwidth)]

    htehta_np_cropped_fft = fftpack.fft2(htehta_np_cropped)
    hthetaFromPred_np_cropped_fft = fftpack.fft2(hthetaFromPred_np_cropped)

    # Input virtual kitti data
    seq, frame, dir = valframe[count].split(' ')
    rgb_vrkitti = pil.open(os.path.join(vrkitti_root, mappings[seq], 'rgb', frame.zfill(5) + '.png'))
    depth_vrkitti = pil.open(os.path.join(vrkitti_root, mappings[seq], 'depthgt', frame.zfill(5) + '.png'))
    depth_vrkitti = torch.from_numpy(np.array(depth_vrkitti).astype(np.float32) / 256).unsqueeze(0).unsqueeze(0).cuda()
    # rgb_vrkitti = F.interpolate(rgb_vrkitti, [gtheight, gtwidth], mode='bilinear', align_corners=True)
    depth_vrkitti = F.interpolate(depth_vrkitti, [gtheight, gtwidth], mode='bilinear', align_corners=True)

    htheta_vrkitti, vtheta_vrkitti = depcriptor.get_theta(depth_vrkitti)
    htheta_vrkitti_np = htheta_vrkitti.detach().cpu().numpy()[0, 0, :, :]
    htheta_vrkitti_np_cropped = htheta_vrkitti_np[int(0.40810811 * gtheight):int(0.99189189 * gtheight), int(0.03594771 * gtwidth):int(0.96405229 * gtwidth)]
    htheta_vrkitti_np_cropped_fft = fftpack.fft2(htheta_vrkitti_np_cropped)

    tensor2disp(1 / depth_vrkitti, vmax = 0.2, ind=0).show()
    tensor2disp(htheta_vrkitti - 1, vmax=4, ind=0).show()
    tensor2disp(htheta_gtsize - 1, vmax=4, ind=0).show()
    tensor2rgb(input_color_gtsize, ind=0).show()

    plt.figure()
    plot_spectrum(htehta_np_cropped_fft)
    plt.title('Fourier transform')

    plt.figure()
    plot_spectrum(hthetaFromPred_np_cropped_fft)
    plt.title('Fourier transform')

    plt.figure()
    plot_spectrum(htheta_vrkitti_np_cropped_fft)
    plt.title('Fourier transform')



if __name__ == "__main__":
    options = MonodepthOptions()
    args = options.parse()
    evaluate(args)

