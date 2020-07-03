from __future__ import absolute_import, division, print_function

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import *
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import glob

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "..", "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    vls_intensity = 100

    loadbz = 1

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.split))
    filenames = filenames[::vls_intensity]
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                       encoder_dict['height'], encoder_dict['width'],[0], 4, is_train=False, theta_gt_path=opt.theta_gt_path)
    dataloader = DataLoader(dataset, loadbz, shuffle=False, num_workers=opt.num_workers, pin_memory=True, drop_last=True)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, num_output_channels=3)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    predw = 1024
    predh = 320
    intrinsicKitti = np.array([
        [0.58 * predw, 0, 0.5 * predw],
        [0, 1.92 * predh, 0.5 * predh],
        [0, 0, 1]], dtype=np.float32)
    localthetadesp = LocalThetaDesp(height=predh, width=predw, batch_size=loadbz, intrinsic=intrinsicKitti).cuda()


    output_folder = "/media/shengjie/c9c81c9f-511c-41c6-bfe0-2fc19666fb32/Visualizations/Project_SemanDepth/vls_models"
    output_subfolder = os.path.join(output_folder, opt.load_weights_folder.split('/')[-2])
    os.makedirs(output_subfolder, exist_ok=True)

    count = 0
    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()
            output = depth_decoder(encoder(input_color))

            _, pred_depth = disp_to_depth(output[("disp", 0)][:,2:3,:,:], opt.min_depth, opt.max_depth)
            preddepth = pred_depth * STEREO_SCALE_FACTOR

            htheta = data['htheta'].cuda()
            vtheta = data['vtheta'].cuda()

            derivx, num_grad = localthetadesp.depth_localgeom_consistency(preddepth, htheta, vtheta, isoutput_grads=True)
            dir3d = localthetadesp.surfnorm_from_localgeom(preddepth, htheta, vtheta)
            dir3d = torch.clamp(((dir3d + 1) / 2), min=0, max=1)
            dir3d = dir3d.permute([0,3,1,2]).contiguous()
            fig3 = tensor2rgb(dir3d, ind=0)

            fig1 = tensor2disp(torch.abs(num_grad), vmax=0.1, ind=0)
            fig2 = tensor2disp(torch.abs(derivx), vmax=0.1, ind=0)
            fig4 = tensor2disp(htheta-1, ind=0, vmax=4)
            fig5 = tensor2disp(output[("disp", 0)][:,2:3,:,:], ind=0, vmax=0.1)
            fig6 = tensor2rgb(input_color, ind=0)

            comps = filenames[count].split(' ')
            filename = comps[0].split('/')[1] + '_' + str(comps[1]).zfill(10) + '.png'
            figcombined = pil.fromarray(np.concatenate([np.array(fig1), np.array(fig2), np.array(fig3), np.array(fig4), np.array(fig5), np.array(fig6)], axis=0))
            figcombined.save(os.path.join(output_subfolder, filename))

            count = count + 1
            print("%d finished" % count)


if __name__ == "__main__":
    options = MonodepthOptions()
    args = options.parse()
    evaluate(args)

