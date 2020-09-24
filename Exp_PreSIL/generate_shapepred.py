from __future__ import absolute_import, division, print_function

import os, sys, inspect
project_rootdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
sys.path.insert(0, project_rootdir)

import torch.optim as optim
from layers import *
from PIL import Image, ImageFile
import torch.utils.data as data
from torchvision import transforms
from utils import *
import networks


import numpy as np
import copy
import time
import argparse
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True # This can prevent some weried error

default_logpath = os.path.join(project_rootdir, 'tmp')
parser = argparse.ArgumentParser(description='Train Dense Depth of PreSIL Synthetic Data')
parser.add_argument("--data_path",                  type=str,                               help="path to dataset")
parser.add_argument("--output_path",                 type=str,                               help="path to dataset")
parser.add_argument("--num_layers",                 type=int,   default=18,                 help="number of resnet layers", choices=[18, 34, 50, 101, 152])
parser.add_argument("--height",                     type=int,   default=320,                help="input image height")
parser.add_argument("--width",                      type=int,   default=1024,               help="input image width")
parser.add_argument("--crph",                       type=int,   default=365,                help="cropped image height")
parser.add_argument("--crpw",                       type=int,   default=1220,               help="cropped image width")


# OPTIMIZATION options
parser.add_argument("--load_weights_folder_norm",   type=str,   default=None,               help="name of models to load")

ImageFile.LOAD_TRUNCATED_IMAGES = True # This can prevent some weried error
class ShapepredDataset(data.Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 crph=365,
                 crpw=1220
                 ):
        super(ShapepredDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width

        self.to_tensor = transforms.ToTensor()
        self.interp = Image.ANTIALIAS
        self.img_ext = '.png'

        self.dirmapping = {'l': 'image_02', 'r': 'image_03'}

        self.crph = crph
        self.crpw = crpw

        self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)

    def preprocess(self, inputs, color_aug, rndseed):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        # Do random Crop
        cropped_color = self.rndcrop_color(inputs["color"], rndseed)

        inputs["color"] = self.resize(cropped_color)

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                inputs[k] = self.to_tensor(f)
                inputs[k + "_aug"] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}

        line = self.filenames[index].split()

        folder = line[0]
        frame_index = int(line[1])
        side = line[2]

        inputs.update(self.get_color(folder, frame_index, side))
        return inputs

    def get_color(self, folder, frame_index, side):
        inputs = dict()
        color = pil.open(os.path.join(self.data_path, folder, self.dirmapping[side], "data/{}.png".format(str(frame_index).zfill(10))))
        colornp = np.array(color)

        colorflipped = color.transpose(pil.FLIP_LEFT_RIGHT)
        colornpflipped = np.array(colorflipped)

        rgbs, weights = self.getpredmask(colornp)
        rgbsflipped, weightsflipped = self.getpredmask(colornpflipped)

        rgbs = torch.from_numpy(rgbs).float().permute([0,3,1,2]) / 255.0
        rgbs = F.interpolate(rgbs, [self.height, self.width], mode='bilinear', align_corners=True)

        weights = torch.from_numpy(weights).float()

        rgbsflipped = torch.from_numpy(rgbsflipped).float().permute([0,3,1,2]) / 255.0
        rgbsflipped = F.interpolate(rgbsflipped, [self.height, self.width], mode='bilinear', align_corners=True)

        weightsflipped = torch.from_numpy(weightsflipped).float()

        inputs['rgbs'] = rgbs
        inputs['rgbsflipped'] = rgbsflipped
        inputs['weights'] = weights
        inputs['weightsflipped'] = weightsflipped

        return inputs

    def getpredmask(self, rgbnp):
        gth, gtw, _ = rgbnp.shape

        rgbs = np.stack([rgbnp[:self.crph, :self.crpw, :], rgbnp[gth-self.crph:, gtw-self.crpw:, :], rgbnp[:self.crph, gtw-self.crpw:, :], rgbnp[gth-self.crph:, :self.crpw, :]], axis=0)

        weights = np.zeros([gth, gtw])
        weights[:self.crph, :self.crpw] = weights[:self.crph, :self.crpw] + 1
        weights[:self.crph, gtw-self.crpw:] = weights[:self.crph, gtw-self.crpw:] + 1
        weights[gth-self.crph:, gtw-self.crpw:] = weights[gth-self.crph:, gtw-self.crpw:] + 1
        weights[gth-self.crph:, :self.crpw] = weights[gth-self.crph:, :self.crpw] + 1

        return rgbs, weights

class Trainer:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}

        self.device = "cuda"

        self.models["encoder_norm"] = networks.ResnetEncoder(self.opt.num_layers, pretrained=True)
        self.models["encoder_norm"].to(self.device)
        self.models["norm"] = networks.DepthDecoder(self.models["encoder_norm"].num_ch_enc, num_output_channels=2)
        self.models["norm"].to(self.device)

        self.set_dataset()

        self.load_model()

        self.crph = 365
        self.crpw = 1220

        os.makedirs(self.opt.output_path, exist_ok=True)
        self.dirmapping = {'l': 'image_02', 'r': 'image_03'}
    def set_dataset(self):
        """properly handle multiple dataset situation
        """

        def get_entry_from_path(imgpath):
            comps = imgpath.split('/')
            if comps[-3] == 'image_02':
                direct = 'l'
            else:
                direct = 'r'
            entry = comps[-5] + '/' + comps[-4] + ' ' + comps[-1].split('.')[0] + ' ' + direct
            return entry

        import glob
        dates = [f.path for f in os.scandir(self.opt.data_path) if f.is_dir()]
        entries = list()
        for date in dates:
            seqs = [f.path for f in os.scandir(date) if f.is_dir()]
            for seq in seqs:
                imgFolder = os.path.join(seq, 'image_02/data')
                for imgpath in glob.glob(imgFolder + '/*.png'):
                    entries.append(get_entry_from_path(imgpath))
        self.shapepreddataset = ShapepredDataset(data_path=self.opt.data_path, filenames=entries, height=self.opt.height, width=self.opt.width)

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def save_ang_pred(self, inputs, inputname, idx):
        wmapping = {'rgbs': 'weights', 'rgbsflipped': 'weightsflipped'}
        anghfoldnamemapping = {'rgbs': 'angh', 'rgbsflipped': 'angh_flipped'}
        angvfoldnamemapping = {'rgbs': 'angv', 'rgbsflipped': 'angv_flipped'}
        gth, gtw = inputs[wmapping[inputname]].shape
        with torch.no_grad():
            outputs_ang = self.models['norm'](self.models['encoder_norm'](inputs[inputname]))

            pred_ang_netsize = outputs_ang[("disp", 0)]
            pred_ang_cropsize = F.interpolate(pred_ang_netsize, [self.crph, self.crpw], mode='bilinear', align_corners=True)
            pred_ang = torch.zeros([2, gth, gtw], device='cuda')

            pred_ang[:, :self.crph, :self.crpw] += pred_ang_cropsize[0]
            pred_ang[:, gth - self.crph:, gtw - self.crpw:] += pred_ang_cropsize[1]
            pred_ang[:, :self.crph, gtw - self.crpw:] += pred_ang_cropsize[2]
            pred_ang[:, gth - self.crph:, :self.crpw] += pred_ang_cropsize[3]
            pred_ang = (pred_ang / inputs[wmapping[inputname]])

            pred_angnp = (pred_ang.detach().cpu().numpy() * 255 * 255).astype(np.uint16)
            pred_angnph = pred_angnp[0]
            pred_angnpv = pred_angnp[1]

            seq, frame, dir = self.shapepreddataset.filenames[idx].split(' ')
            svpathh_fold = os.path.join(self.opt.output_path, anghfoldnamemapping[inputname], seq, self.dirmapping[dir])
            svpathv_fold = os.path.join(self.opt.output_path, angvfoldnamemapping[inputname], seq, self.dirmapping[dir])
            os.makedirs(svpathh_fold, exist_ok=True)
            os.makedirs(svpathv_fold, exist_ok=True)
            pil.fromarray(pred_angnph).save(os.path.join(svpathh_fold, '{}.png'.format(frame.zfill(10))))
            pil.fromarray(pred_angnpv).save(os.path.join(svpathv_fold, '{}.png'.format(frame.zfill(10))))
        return pred_ang


    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        st = time.time()
        for idx in range(self.shapepreddataset.__len__()):
            inputs = self.shapepreddataset.__getitem__(idx)

            for key, ipt in inputs.items():
                inputs[key] = ipt.to(self.device)

            self.save_ang_pred(inputs, 'rgbsflipped', idx)
            self.save_ang_pred(inputs, 'rgbs', idx)

            dur = time.time() - st
            remainhours = (self.shapepreddataset.__len__() - idx - 1) * (dur / (idx + 1)) / 60 / 60
            print("Finsh ind: %d, remain hours: %f" % (idx, remainhours))

    def load_model(self):
        """Load model(s) from disk
        """
        load_norm_folder = os.path.expanduser(self.opt.load_weights_folder_norm)
        assert os.path.isdir(load_norm_folder), "Cannot find folder {}".format(self.opt.load_weights_folder_norm)
        models_to_load = ['encoder_norm', 'norm']
        pthfilemapping = {'encoder_norm': 'encoder', 'norm': 'depth'}
        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder_norm, "{}.pth".format(pthfilemapping[n]))
            if n in self.models:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)

if __name__ == "__main__":
    trainer = Trainer(parser.parse_args())
    trainer.val()
