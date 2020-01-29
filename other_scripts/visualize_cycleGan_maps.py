import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from options import MonodepthOptions
from datasets import create_dataset
from networks import create_model
from utils import tensor2rgb
import torch
import time
import os

torch.backends.cudnn.benchmark = True
options = MonodepthOptions()
opts = options.parse()

def init_opts(opts):
    opts.phase = 'train'
    opts.lr_policy = 'linear'
    opts.norm = 'instance'
    opts.init_type = 'normal'
    opts.netG = 'resnet_9blocks'
    opts.netD = 'basic'
    opts.gan_mode = 'lsgan'
    opts.load_epoch = 'latest' # which epoch to load? set to latest to use latest cached model

    opts.gpu_ids = [0]

    opts.lambda_identity = 0.5
    opts.lambda_A = 10.0
    opts.lambda_B = 10.0
    opts.input_nc = 3
    opts.output_nc = 3
    opts.ngf = 64
    opts.ndf = 64
    opts.init_gain = 0.02
    opts.n_layers_D = 3
    opts.pool_size = 50
    opts.beta1 = 0.5
    opts.load_iter = 0
    opts.epoch_count = 1
    opts.n_epochs = 100 # number of epochs with the initial learning rate
    opts.n_epochs_decay = 100 # number of epochs to linearly decay learning rate to zero
    opts.print_freq = 10
    opts.display_freq = 400
    opts.save_epoch_freq = 5
    opts.save_latest_freq = 5000

    opts.isTrain = True
    opts.verbose = False
    opts.no_dropout = True
    opts.continue_train = False
    opts.save_by_iter = False

    opts.mergedModel_path = '/media/shengjie/other/Depins/Depins/tmp/cycleGan_maps_merged/model'
    opts.orgModel_path = '/media/shengjie/other/Depins/Depins/tmp/cycleGan_maps_org/model'
    return opts

def visualize_results(model):
    visualize_dict = [['real_A', 'fake_A', 'rec_A'], ['real_B', 'fake_B', 'rec_B']]
    figlist = list()
    for cat in visualize_dict:
        tmplist = list()
        for re in cat:
            x = getattr(model, re)
            tmplist.append(x)
        tmplist = torch.cat(tmplist, dim=3)
        figlist.append(tmplist)
    figlist = torch.cat(figlist, dim=2)
    return figlist
if __name__ == "__main__":
    opts = init_opts(opts)
    dataset = create_dataset(opts)
    dataset_size = len(dataset)  # get the number of images in the dataset.

    model_org = create_model(opts)  # create a model given opt.model and other options
    model_org.setup(opts)  # regular setup: load and print networks; create schedulers
    model_org.load_networks(opts.orgModel_path)
    model_org.eval()

    model_merged = create_model(opts)  # create a model given opt.model and other options
    model_merged.setup(opts)  # regular setup: load and print networks; create schedulers
    model_merged.load_networks(opts.mergedModel_path)
    model_merged.eval()

    visualize_root = '/media/shengjie/other/Depins/Depins/visualization/cycleGan_maps'

    tot_vis = 20
    with torch.no_grad():
        for i, data in enumerate(dataset):  # inner loop within one epoch
            model_org.set_input(data)         # unpack data from dataset and apply preprocessing
            model_org.forward()
            fig_org = visualize_results(model_org)

            model_merged.set_input(data)
            model_merged.forward()
            fig_merged = visualize_results(model_merged)

            fig_compare = torch.cat([fig_org, fig_merged], dim=2)
            tensor2rgb((fig_compare + 1) / 2, ind=0).save(os.path.join(visualize_root, str(i).zfill(10) + '.png'))

            if i > tot_vis:
                break