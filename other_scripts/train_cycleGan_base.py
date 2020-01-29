import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from options import MonodepthOptions
from datasets import create_dataset
from networks import create_model
from tensorboardX import SummaryWriter
import torch
import time
import os
from utils import tensor2rgb

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
    opts.verbose = True
    opts.no_dropout = True
    opts.continue_train = False
    opts.save_by_iter = False
    return opts

if __name__ == "__main__":
    opts = init_opts(opts)
    dataset = create_dataset(opts)
    dataset_size = len(dataset)  # get the number of images in the dataset.

    model = create_model(opts)  # create a model given opt.model and other options
    model.setup(opts)  # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    sum_writers = SummaryWriter(os.path.join(opts.log_dir, opts.model_name, 'train'))

    os.makedirs(os.path.join(opts.log_dir, opts.model_name, 'model'), exist_ok=True)

    train_start_time = time.time()

    for epoch in range(opts.epoch_count, opts.n_epochs + opts.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration

            total_iters += opts.batch_size
            epoch_iter += opts.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opts.print_freq == 0:
                losses = model.get_current_losses()
                for l, v in losses.items():
                    sum_writers.add_scalar(l, v, total_iters)

            if total_iters % opts.display_freq == 0:   # display images on visdom and save images to a HTML file
                to_visuals = model.get_current_visuals()
                for l, v in to_visuals.items():
                    sum_writers.add_image(l, (v[0, :, :, :] + 1) / 2, total_iters)
                train_time = time.time() - train_start_time
                print("Epoch %d, time left %f hours" % (epoch, train_time / total_iters * dataset.__len__() * (opts.n_epochs + opts.n_epochs_decay) / 60 / 60))

            if total_iters % opts.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opts.save_by_iter else 'latest'
                model.save_networks(save_suffix)

        if epoch % opts.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()                     # update learning rates at the end of every epoch.
