import torch
import itertools
from util.image_pool import ImagePool
from .cycle_gan_base_model import BaseModel
from . import networks
from utils import *
from layers import *

class SfnD(nn.Module):

    def __init__(self, opt):
        super(SfnD, self).__init__()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.gpu_ids = [0]
        self.device = torch.device("cuda")
        self.opt = opt
        self.netD = networks.define_D(3, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        self.fake_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
        # define loss functions
        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lrD, betas=(opt.beta1, 0.999))

        self.compute_sfnorm = dict()
        for i in self.opt.scales:
            self.compute_sfnorm[('scale', i)] = ComputeSurfaceNormal(batch_size=self.opt.batch_size, height=int(self.opt.height / np.power(2, i)), width=int(self.opt.width / np.power(2, i)), minDepth=self.opt.min_depth, maxDepth=self.opt.max_depth).cuda()

        self.set_requires_grad(self.netD, requires_grad=True)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    def set_input(self, real, syn, inv_camK, scale):
        self.real = real
        self.syn = syn
        self.inv_camK = inv_camK
        self.scale = scale

        self.real_sfnorm = self.compute_sfnorm[('scale', self.scale)](self.real, self.inv_camK)
        self.syn_sfnorm = self.compute_sfnorm[('scale', self.scale)](self.syn, self.inv_camK)

    def forward(self):
        pred_real = self.netD(self.real_sfnorm)
        loss_D_real = self.criterionGAN(pred_real, False)
        return loss_D_real

    def optimize_parameters(self):
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        # Real
        pred_real = self.netD(self.real_sfnorm.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = self.netD(self.syn_sfnorm.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        self.optimizer_D.step()  # update D_A and D_B's weights