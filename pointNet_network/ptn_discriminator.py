import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import torch
import itertools
from pointNet_network.ptn_pool import PtnPool
from networks.cycle_gan_model import BaseModel
from networks import networks
from utils import *
from layers import *
from pointNet_network.pointNet_model import PointNetCls

class PtnD(nn.Module):

    def __init__(self, opt, k = 1, feature_transform = False):
        super(PtnD, self).__init__()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.gpu_ids = [0]
        self.device = torch.device("cuda")
        self.opt = opt
        self.netD = PointNetCls(k=k, feature_transform=feature_transform)
        self.eps = 1e-5

        self.fake_pool = PtnPool(opt.pool_size, batch_size=self.opt.batch_size)  # create image buffer to store previously generated images
        # define loss functions
        self.criterionGAN = networks.GANLoss('vanilla', reduction='none').to(self.device)  # define GAN loss.
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lrD, betas=(opt.beta1, 0.999))
        # self.optimizer_D = torch.optim.SGD(itertools.chain(self.netD.parameters()), lr=1e-5)
        # self.optimizer_D = torch.optim.SGD(list(self.netD.parameters()), lr=1e-5)

        # self.compute_sfnorm = dict()
        # for i in self.opt.scales:
        #     self.compute_sfnorm[('scale', i)] = ComputeSurfaceNormal(batch_size=self.opt.batch_size, height=int(self.opt.height / np.power(2, i)), width=int(self.opt.width / np.power(2, i)), minDepth=self.opt.min_depth, maxDepth=self.opt.max_depth).cuda()

        self.set_requires_grad(self.netD, requires_grad=True)
        # self.init_weights(self.netD)

        self.mean = torch.Tensor([25, 0, 0]).unsqueeze(0).unsqueeze(2).expand([self.opt.batch_size,-1,10000]).cuda()
        self.std = torch.Tensor([25, 10, 5]).unsqueeze(0).unsqueeze(2).expand([self.opt.batch_size,-1,10000]).cuda()
        self.pts_num = 10000
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
    def set_input(self, real, realv, syn, synv):
        self.real = real
        self.realv = realv

        self.syn = syn
        self.synv = synv

        # Normalization
        self.real = (self.real - self.mean) / self.std
        self.syn = (self.syn - self.mean) / self.std

        # print(torch.mean(self.real, dim=[1,2]))
        # print(torch.mean(self.syn, dim=[1, 2]))

    # def init_weights(self, nets):
    #     if not isinstance(nets, list):
    #         nets = [nets]
    #     for net in nets:
    #         if net is not None:
    #             for sin_modules in list(net.modules()):
    #                 if type(sin_modules) == nn.Linear:
    #                     sin_modules.weight = torch.nn.init.xavier_uniform_(sin_modules.weight)


    def forward(self):
        self.set_requires_grad(self.netD, requires_grad=False)
        pred_real = self.netD.discriminator_forward(self.real)
        loss_D_real = torch.sum(self.criterionGAN(pred_real, False) * self.realv) / (torch.sum(self.realv) + self.eps)
        return loss_D_real

    def optimize_parameters(self):
        self.set_requires_grad(self.netD, requires_grad=True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero

        # Real
        # real, realv = self.fake_pool.query(pts=self.real, ptsv=self.realv)
        # pred_real = self.netD.discriminator_forward(real.detach())
        # loss_D_real = torch.sum(self.criterionGAN(pred_real, True) * realv) / (torch.sum(realv) + self.eps)

        # real, realv = self.fake_pool.query(pts=self.real, ptsv=self.realv)
        real = self.real.detach()
        realv = self.realv.detach()
        pred_real = self.netD.discriminator_forward(real.detach())
        loss_D_real = torch.sum(self.criterionGAN(pred_real, True) * realv) / (torch.sum(realv) + self.eps)


        # Fake
        pred_fake = self.netD.discriminator_forward(self.syn.detach())
        loss_D_fake = torch.sum(self.criterionGAN(pred_fake, False) * self.synv) / (torch.sum(self.synv) + self.eps)

        # Combined loss and calculate gradients
        # loss_D = (loss_D_real + loss_D_fake) * 0.5
        # loss_D = torch.mean((pred_real - 1) * (pred_real - 1) + (pred_fake - 0) * (pred_fake - 0))
        loss_D = torch.mean((pred_real - 1) * (pred_real - 1))

        print(loss_D)
        print(pred_real)
        print(pred_fake)
        loss_D.backward()
        self.optimizer_D.step()  # update D_A and D_B's weights


        # a,_,_ = self.netD.feat(self.real.detach())
        # a = F.relu(self.netD.bn1(self.netD.fc1(a)))
        # b,_,_ = self.netD.feat(self.syn.detach())
        # b = F.relu(self.netD.bn1(self.netD.fc1(b)))
        # print(torch.mean(real, dim=[1,2]))
        # print(torch.mean(self.syn, dim=[1, 2]))

        return loss_D