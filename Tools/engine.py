import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import tqdm
from Model import *


class Inpainting_Engine():
    def __init__(self, train_loader, eval_loader, writer, args, device):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.writer = writer
        self.args = args
        self.device = device
        self.count = 0
        # Generator
        self.netDFBN = DFBN().to(self.device)
        self.netDFBN.apply(self.init_weights)

        if self.args.mode in ['eval', 'test']:
            return
        # Discriminator
        self.netD = GlobalLocalDiscriminator(3, cnum=self.args.d_cnum, act=F.leaky_relu).to(self.device)
        self.netDFBN.apply(self.init_weights)

        self.zeros = torch.zeros((self.args.batch_size, 1)).to(self.device)
        self.ones = torch.ones((self.args.batch_size, 1)).to(self.device)
        
        def weight_fn(layer):
            s = layer.shape
            return 1e3 / (s[1] * s[1] * s[1] * s[2] * s[3])
        self.weight_fn = weight_fn
        # Optimizer
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netDFBN.parameters(), lr=self.args.lr, betas=(0.5, 0.9))
        self.optimizers += [self.optimizer_G]
        self.optimizer_D = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netD.parameters()), lr=self.args.lr, betas=(0.5, 0.9))
        self.optimizers += [self.optimizer_D]
        # Schedulers
        self.schedulers = []
        for opt in self.optimizers:
            self.schedulers.append(lr_scheduler.MultiStepLR(opt, self.args.schedule_milestone, 0.5))

        # Losses
        self.loss_lambda = 25
        self.loss_eta = 5
        self.loss_mu = 0.03
        self.loss_vgg = 1

        self.vggloss = VGGLoss()
        self.aeloss = nn.L1Loss()
        self.BCEloss = nn.BCEWithLogitsLoss().cuda()

    def Dra(self, x1, x2):
        return x1 - torch.mean(x2)

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)
            try:
                nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            try:
                nn.init.constant_(m.bias, 0)
            except AttributeError:
                pass

    def optimize_parameters(self, gin, gt, mask_01):
        self.min_loss = 1e5
        self.pred = self.netDFBN(gin)
        self.completed = self.pred * mask_01 + gt * (1 - mask_01)
        # define local:
        self.gt_local = gt * mask_01
        self.completed_local = self.completed * mask_01

        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        self.forward_G()
        self.G_loss.backward()

        
    def forward_G(self):
        self.G_loss_ae = self.aeloss(self.completed_local, self.gt_local)

        # vgg loss
        mask_error = torch.mean(F.mse_loss(self.completed_local, self.gt_local, reduction='none'), dim=1)
        mask_max = mask_error.max(1, True)[0].max(2, True)[0]
        mask_min = mask_error.min(1, True)[0].min(2, True)[0]
        mask_guidance = (mask_error - mask_min) / (mask_max - mask_min)
        self.G_loss_vgg = self.vggloss(self.completed_local, self.gt_local.detach(), mask_guidance.detach(), self.weight_fn)

        # adv loss
        xf = self.netD('adv', self.completed, self.completed_local)
        xr = self.netD('adv', self.gt, self.gt_local)
        self.G_loss_adv = (self.BCEloss(self.Dra(xr, xf), self.zeros) + self.BCEloss(self.Dra(xf, xr), self.ones)) / 2
        
        # fm dis loss
        self.G_loss_fm_dis = self.netD('fm_dis', self.gt_local, self.completed_local, self.weight_fn)
        self.G_loss = self.G_loss_ae + self.loss_vgg * self.G_loss_vgg + self.loss_mu * self.G_loss_adv + self.loss_eta * self.G_loss_fm_dis

    def forward_D(self):
        xf = self.netD('dis', self.completed.detach(), self.completed_local.detach())
        xr = self.netD('dis', self.gt, self.gt_local)
        # hinge loss
        self.D_loss = (self.BCEloss(self.Dra(xr, xf), self.ones) + self.BCEloss(self.Dra(xf, xr), self.zeros)) / 2

    def backward_G(self):
        self.G_loss.backward()
        