import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import torchvision
from torchvision import  models

from tensorboardX import SummaryWriter

import numpy as np
import time
import os
import copy
import pickle
from skimage import color,img_as_ubyte

import matplotlib.pyplot as plt
import csv
import pandas as pd
import skimage.io as io
import random 
from PIL import Image, ImageChops

from IPython.display import HTML

from Tools import Args, get_files, CelebA_Pre
from tensorboardX import SummaryWriter
from Dataset import CelebA
from Model.DC_GAN_v4 import *
import torch.utils.data as data
from torchsummary import summary
plt.ion()   # interactive mode

args = Args('./Experiments/configs/demo_config.json')
SAVE_PATH = '/home/wchai01/minmin/'
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_optimizer(net, lr=1e-4):
    optimizer = torch.optim.Adam(net.parameters(), lr, betas=(0.5, 0.999))
    return optimizer

def imshow(masked, gen, gd, idx=0, epoch=0):


    masked = masked.detach().numpy().transpose((1,2,0))
    gen = gen.detach().numpy().transpose((1,2,0))
    gd = gd.detach().numpy().transpose((1,2,0))

    mean = np.array([0.5,0.5,0.5])
    std = np.array([0.5,0.5,0.5])

    gen = std * gen + mean
    masked = std * masked + mean
    gd = gd * std + mean
    # normalize to (0,1)
    # for i in range(8):
    #     min_masked = np.min(masked[i])
    #     max_masked = np.max(masked[i])
    #     masked[i] = (masked[i] - min_masked) / (max_masked - min_masked)

    #     min_gen = np.min(gen[i])
    #     max_gen = np.max(gen[i])
    #     gen[i] = (gen[i] - min_gen) / (max_gen - min_gen)

    #     min_gd = np.min(gd[i])
    #     max_gd = np.max(gd[i])
    #     gd[i] = (gd[i] - min_gd) / (max_gd - min_gd)
    
    fig = plt.figure(figsize = (20,10))
    # gen = np.clip(gen, 0, 1)
    plt.subplot(3, 1, 1)
    plt.imshow(masked)
    plt.subplot(3, 1, 2)
    plt.imshow(gen)
    plt.subplot(3, 1, 3)
    plt.imshow(gd)
    print("save results...")
    fig.savefig(SAVE_PATH+'DC_GAN_results/DC_GAN_V4/'+'epoch_'+str(epoch)+'batch_'+str(idx)+'.png')


if __name__ == "__main__":
    # open a file to record the process of training
    fp = open("train_log.txt",'w')
    # initialize model
    G_net = DCGAN_G().to(device)
    GL_D_net = GlobalLocalDiscriminator().to(device)

    # summary(G_net, (4, 128, 128))
    # summary(D_net, (3, 128, 128))
    
    criterion = nn.BCELoss()
   

    optimizer_D = get_optimizer(GL_D_net)
    optimizer_G = get_optimizer(G_net)

    # prepare data
    celeba = CelebA_Pre(args.data_dir, 
                            args.bbox_dir,
                            args.ldmk_dir,
                            args.parts_dir)
    train_ids = celeba.get_parts('train')
    eval_ids = celeba.get_parts('eval')
    test_ids = celeba.get_parts('test')

    legi_fold, train_folds, eval_folds = celeba.cross_val_folds(
                                    args.masked_dir, 
                                    train_ids, eval_ids, test_ids, num_folds=args.num_folds)
     # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    # cross validation
    # training process
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    since = time.time()
    writer = SummaryWriter('log')
    iter_count = 0

    for index in args.folds_list:
        train_fold = train_folds[index]
        eval_fold = eval_folds[index]
        # print("the shape of AAA: ", np.shape(train_fold))
        train_dataset = eval(args.dataset)(gt_root=args.data_dir, g_root=args.masked_dir, mask_root=args.mask_dir, files=train_fold, augmentation=True)
        eval_dataset = eval(args.dataset)(args.data_dir, args.masked_dir, args.mask_dir, eval_fold, augmentation=True)

        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        eval_loader  = data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
        

        for epoch in range(args.epochs):
            for batch_idx, batch in enumerate(train_loader):
                g_in_batch, gt_data_batch, mask_01 = batch
                # the input of local discriminator
                real_local_disc_input = np.multiply(gt_data_batch, mask_01)
                real_local_disc_input = real_local_disc_input.to(device, dtype=torch.float).permute(0,3,1,2)
                # input for generator
                g_in_batch = g_in_batch.to(device, dtype=torch.float).permute(0, 3, 1, 2)           # [batch, 4, H, W]
                # ground truth
                gt_data_batch = gt_data_batch.to(device, dtype=torch.float).permute(0, 3, 1, 2)             # [batch, 3, H, W]
                # binary masks
                mask_01 = mask_01.to(device, dtype=torch.float).permute(0, 3, 1, 2)
                # (g_in_batch, gt_data_batch, b_mask_batch) 
                # print("the shape of g_in_batch: ", batch[0])
                # g_in_batch = Variable(batch[0]).cuda()
                # gt_data_batch = Variable(batch[1]).cuda()
                
                GL_D_net.zero_grad()

                label = torch.full((args.batch_size,), real_label, device = device)
                # forward pass real batch through D
                output = GL_D_net(gt_data_batch, real_local_disc_input).view(-1)

                # calculate loss on all-real batch
                errD_real = criterion(output, label)
                # calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## train with all-fake batch
                # generate batch of latent batch with G
                fake = G_net(g_in_batch)
                label.fill_(fake_label)
                # input for local disc
                fake_local_disc = torch.mul(fake, mask_01)
                # classify all fake batch with D
                output = GL_D_net(fake, fake_local_disc).view(-1)
                # calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # calculate the gradients for this batch
                errD_fake.backward(retain_graph=True)
                D_G_masked1 = output.mean().item()
                # add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # update D
                optimizer_D.step()


                ##############################
                # 2. update G network: maximize log(D(G(Z)))
                ##############################
                G_net.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # since we just updated D, perform another forward pass of all-fake batch through D
                output = GL_D_net(fake, fake_local_disc).view(-1)
                # calculate G's loss based on this output
                errG = criterion(output, label)
                # calculate gradients for G
                errG.backward(retain_graph=True)
                D_G_masked2 = output.mean().item()
                # update G
                optimizer_G.step()

                # output training stats
                if batch_idx%246 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, args.epochs, batch_idx, len(train_loader),
                            errD.item(), errG.item(), D_x, D_G_masked1, D_G_masked2))
                    fp.write('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\n'
                        % (epoch, args.epochs, batch_idx, len(train_loader),
                            errD.item(), errG.item(), D_x, D_G_masked1, D_G_masked2))
                    fp.flush()
                
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # validation: check how the generator is doing by saving G's output on maksed images
                if ((batch_idx % 20000) == 0) or ((epoch == args.epochs-1) and (batch_idx == len(train_loader)-1)):
                    with torch.no_grad():
                        for btch, e_data in enumerate(eval_loader):
                            if btch > 0:
                                break
                            e_gin, e_gt, e_mask_01 = e_data
                            e_gin = e_gin.to(device, dtype=torch.float).permute(0, 3, 1, 2)
                            e_gt = e_gt.to(device, dtype=torch.float).permute(0, 3, 1, 2)
                            e_mask_01 = e_mask_01.to(device, dtype=torch.float).permute(0, 3, 1, 2)
                            # eval_gd_data, eval_masked_data = load_batch(batch_idx, False)
                            fake = G_net(e_gin).detach().cpu()
                            masked_face = e_gin[:,:3,:,:]
                            
                            plot_img_fake = torchvision.utils.make_grid(fake.detach().cpu()[0:8])
                            plot_img_gd = torchvision.utils.make_grid(e_gt.detach().cpu()[0:8])
                            plot_img_masked = torchvision.utils.make_grid(masked_face.detach().cpu()[0:8])

                            imshow(plot_img_masked, plot_img_fake, plot_img_gd, btch, epoch)
                            # print("----------")
                            # io.imsave(SAVE_PATH+'DC_GAN_results/DC_GAN_V4/'+'plot_img_gd.png', (plot_img_gd))
                            # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
    fp.close()
                            

   


