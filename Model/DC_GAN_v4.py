import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from Tools.layers import Flatten,Concatenate
import torchvision
from torchvision import  models

from tensorboardX import SummaryWriter

import numpy as np
import time
import os
import copy
import pickle
from skimage import color

import matplotlib.pyplot as plt
import csv
import pandas as pd
import skimage.io as io
import random 
from PIL import Image, ImageChops

from IPython.display import HTML

import Dataset.datasets as celebA

from torchsummary import summary
plt.ion()   # interactive mode

# weight initialiation
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

# GAN architecture
class DCGAN_D(nn.Module):
    def __init__(self):
        super(DCGAN_D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64,  kernel_size=4,  stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # in: (?, 64, 64, 64)
            nn.Conv2d(64, 128,  kernel_size=4,  stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # in: (?, 128, 32, 32)
            nn.Conv2d(128, 256,  kernel_size=4,  stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout2d(0.3),
            # in: (?, 256, 16, 16)
            nn.Conv2d(256, 512,  kernel_size=4,  stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            # nn.Dropout2d(0.3),

            
            # in: (?, 512, 8, 8)
            nn.Conv2d(512, 512,  kernel_size=5,  stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout2d(0.3),

            # in : (?, 512, 4, 4)
            nn.Flatten(),
            nn.Linear(512*4*4, 1024),
            nn.LeakyReLU(0.2,inplace=True),
        )
        # nn.Conv2d(1024, 1,  kernel_size=4,  stride=1, padding=0),

            # nn.Sigmoid(),

        
        
      
        initialize_weights(self)

        
    def forward(self, x):
        x = self.conv(x)
    
        # x = x.view(x.shape[0], -1)
        return x

class GlobalLocalDiscriminator(nn.Module):
    def __init__(self, input_dim=(64, 3, 128, 128)):
        super(GlobalLocalDiscriminator, self).__init__()

        self.output_shape = (1,)
        # self.input_shape = [input_dim, input_dim]

        self.global_discriminator = DCGAN_D()
        self.local_discriminator = DCGAN_D()

        # in_features = self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
        # self.concat1 = Concatenate(dim=-1)
        # input_shape: (None, 2048)
        # self.linear1 = nn.Linear(in_features, 1)
        # self.act1 = nn.Sigmoid()

        self.linear = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_g, x_l):
        x_global = self.global_discriminator(x_g)
        x_local = self.local_discriminator(x_l)
        ca = torch.cat([x_global, x_local], -1)
        logit = self.sigmoid(self.linear(F.leaky_relu(ca)))
        return logit

class DCGAN_G(nn.Module):
    def __init__(self):
        super(DCGAN_G, self).__init__()

        # input shape: (None, 4, 128, 128)

        self.conv1 = nn.Conv2d(4, 64,  kernel_size=4,  stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.dp1 = nn.Dropout2d(0.3)
        # input shape: (None, 64, 64, 64)
        self.conv2 = nn.Conv2d(64, 128,  kernel_size=3,  stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        self.dp2 = nn.Dropout2d(0.3)
        # input shape: (None, 128, 64, 64)
        self.conv3 = nn.Conv2d(128, 128,  kernel_size=4,  stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = nn.ReLU()
        self.dp3 = nn.Dropout2d(0.3)
        # input shape: (None, 256, 32, 32)
        self.conv4 = nn.Conv2d(128, 256,  kernel_size=3,  stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = nn.ReLU()
        self.dp4 = nn.Dropout2d(0.3)
        # input shape: (None, 256, 32, 32)
        self.conv5 = nn.Conv2d(256, 256,  kernel_size=3,  stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.act5 = nn.ReLU()
        self.dp5 = nn.Dropout2d(0.3)
        # # input shape: (None, 256, 32, 32)
        self.conv6 = nn.Conv2d(256, 256,  kernel_size=3,  stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.act6 = nn.ReLU()
        self.dp6 = nn.Dropout2d(0.3)
        # input shape: (None, 256, 32, 32)
        self.conv7 = nn.Conv2d(256, 256,  kernel_size=3,  stride=1, dilation=2, padding=2)
        self.bn7 = nn.BatchNorm2d(256)
        self.act7 = nn.ReLU()
        # input shape: (None, 256, 32, 32)
        self.conv8 = nn.Conv2d(256, 256,  kernel_size=3,  stride=1, dilation=4, padding=4)
        self.bn8 = nn.BatchNorm2d(256)
        self.act8 = nn.ReLU()
        # input shape: (None, 256, 32, 32)
        self.conv9 = nn.Conv2d(256, 256,  kernel_size=3,  stride=1, dilation=8, padding=8)
        self.bn9 = nn.BatchNorm2d(256)
        self.act9 = nn.ReLU()
        # input shape: (None, 256, 32, 32)
        self.conv10 = nn.Conv2d(256, 256,  kernel_size=3,  stride=1, dilation=16, padding=16)
        self.bn10 = nn.BatchNorm2d(256)
        self.act10 = nn.ReLU()
        # input shape: (None, 256, 32, 32)
        self.conv11 = nn.Conv2d(256, 256,  kernel_size=3,  stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(256)
        self.act11 = nn.ReLU()
        # # input shape: (None, 256, 32, 32)
        self.conv12 = nn.Conv2d(256, 256,  kernel_size=3,  stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(256)
        self.act12 = nn.ReLU()
        # input shape: (None, 256, 32, 32)
        self.deconv13 = nn.ConvTranspose2d(256, 128,  kernel_size=4,  stride=2, padding=1)
        self.bn13 = nn.BatchNorm2d(128)
        self.act13 = nn.ReLU()

        # input shape: (None, 128, 64, 64)
        self.conv14 = nn.Conv2d(128, 128,  kernel_size=3,  stride=1, padding=1)
        self.bn14 = nn.BatchNorm2d(128)
        self.act14 = nn.ReLU()

        self.deconv15 = nn.ConvTranspose2d(128, 64,  kernel_size=4,  stride=2, padding=1)
        self.bn15 = nn.BatchNorm2d(64)
        self.act15 = nn.ReLU()

        self.conv16 = nn.Conv2d(64, 32,  kernel_size=3,  stride=1, padding=1)
        self.bn16 = nn.BatchNorm2d(32)
        self.act16 = nn.ReLU()

        self.conv17 = nn.Conv2d(32, 3,  kernel_size=3,  stride=1, padding=1)
        self.act17 = nn.Tanh()

        initialize_weights(self)
        
    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))
        x = self.bn4(self.act4(self.conv4(x)))
        x = self.bn5(self.act5(self.conv5(x)))
        x = self.bn6(self.act6(self.conv6(x)))
        x = self.bn7(self.act7(self.conv7(x)))
        x = self.bn8(self.act8(self.conv8(x)))
        x = self.bn9(self.act9(self.conv9(x)))
        x = self.bn10(self.act10(self.conv10(x)))
        x = self.bn11(self.act11(self.conv11(x)))
        x = self.bn12(self.act12(self.conv12(x)))
        x = self.bn13(self.act13(self.deconv13(x)))
        x = self.bn14(self.act14(self.conv14(x)))
        x = self.bn15(self.act15(self.deconv15(x)))
        x = self.bn16(self.act16(self.conv16(x)))
        x = self.act17(self.conv17(x))

        return x
