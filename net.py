from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from model import BasicUNet
from model.BasicUNet.model_BasicUNet import *
from utils import *
import os
from loss import *
from model import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()
        self.model_name = model_name
        
        self.cal_loss = SoftIoULoss()

        if model_name == 'DNANet':
            self.model = DNANet()  
        elif model_name == 'BasicUNet_plus':
            self.model = BasicUNet_plus()

    def weights_init_kaiming(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    def forward(self, img, f=None, epoch=None):
        if f is not None:
            return self.model(img, f)
        elif epoch is not None:
            return self.model(img, epoch)
        else:
            return self.model(img)

    def loss(self, pred, gt_mask, epoch=None):
        if epoch is not None:
            loss = self.cal_loss(pred, gt_mask, epoch)
        else:
            loss = self.cal_loss(pred, gt_mask)
        return loss
    # def con_minloss(self, pred, gt_mask):
    #     loss = 
