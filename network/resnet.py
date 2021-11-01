import torch 
import torch.nn as nn
import torch.cuda
import torchvision.models as models
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.datasets as dset
from residualblocks import ResNetBlock


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_paths = {
    'resnet18': 'networks/resnet18-5c106cde.pth',
    'resnet34': 'networks/resnet34-333f7ec4.pth',
    'resnet50': 'networks/resnet50-19c8e357.pth',
    'resnet101': 'networks/resnet101-5d3b4d8f.pth',
    'resnet152': 'networks/resnet152-b121ed2d.pth',
}


model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
model.eval()

   

'''
class WideResNet():
    def __init__(self):
        self.lr = 0.001
        self.momentum = 0.9
        self.wd = 5*10e-4
        self.batch_size = 128


'''