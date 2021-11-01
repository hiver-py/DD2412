import torch 
import torch.nn as nn
import torch.cuda
import torchvision.models as models
import math
import numpy as np
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.datasets as dset





def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain = np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)



def conv3x3(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, stride, kernel_size=3,padding=1, bias=False)



class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, wide, stride = 1, drop_out = 0, downsample = None):
        super(ResNetBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        # Wide is a True/False depending on if its wide resnet or regular.
        self.wide = wide
        self.stride = stride
        self.drop_out = drop_out
        self.downsample = downsample

        if self.wide:
            self.bn1 = nn.BatchNorm2d(inplanes)
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=True)
            if self.dropout_rate > 0:
                self.dropout = nn.Dropout(p=self.drop_out)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

            self.shortcut = nn.Sequential()
            if stride != 1 or inplanes != planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=True))

        else:            
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 =  nn.BatchNorm2d(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 =  nn.BatchNorm2d(planes)
            self.downsample = downsample
            self.stride = stride

        
    def forward(self, x):

        if self.wide:
            out = self.conv1(F.relu(self.bn1(x)))

            if self.dropout_rate>0:
                out = self.dropout(out)

            out = self.conv2(F.relu(self.bn2(out)))
            out += self.shortcut(x)

        else:

            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)
                out += identity
                out = self.relu(out)

        return out


