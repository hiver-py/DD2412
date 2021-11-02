import torch 
import torch.nn as nn
import torch.cuda
import torchvision.models as models
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.datasets as dset
from residualblocks import Block, conv3x3






class Wide_ResNet(nn.Module):
    def __init__(self, dropout_rate=0, depth=28, widen_factor=2, num_classes=10):
        super().__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(Block, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(Block, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(Block, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, get_feat=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        
        if get_feat:
            return out
        else:
            return self.linear(out)