import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


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



class Block(nn.Module):
    def __init__(self, inplanes, planes, wide = True, stride = 1, drop_out = 0, downsample = None):
        super(Block, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.wide = wide
        self.stride = stride
        self.drop_out = drop_out
        self.downsample = downsample
        self.expansion = 1
        
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




class Wide_ResNet(nn.Module):
    def __init__(self, dropout_rate=0, depth=28, widen_factor=2, num_classes=100):
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


