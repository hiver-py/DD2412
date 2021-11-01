import torch 
import torch.nn as nn
import torch.cuda
import torchvision.models as models
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.datasets as dset


class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, inplanes, planes, stride=1, downsample=None):

        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, planes, layers, blocks, num_classes=1000):
        self.block = block
        self.planes = planes
        self.layers = layers
        self.blocks = blocks
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.block, 64, layers[0])
        self.layer2 = self._make_layer(self.block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.fc = nn.Linear(512 * self.block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != self.planes * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes * self.block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.planes * self.block.expansion),
            )

        layers = []
        layers.append(self.block(self.inplanes, self.planes, stride, downsample))
        self.inplanes = self.planes * self.block.expansion
        for i in range(1, self.blocks):
            layers.append(self.block(self.inplanes, self.planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
        #model.load_state_dict(torch.load('networks/resnet50-19c8e357.pth'))
    return model