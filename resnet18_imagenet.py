import math
import torch
import torch.nn as nn
from orn.modules import ORConv2d
from BinActivateFunc import BinActivateFunc, BinActivateFunc2, BinActivateFunc_bireal

def conv3x3(in_planes, out_planes, stride=1, M=1, method='MConv', binary=False):
    """3x3 convolution with padding"""
    return ORConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1, bias=False, arf_config=4)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, M=1, method='MConv', binary=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes * M)
        self.conv1 = conv3x3(inplanes, planes, stride, M=M, method=method, binary=binary)
        self.prelu = nn.PReLU(planes * M)
        self.downsample = downsample
        self.binary = BinActivateFunc_bireal.apply

    def forward(self, x):
        residual = x

        out = self.conv1(x)       
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.prelu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, M=1, method='MConv'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64*M, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64 * M)
        self.prelu = nn.PReLU(64 * M)  # TODO: Need or not?
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], M=M, method=method)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, M=M, method=method)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, M=M, method=method)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, M=M, method=method)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion * M, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1, M=1, method='MConv'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            shortcut_conv = nn.Conv2d(self.inplanes * M, planes * block.expansion * M, kernel_size=1, \
                                 stride=1, bias=False)
            downsample = nn.Sequential(
                shortcut_conv,
                nn.BatchNorm2d(planes * block.expansion * M),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, M=M, method=method, binary=True))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, M=M, method=method, binary=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, M=1, method='MConv'):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    assert method in ['MConv', 'GConv', 'UMConv']
    model = ResNet(BasicBlock, [4, 4, 4, 4], M=M, method=method)
    if pretrained:
        raise ValueError
    return model
