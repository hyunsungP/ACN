"""Combination Network for ACN"""
import torch.nn as nn
from utils.misc import *


class BottleneckCombine(nn.Module):
    """Combination Network"""
    def __init__(self, inplanes, middleplanes, outplanes, stride=1):
        super(BottleneckCombine, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, middleplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(middleplanes)
        self.conv2 = nn.Conv2d(middleplanes, middleplanes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(middleplanes)
        self.conv3 = nn.Conv2d(middleplanes, outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

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


class CombinationNet(nn.Module):
    """Combine Two heatmaps and low-level features"""
    def __init__(self, out_size, num_class):
        super(CombinationNet, self).__init__()
        self.num_class = num_class
        self.combine = BottleneckCombine(num_class * 2 + 256 + 512, num_class * 2, num_class)
        self.upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=True)

    def forward(self, heatmap_global, heatmap_local, mask, feature):
        mask_repeat = mask.repeat(1, self.num_class, 1, 1)
        heatmap_masked = heatmap_local * mask_repeat                                       # element-wise multiplication
        feature2 = self.upsample(feature[2])                               # fit second low-level feature to output w, h
        heatmap = torch.cat([heatmap_global, heatmap_masked, feature[3], feature2], dim=1)    # concat heatmaps,features
        return self.combine(heatmap)                                                          # combination network
