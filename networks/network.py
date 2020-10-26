from .resnet import *
import torch.nn as nn
from .globalNet import globalNet
from .refineNet import refineNet
from .coordinateNet import CoordinateNet
from .heatmapNet import HeatmapNet
from .combinationNet import CombinationNet
from .coord2heatmapNet import Coord2HeatmapNet
__all__ = ['COORDINATENET', 'HEATMAPNET', 'COMBINATIONNET']


class COORDINATE(nn.Module):
    """Coordinate Regression Model"""
    def __init__(self, resnet, cfg):
        super(COORDINATE, self).__init__()
        self.resnet = resnet
        self.coordinate_net = CoordinateNet(cfg.num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        coordinate_out = self.coordinate_net(res_out)
        global_outs = []    # dummy
        attention_out = []
        heatmap_out = []
        combination_out = []
        return coordinate_out, global_outs, attention_out, heatmap_out, combination_out


class HEATMAP(nn.Module):
    """Hetamp Regression Model"""
    def __int__(self, cfg):
        super(HEATMAP, self).__init__()
        self.heatmap_net = HeatmapNet(init_chan_num=cfg.init_chan_num, neck_size=cfg.neck_size,
                                      growth_rate=cfg.growth_rate, class_num=cfg.class_num, layer_num=cfg.layer_num,
                                      order=cfg.order, loss_num=cfg.loss_num)

    def forward(self, x):
        heatmap_out = self.heatmap_net(x)
        coordinate_out = []         # dummy
        global_outs = []
        attention_out = []
        combination_out = []
        return coordinate_out, global_outs, attention_out, heatmap_out, combination_out


class COMBINATION(nn.Module):
    """ACN model"""
    def __init__(self, resnet, cfg):
        super(COMBINATION, self).__init__()
        self.resnet = resnet                                                                           # backbone
        self.coordinate_net = CoordinateNet(cfg.num_class)                                             # coordinate
        self.coord2heatmap_net = Coord2HeatmapNet(cfg.output_shape, cfg.num_class, sigma=cfg.gk0[0])   # c2h
        self.global_net = globalNet(cfg.channel_settings, cfg.output_shape, cfg.num_class)             # attention mask1
        self.refine_net = refineNet(cfg.channel_settings[-1], cfg.output_shape, 1)                     # attention mask2
        self.heatmap_net = HeatmapNet(init_chan_num=cfg.init_chan_num, neck_size=cfg.neck_size,        # heatmap
                                      growth_rate=cfg.growth_rate, class_num=cfg.class_num, layer_num=cfg.layer_num,
                                      order=cfg.order, loss_num=cfg.loss_num)
        self.combination_net = CombinationNet(cfg.output_shape, cfg.num_class)                         # combination

    def forward(self, x):
        res_out = self.resnet(x)                                                                 # backbone
        coordinate_out = self.coordinate_net(res_out)                                            # coordinate regression
        c2h_out = self.coord2heatmap_net(coordinate_out)                                         # coordinate-to-heatmap
        global_fms, global_outs = self.global_net(res_out)                                       # occlusion global
        attention_out = self.refine_net(global_fms)                                              # occlusion local
        heatmap_out = self.heatmap_net(x)                                                        # heatmap regression
        combination_out = self.combination_net(c2h_out, heatmap_out[-1], attention_out, res_out)  # combination
        return coordinate_out, global_outs, attention_out, heatmap_out, combination_out


def COORDINATENET(cfg, pretrained=True):
    """Create coordinate regression model"""
    res50 = resnet50(pretrained=pretrained)
    model = COORDINATE(res50, cfg)
    return model


def HEATMAPNET(cfg, pretrained=True):
    """Create heatmp regression model"""
    model = HEATMAP(cfg)
    return model


def COMBINATIONNET(cfg, pretrained=True):
    """Create ACN model"""
    res50 = resnet50(pretrained=pretrained)
    model = COMBINATION(res50, cfg)
    return model
