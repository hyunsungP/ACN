"""Heatmap-to-coordinate Network in ACN"""
import torch.nn as nn
from utils.misc import *
from .coord2heatmapNet import Coord2HeatmapNet


class Heatmap2coordNet(nn.Module):
    """Convert heatmap to coordinates vector"""
    def __init__(self, heatmap_shape, num_class):
        super(Heatmap2coordNet, self).__init__()
        self.out_shape = heatmap_shape
        self.n_channel_element = heatmap_shape[0] * heatmap_shape[1]
        self.num_class = num_class
        self._upsample = nn.Upsample(size=self.out_shape)
        self.coord2heatmap_net = Coord2HeatmapNet(heatmap_shape, num_class, 7.0)

        # generate meshgrid
        x = np.linspace(0.5 / heatmap_shape[0], (heatmap_shape[0] - 0.5) / heatmap_shape[0], heatmap_shape[0],
                        dtype=np.float32)
        y = np.linspace(0.5 / heatmap_shape[1], (heatmap_shape[1] - 0.5) / heatmap_shape[1], heatmap_shape[1],
                        dtype=np.float32)
        xg_np, yg_np = np.meshgrid(x, y)
        x_repeat = np.zeros((self.num_class, heatmap_shape[0], heatmap_shape[1]), np.float32)
        y_repeat = np.zeros((self.num_class, heatmap_shape[0], heatmap_shape[1]), np.float32)
        for i in range(self.num_class):
            x_repeat[i, :, :] = xg_np
            y_repeat[i, :, :] = yg_np
        xg = to_torch(x_repeat).cuda(async=True)
        yg = to_torch(y_repeat).cuda(async=True)
        self.register_buffer('grid_x', xg)
        self.register_buffer('grid_y', yg)

    def forward(self, input_heatmap):
        # --- make weights around max ---
        # get max coordinates
        m = input_heatmap.view(-1, self.num_class, self.n_channel_element)  # (batch, 68, 64*64)
        max_val, max_idx = torch.max(m, 2)                      # get max index
        max_idx = max_idx.to(dtype=torch.float32)
        max_idx_x = max_idx % self.out_shape[1]                 # get x index
        max_idx_x = (max_idx_x + 0.5) / self.out_shape[1]       # normalize to 0 ~ 1
        max_idx_y = max_idx // self.out_shape[0]                # get y index
        max_idx_y = (max_idx_y + 0.5) / self.out_shape[0]       # normalize to 0 ~ 1
        max_idx_x = max_idx_x.view(-1, self.num_class, 1)       # reshape to (batch, 68, 1)
        max_idx_y = max_idx_y.view(-1, self.num_class, 1)       # reshape to (batch, 68, 1)
        max_idx_xy = torch.cat([max_idx_x, max_idx_y], dim=2)   # concat to (batch, 68, 2) (x1,y1,x2,y2,...,x68,y68)
        max_idx_xy_re = max_idx_xy.view(-1, self.num_class*2)   # reshape to (batch, 136)
        # get weights
        weights_max_around = self.coord2heatmap_net(max_idx_xy_re)
        x_weighted = input_heatmap * weights_max_around

        # --- get final coordinates ---
        # get coordinates
        x_sum = x_weighted.sum(3).sum(2)                        # get sum of all elements in each channel (batch, 68)
        x_sum_re = x_sum.view(-1, self.num_class, 1, 1)         # reshape to (batch, 68, 1, 1)
        x_sum_upsampled = self._upsample(x_sum_re)              # upsample to (Batch, 68, 64, 64)
        hx = x_weighted / x_sum_upsampled * self.grid_x                  # normalize and multiplication
        hy = x_weighted / x_sum_upsampled * self.grid_y
        hx_sum = hx.sum(3).sum(2)                               # Sum(hx*x)
        hy_sum = hy.sum(3).sum(2)                               # Sum(hx*y)
        x_coord = hx_sum.view(-1, self.num_class, 1)
        y_coord = hy_sum.view(-1, self.num_class, 1)
        pts = torch.cat([x_coord, y_coord], dim=2)              # cat x and y
        pts_re = pts.view(-1, self.num_class*2)                 # reshape to 136 dimensional vector

        return pts_re
