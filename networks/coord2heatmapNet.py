"""Coordinate-to-heatmap Network in ACN"""
import torch.nn as nn
from utils.misc import *


class Coord2HeatmapNet(nn.Module):
    """Convert coordinates vector to heatmap"""
    def __init__(self, out_shape, num_class, sigma):
        super(Coord2HeatmapNet, self).__init__()
        self.out_shape = out_shape
        self.num_class = num_class
        self.fc_out = num_class * 2
        self._upsample = nn.Upsample(size=out_shape)
        self.sigma = sigma / out_shape[1]

        # denominator for gaussian filter
        self.denom = 2.0 * self.sigma ** 2

        # pointing layer
        self._pointing_layer = nn.Conv2d(self.fc_out, self.num_class, bias=False, kernel_size=1, groups=self.num_class)
        self._pointing_layer.weight.data.fill_(1)

        # generate meshgrid
        x = np.linspace(0.5/out_shape[0], (out_shape[0] - 0.5)/out_shape[0], out_shape[0], dtype=np.float32)
        y = np.linspace(0.5/out_shape[1], (out_shape[1] - 0.5)/out_shape[1], out_shape[1], dtype=np.float32)
        xg_np, yg_np = np.meshgrid(x, y)
        xy_repeat = np.zeros((self.fc_out, out_shape[0], out_shape[1]), np.float32)
        for i in range(self.num_class):
            xy_repeat[i*2, :, :] = xg_np
            xy_repeat[i*2+1, :, :] = yg_np
        xyg = to_torch(xy_repeat).cuda(non_blocking=True)
        self.register_buffer('grid', xyg)

    def forward(self, x):
        x = x.view(-1, self.fc_out, 1, 1)       # n * (num_class*2) * 1 * 1 layer
        x = self._upsample(x)                   # n * (num_class*2) * h * w layer
        for i in range(x.size()[0]):            # x-x0, y-y0
            x[i] = self.grid - x[i]
        x = torch.pow(x, 2) / self.denom        # (x-x0)^2 / 2 *sigma^2, (y-y0)^2 / 2 *sigma^2
        x = self._pointing_layer(x)             # (n * (num_class*2) * h * w layer) -> (n * num_class * h * w layer)
        x = torch.exp(-x) * 10.0                # exp(-(((x-x0)^2 / (2*sigma^2)) + ((y-y0)^2 / (2*sigma^2)))
        return x
