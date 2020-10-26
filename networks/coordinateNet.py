import torch.nn as nn


class CoordinateNet(nn.Module):
    def __init__(self, num_class):
        super(CoordinateNet, self).__init__()
        fc_dim = (2048, 1024, 1024, num_class * 2)                            # dimensions of fully-connected layers
        self.n_fc = len(fc_dim) - 1
        self.fc_input_dim = fc_dim[0]
        fcs = []
        fcs.append(self._fcs(fc_dim))                                         # generate fully-connected layers
        self.fc = nn.ModuleList(fcs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                           # global average pooling from backbone

    def _fcs(self, dims):
        layers = []
        for i in range(self.n_fc):
            current_fc = nn.Linear(dims[i], dims[i + 1])
            if i < self.n_fc - 1:                               # no output layer
                # current_fc.weight.data.normal_(0, 0.01)                     # explicit weight initialization
                # current_fc.bias.data.normal_(0, 0.01)
                pass
            else:                                               # output layer
                current_fc.bias.data.normal_(0.5, 0.01)                       # explicit weight initialization
            layers.append(current_fc)
            if i < self.n_fc - 1:  # when not output layer
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x[0]
        out = self.avgpool(out)
        out = out.view(-1, self.fc_input_dim)
        out = self.fc[0](out)
        return out
