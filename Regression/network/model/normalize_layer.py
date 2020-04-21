import torch.nn as nn
from ...config import config


class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()

    def forward(self, x):
        x[:, 0] = nn.Sigmoid()(x[:, 0])

        if 'elev' in config.dataset.labels:
            x[:, 1] = nn.Tanh()(x[:, 1])

        if 'azim' in config.dataset.labels:
            x[:, 2] = nn.Sigmoid()(x[:, 2])

        return x
