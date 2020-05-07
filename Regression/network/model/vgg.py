import numpy as np
import torch.nn as nn
from torchvision.models import vgg19_bn
from .normalize_layer import NormalizeLayer
from ...config import config


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self._features = None
        self._model = vgg19_bn(pretrained=True)
        self._model.avgpool = self._make_avg_pool()
        self._model.regression = self._make_regression()

    def forward(self, x):
        out = self._model.features(x)
        out = self._model.avgpool(out)
        out = out.view(out.size(0), -1)
        self._features = out.clone()
        out = self._model.regression(out)

        return out

    @staticmethod
    def _make_regression():
        return nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.Dropout(p=0.5),
            nn.Linear(64, len(config.dataset.labels)),
            NormalizeLayer()
        )

    @staticmethod
    def _make_avg_pool():
        return nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(2, 2)))

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            raise ValueError('Features of VGG19 model is empty!')

        features = self._features
        if features.requires_grad:
            features = features.detach()

        if config.cuda.device != 'cpu':
            features = features.cpu()

        return features.numpy()
