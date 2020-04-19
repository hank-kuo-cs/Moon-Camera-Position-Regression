import numpy as np
import torch.nn as nn
from torchvision.models import vgg19_bn
from ...config import config


vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self._features = None
        self._model = vgg19_bn(pretrained=True)
        self._model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self._model.avgpool = self._make_avg_pool()
        self._model.classifier = self._make_regression()
        # self.image_size = image_size
        # self.network = self._make_network()
        # self.avg_pool =
        # self.regression = self._make_regression()

    def forward(self, x):
        # out = self.network(x)
        out = self._model.features(x)
        out = self._model.avgpool(out)
        out = out.view(out.size(0), -1)
        self._features = out.clone()
        out = self._model.classifier(out)

        return out

    def _make_network(self):
        layers = []
        in_channels = 1

        for layer in vgg_cfg:
            if layer == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                self.image_size //= 2
            else:
                layers += [nn.Conv2d(in_channels, layer, kernel_size=3, padding=1),
                           nn.BatchNorm2d(layer),
                           nn.ReLU(inplace=True)]
                in_channels = layer

        return nn.Sequential(*layers)

    @staticmethod
    def _make_regression():
        return nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.Dropout(p=0.5),
            nn.Linear(64, len(config.dataset.labels))
        )

    @staticmethod
    def _make_avg_pool():
        return nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(2, 2)))

    @property
    def features(self) -> np.ndarray:
        if self._features is None:
            raise ValueError('Features of VGG19 model is empty!')

        features = self._features
        if features.requires_frad:
            features = features.detach()

        if config.cuda.device != 'cpu':
            features = features.cpu()

        return features.numpy()
