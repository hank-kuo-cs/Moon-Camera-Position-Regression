import torch.nn as nn
from config import config


vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']


class VGG19(nn.Module):
    def __init__(self, image_size: int):
        super(VGG19, self).__init__()
        self._features = None
        self._image_size = image_size
        self.network = self._make_network()
        self.regression1 = nn.Linear(512, 256)
        self.regression2 = nn.Linear(256, 128)
        self.regression3 = nn.Linear(128, 64)
        self.regression4 = nn.Linear(64, 16)
        self.regression5 = nn.Linear(16, len(config.dataset.labels))

    def forward(self, x):
        out = self.network(x)

        out = out.view(out.size(0), -1)

        self._features = out.clone()

        out = self.regression1(out)
        out = self.regression2(out)
        out = self.regression3(out)
        out = self.regression4(out)
        out = self.regression5(out)

        return out

    def _make_network(self):
        image_size = self._image_size
        layers = []
        in_channels = 1

        for layer in vgg_cfg:
            if layer == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                image_size //= 2
            else:
                layers += [nn.Conv2d(in_channels, layer, kernel_size=3, padding=1),
                           nn.BatchNorm2d(layer),
                           nn.ReLU(inplace=True)]
                in_channels = layer

        layers.append(nn.AvgPool2d(kernel_size=(image_size, image_size)))

        return nn.Sequential(*layers)

    @property
    def features(self):
        if self._features is None:
            raise ValueError('Features of VGG19 model is empty!')
        return self._features
