import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50
from .normalize_layer import NormalizeLayer
from ...config import config


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self._features = None
        self._model = resnet18(pretrained=True)
        self._model.regression = self._make_regression()

    def forward(self, x):
        output = self._model.conv1(x)
        output = self._model.bn1(output)
        output = self._model.relu(output)
        output = self._model.maxpool(output)

        output = self._model.layer1(output)
        output = self._model.layer2(output)
        output = self._model.layer3(output)
        output = self._model.layer4(output)
        output = self._model.avgpool(output)

        output = output.view(output.size(0), -1)
        self._features = output.clone()

        output = self._model.regression(output)

        return output

    @staticmethod
    def _make_regression():
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.Dropout(p=0.5),
            nn.Linear(64, len(config.dataset.labels)),
            NormalizeLayer()
        )

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


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self._features = None
        self._model = resnet34(pretrained=True)
        self._model.regression = self._make_regression()

    def forward(self, x):
        output = self._model.conv1(x)
        output = self._model.bn1(output)
        output = self._model.relu(output)
        output = self._model.maxpool(output)

        output = self._model.layer1(output)
        output = self._model.layer2(output)
        output = self._model.layer3(output)
        output = self._model.layer4(output)
        output = self._model.avgpool(output)

        output = output.view(output.size(0), -1)
        self._features = output.clone()

        output = self._model.regression(output)

        return output

    @staticmethod
    def _make_regression():
        return nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 64),
            nn.Dropout(p=0.5),
            nn.Linear(64, len(config.dataset.labels)),
            NormalizeLayer()
        )

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


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self._features = None
        self._model = resnet50(pretrained=True)
        self._model.regression = self._make_regression()

    def forward(self, x):
        output = self._model.conv1(x)
        output = self._model.bn1(output)
        output = self._model.relu(output)
        output = self._model.maxpool(output)

        output = self._model.layer1(output)
        output = self._model.layer2(output)
        output = self._model.layer3(output)
        output = self._model.layer4(output)
        output = self._model.avgpool(output)

        output = output.view(output.size(0), -1)
        self._features = output.clone()

        output = self._model.regression(output)

        return output

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
