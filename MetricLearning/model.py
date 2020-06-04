import torch.nn as nn
from torchvision.models import resnet18


class TripletAngleFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = resnet18(pretrained=True)
        self._model.avgpool = self._make_avg_pool()
        self._model.linear = self._make_linear()

    def forward(self, samples, positives, negatives):
        samples_features = self.get_features(samples)
        positives_features = self.get_features(positives)
        negatives_features = self.get_features(negatives)

        return samples_features, positives_features, negatives_features

    def get_features(self, inputs):
        features = self._model.conv1(inputs)
        features = self._model.bn1(features)
        features = self._model.relu(features)
        features = self._model.maxpool(features)
        features = self._model.layer1(features)
        features = self._model.layer2(features)
        features = self._model.layer3(features)
        features = self._model.layer4(features)
        features = self._model.avgpool(features)
        features = features.view(features.size(0), -1)
        features = self._model.linear(features)

        return features

    @staticmethod
    def _make_linear():
        return nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(p=0.5),
            nn.Linear(256, 64),
        )

    @staticmethod
    def _make_avg_pool():
        return nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
