import torch
import torch.nn as nn
from .triplet_cnn import TripletAngleFeatureExtractor
from ...config import config


class CustomAngleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.angle_feature_extractor = self._set_angle_feature_extractor()
        self.l1_loss_func = nn.L1Loss()

    @staticmethod
    def _set_angle_feature_extractor():
        model_path = '/home/hank/angle_feature_extractor.pth'
        model = TripletAngleFeatureExtractor().to(config.cuda.device)
        model.load_state_dict(torch.load(model_path))

        return model

    def forward(self, predict_img, target_img):
        assert isinstance(self.angle_feature_extractor, TripletAngleFeatureExtractor)
        predict_features = self.angle_feature_extractor.get_features(predict_img)
        target_features = self.angle_feature_extractor.get_features(target_img)

        return self.l1_loss_func(predict_features, target_features)
