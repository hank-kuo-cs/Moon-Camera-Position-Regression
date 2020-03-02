import torch
from loss.spherical_loss import SphericalLoss
from torch.nn import MSELoss, L1Loss
from config import config


class MoonLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.predicts = None

    def forward(self, predicts: torch.FloatTensor, labels: torch.FloatTensor) -> torch.Tensor:
        assert isinstance(predicts, torch.FloatTensor) and isinstance(labels, torch.FloatTensor)
        assert predicts.shape == labels.shape

        constant_loss = torch.tensor(0, dtype=torch.float)

        if labels.shape[1] >= 3:
            self.transform_spherical_angle_label(predicts, labels)
            constant_loss = self.get_spherical_angle_constant_loss(predicts)

        mse_loss = MSELoss()(predicts, labels)

        return torch.add(mse_loss, constant_loss)

    @staticmethod
    def transform_spherical_angle_label(predicts, labels):
        tmp = torch.zeros((config.network.batch_size, 2), dtype=torch.float)

        over_one_radius_indices = torch.abs(predicts[:, 1:3] - labels[:, 1:3]) > 0.5

        tmp[over_one_radius_indices & (predicts[:, 1:3] < labels[:, 1:3])] = 1
        tmp[over_one_radius_indices & (predicts[:, 1:3] >= labels[:, 1:3])] = -1

        labels[:, 1:3] += tmp

    @staticmethod
    def get_spherical_angle_constant_loss(predicts):
        constant_loss = torch.abs(predicts[:, 1:3] // 1)
        constant_loss = torch.sum(constant_loss) / config.network.batch_size

        return constant_loss
