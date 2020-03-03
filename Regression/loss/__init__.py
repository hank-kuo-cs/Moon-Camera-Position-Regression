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
        type_check = isinstance(predicts, torch.FloatTensor) and isinstance(labels, torch.FloatTensor)
        type_check_gpu = isinstance(predicts, torch.cuda.FloatTensor) and isinstance(labels, torch.cuda.FloatTensor)
        assert type_check or type_check_gpu
        assert predicts.shape == labels.shape

        constant_loss = torch.tensor(0, dtype=torch.float)

        if labels.shape[1] >= 3:
            self.transform_spherical_angle_label(predicts, labels)
            constant_loss = self.get_spherical_angle_constant_loss(predicts)

        mse_loss = MSELoss()(predicts, labels)
        loss = torch.add(mse_loss, constant_loss)

        return mse_loss, constant_loss

    @staticmethod
    def transform_spherical_angle_label(predicts, labels):
        tmp = torch.zeros((config.network.batch_size, 2), dtype=torch.float).to(config.cuda.device)

        predicts[:, 1: 3] = torch.remainder(predicts[:, 1: 3], 1)

        over_one_radius_indices = torch.abs(predicts[:, 1:3] - labels[:, 1:3]) > 0.5

        tmp[over_one_radius_indices & (labels[:, 1:3] < predicts[:, 1:3])] = 1
        tmp[over_one_radius_indices & (labels[:, 1:3] >= predicts[:, 1:3])] = -1

        labels[:, 1:3] += tmp

    @staticmethod
    def get_spherical_angle_constant_loss(predicts):
        constant_loss = torch.abs(predicts[:, 1:3] // 1)
        constant_loss = torch.sum(constant_loss) / config.network.batch_size

        return constant_loss
