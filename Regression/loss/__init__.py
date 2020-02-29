import torch
from loss.spherical_loss import SphericalLoss
from torch.nn import MSELoss, L1Loss


class MoonLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.predicts = None

    def forward(self, predicts: torch.FloatTensor, labels: torch.FloatTensor) -> torch.FloatTensor:
        pass

    def calculate_angle_loss(self):
        pass

    def calculate_distance_loss(self):
        pass
