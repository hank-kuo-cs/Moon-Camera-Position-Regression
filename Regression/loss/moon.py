import torch
from torch.nn import MSELoss, L1Loss
from loss.spherical_transform import transform_spherical_angle_label, get_spherical_angle_constant_loss


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
            transform_spherical_angle_label(predicts, labels)
            constant_loss = get_spherical_angle_constant_loss(predicts)

        mse_loss = MSELoss()(predicts, labels)
        loss = torch.add(mse_loss, constant_loss)

        return loss
