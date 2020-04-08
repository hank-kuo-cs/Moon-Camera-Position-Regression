import torch
from torch.nn import MSELoss
from .spherical_transform import transform_spherical_angle_label, get_spherical_angle_constant_loss
from ..config import config


def get_mse_loss(predicts, labels):
    label_num = len(config.dataset.labels)
    labels = labels.clone()[:, :label_num]
    constant_loss = torch.tensor(0, dtype=torch.float)

    if labels.shape[1] >= 3:
        transform_spherical_angle_label(predicts, labels)
        constant_loss = get_spherical_angle_constant_loss(predicts)

    mse_loss = MSELoss()(predicts, labels)
    loss = torch.add(mse_loss, constant_loss)

    return loss
