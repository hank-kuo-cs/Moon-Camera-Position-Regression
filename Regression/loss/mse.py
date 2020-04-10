import torch
from torch.nn import MSELoss
from .spherical_transform import transform_spherical_angle_label, get_spherical_angle_constant_loss
from ..config import config
from ..generate.config import MOON_MAX_RADIUS_IN_GL_UNIT, KM_TO_GL_UNIT


def get_mse_loss(predicts, labels):
    label_num = len(config.dataset.labels)
    labels = labels.clone()[:, :label_num]
    constant_loss = torch.tensor(0, dtype=torch.float)

    if labels.shape[1] >= 3:
        transform_spherical_angle_label(predicts, labels)
        constant_loss = get_spherical_angle_constant_loss(predicts)

    predicts[:, 0] = torch.div(predicts[:, 0], MOON_MAX_RADIUS_IN_GL_UNIT + (config.dataset.dist_range * KM_TO_GL_UNIT))
    labels[:, 0] = torch.div(labels[:, 0], MOON_MAX_RADIUS_IN_GL_UNIT + (config.dataset.dist_range * KM_TO_GL_UNIT))

    predicts[:, 3:] = torch.div(predicts[:, 3:], config.dataset.normalize_point_weight * KM_TO_GL_UNIT)
    labels[:, 3:] = torch.div(labels[:, 3:], config.dataset.normalize_point_weight * KM_TO_GL_UNIT)

    mse_loss = MSELoss()(predicts, labels)
    loss = torch.add(mse_loss, constant_loss)

    return loss
