import torch
from torch.nn import MSELoss
from ..config import config


def get_mse_loss(predicts, gts):
    if 'azim' in config.dataset.labels:
        azim_index = config.dataset.labels.index('azim')
        gts[:, azim_index] = adjust_azim_labels_to_use_scmse(predicts[:, azim_index], gts[:, azim_index])

    return MSELoss()(predicts, gts)


def adjust_azim_labels_to_use_scmse(azim_predicts, azim_gts):
    condition1 = torch.abs(azim_predicts - azim_gts) > 0.5
    condition2 = azim_predicts > azim_gts
    condition3 = azim_predicts < azim_gts

    azim_gts = torch.where(condition1 & condition2, azim_gts + 1, azim_gts)
    azim_gts = torch.where(condition1 & condition3, azim_gts - 1, azim_gts)

    return azim_gts
