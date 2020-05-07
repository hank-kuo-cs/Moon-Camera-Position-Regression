import torch
from torch.nn import MSELoss
from ..config import config


def get_mse_loss(predicts, gts):
    if 'azim' in config.dataset.labels:
        azim_index = config.dataset.labels.index('azim')
        gts[:, azim_index] = adjust_azim_labels_to_use_scmse(predicts[:, azim_index], gts[:, azim_index])

    labels_num = config.dataset.labels_num
    l_mse_dist, l_mse_elev, l_mse_azim = config.network.l_mse_dist, config.network.l_mse_elev, config.network.l_mse_azim
    l_mse_p, l_mse_u = config.network.l_mse_p, config.network.l_mse_u

    dist_loss, elev_loss, azim_loss, p_loss, u_loss = 0.0, 0.0, 0.0, 0.0, 0.0

    dist_loss = MSELoss()(predicts[:, 0], gts[:, 0]) * l_mse_dist

    if labels_num > 1:
        elev_loss = MSELoss()(predicts[:, 1], gts[:, 1]) * l_mse_elev
        azim_loss = MSELoss()(predicts[:, 2], gts[:, 2]) * l_mse_azim

    if labels_num > 3:
        p_loss = MSELoss()(predicts[:, 3:6], gts[:, 3:6]) * l_mse_p

    if labels_num > 6:
        u_loss = MSELoss()(predicts[:, 6:9], gts[:, 6:9]) * l_mse_u

    mse_loss = (dist_loss + elev_loss + azim_loss + p_loss + u_loss) / labels_num

    return mse_loss


def adjust_azim_labels_to_use_scmse(azim_predicts, azim_gts):
    condition1 = torch.abs(azim_predicts - azim_gts) > 0.5
    condition2 = azim_predicts > azim_gts
    condition3 = azim_predicts < azim_gts

    azim_gts = torch.where(condition1 & condition2, azim_gts + 1, azim_gts)
    azim_gts = torch.where(condition1 & condition3, azim_gts - 1, azim_gts)

    return azim_gts
