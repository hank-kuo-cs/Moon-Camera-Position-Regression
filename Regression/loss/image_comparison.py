import torch
import numpy as np
from copy import deepcopy
from torch.nn import L1Loss
from ..generate.main import Pytorch3DRenderer
from ..generate.loader import load_view
from ..config import config
from ..generate.config import MOON_MAX_RADIUS_IN_GL_UNIT, KM_TO_GL_UNIT


def get_image_comparison_loss(renderer: Pytorch3DRenderer, predicts, labels):
    batch_size = config.network.batch_size
    arr = []
    labels = labels.clone()

    predict_cameras, label_cameras = transform_regress_variables_to_cameras(predicts, labels)

    for i in range(batch_size):
        renderer.set_cameras(predict_cameras[i])
        predict_img = renderer.render_image()

        renderer.set_cameras(label_cameras[i])
        label_img = renderer.render_image()

        l1_loss = L1Loss()(predict_img, label_img)
        l1_loss_normalized_by_batch = torch.div(l1_loss, batch_size)
        arr.append(l1_loss_normalized_by_batch)

    img_compare_loss = torch.sum(torch.stack(arr))

    return img_compare_loss


def transform_regress_variables_to_cameras(predicts, labels):
    label_num = len(config.dataset.labels)
    # distance
    labels[:, 0] *= config.dataset.dist_range
    labels[:, 0] *= KM_TO_GL_UNIT
    labels[:, 0] += MOON_MAX_RADIUS_IN_GL_UNIT
    predicts[:, 0] *= config.dataset.dist_range
    predicts[:, 0] *= KM_TO_GL_UNIT
    predicts[:, 0] += MOON_MAX_RADIUS_IN_GL_UNIT

    # c theta & phi
    labels[:, 1:3] *= np.pi * 2
    if label_num > 1:
        predicts[:, 1:3] *= np.pi * 2

    # p & u
    labels[:, 3:] *= config.dataset.normalize_point_weight
    if label_num > 3:
        predicts[:, 3:] *= config.dataset.normalize_point_weight

    predict_cameras, label_cameras = [], []

    for i in range(config.network.batch_size):
        predict_cameras.append(load_view())
        label_cameras.append(load_view())

        # eye
        if label_num == 1:
            predict_cameras[i].eye = transform_spherical_to_cartesian([predicts[i, 0], labels[i, 1], labels[i, 2]])
        else:
            predict_cameras[i].eye = transform_spherical_to_cartesian(predicts[i, :3])
        label_cameras[i].eye = transform_spherical_to_cartesian(labels[i, :3])

        # at
        label_cameras[i].at = labels[i, 3: 6]
        predict_cameras[i].at = deepcopy(label_cameras[i].at) if label_num <= 3 else predicts[3: 6]

        # up
        label_cameras[i].up = labels[i, 6: 9]
        predict_cameras[i].up = deepcopy(label_cameras[i].up) if label_num <= 6 else predicts[6: 9]

    return predict_cameras, label_cameras


def transform_spherical_to_cartesian(spherical_point):
    cartesian_point = [0.0, 0.0, 0.0]

    cartesian_point[0] = spherical_point[0] * torch.sin(spherical_point[1]) * torch.cos(spherical_point[2])
    cartesian_point[1] = spherical_point[0] * torch.sin(spherical_point[1]) * torch.sin(spherical_point[2])
    cartesian_point[2] = spherical_point[0] * torch.cos(spherical_point[1])

    return cartesian_point
