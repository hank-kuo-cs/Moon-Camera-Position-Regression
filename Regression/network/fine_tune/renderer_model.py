import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.nn import L1Loss, MSELoss
from ...loss import SSIM
from ..triplet_angle_feature_extractor import CustomAngleLoss
from torch.optim import Adam
from ...config import config


class RendererModel(nn.Module):
    def __init__(self, renderer, target_image, init_dist, init_elev, init_azim):
        super().__init__()
        self.renderer = renderer
        self.img_size = config.generate.image_size
        self.device = config.cuda.device
        self.target_image = target_image
        self.l1_loss_func = L1Loss()
        self.mse_loss_func = MSELoss()
        self.ssim_loss_func = SSIM()
        self.custom_angle_loss_func = CustomAngleLoss()

        self.dist_parameter = DistParameter(init_dist)
        self.elev_parameter = ElevParameter(init_elev)
        self.azim_parameter = AzimParameter(init_azim)

        self.dist_optimizer = Adam(self.dist_parameter.parameters(), lr=config.fine_tune.dist_optimizer_lr)
        self.elev_optimizer = Adam(self.elev_parameter.parameters(), lr=config.fine_tune.elev_optimizer_lr)
        self.azim_optimizer = Adam(self.azim_parameter.parameters(), lr=config.fine_tune.azim_optimizer_lr)

    def forward(self):
        self.reset_optimizer()
        dist, elev, azim = self.get_camera_positions()

        self.renderer.set_cameras(dist=dist,
                                  elev=elev,
                                  azim=azim,
                                  at=(0, 0, 0),
                                  up=(0, 1, 0))
        predict_image = self.renderer.render_image()
        predict_image = self.refine_predict_image(predict_image)

        loss = self.get_loss(predict_image)
        loss.backward()

        self.update_optimizer()

        return loss.item()

    def get_camera_positions(self):
        dist_range = [config.generate.dist_low_gl, config.generate.dist_high_gl]
        elev_range = [-np.pi / 2, np.pi / 2 - 0.001]  # add -0.001 to prevent camera direction be same as up vector
        azim_range = [0, np.pi * 2]

        dist = self.dist_parameter.dist
        elev = self.elev_parameter.elev
        azim = self.azim_parameter.azim

        dist = torch.clamp(dist, dist_range[0], dist_range[1])

        # elev = torch.clamp(elev, elev_range[0], elev_range[1])
        # azim = torch.clamp(azim, azim_range[0], azim_range[1])

        return dist, elev, azim

    def get_loss(self, predict_image):
        # loss = L1Loss()(predict_image, self.target_image)
        # loss = MSELoss()(predict_image, self.target_image)
        # loss = 1 - SSIM()(self.target_image, predict_image)
        loss = self.custom_angle_loss_func(predict_image, self.target_image)

        return loss

    def reset_optimizer(self):
        self.dist_optimizer.zero_grad()
        self.elev_optimizer.zero_grad()
        self.azim_optimizer.zero_grad()

    def update_optimizer(self):
        self.dist_optimizer.step()
        self.elev_optimizer.step()
        self.azim_optimizer.step()

    def refine_predict_image(self, predict_image):
        predict_image = predict_image[..., :3]
        predict_image = predict_image.reshape(3, self.img_size, self.img_size)
        predict_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(predict_image)

        return predict_image[None, ...]

    @property
    def dist(self):
        return self.dist_parameter.dist.item()

    @property
    def elev(self):
        return self.elev_parameter.elev.item()

    @property
    def azim(self):
        return self.azim_parameter.azim.item()


class DistParameter(nn.Module):
    def __init__(self, init_dist):
        super().__init__()
        dist_tensor = torch.tensor(init_dist, dtype=torch.float).to(config.cuda.device)
        self.dist = nn.Parameter(dist_tensor, requires_grad=True)


class ElevParameter(nn.Module):
    def __init__(self, init_elev):
        super().__init__()
        elev_tensor = torch.tensor(init_elev, dtype=torch.float).to(config.cuda.device)
        self.elev = nn.Parameter(elev_tensor, requires_grad=True)


class AzimParameter(nn.Module):
    def __init__(self, init_azim):
        super().__init__()
        azim_tensor = torch.tensor(init_azim, dtype=torch.float).to(config.cuda.device)
        self.azim = nn.Parameter(azim_tensor, requires_grad=True)
