import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn import L1Loss, MSELoss
from torch.optim import Adam
from ...loss.fine_tune import SSIM
from ...config import config


class RendererModel(nn.Module):
    def __init__(self, renderer, target_image, init_dist, init_elev, init_azim):
        super().__init__()
        self.renderer = renderer
        self.img_size = config.generate.image_size
        self.device = config.cuda.device
        self.target_image = target_image

        self.dist_parameter = DistParameter(init_dist)
        self.angle_parameters = AngleParameters(init_elev, init_azim)

        self.dist_optimizer = Adam(self.dist_parameter.parameters(), lr=0.0001, weight_decay=0.0001)
        self.angles_optimizer = Adam(self.angle_parameters.parameters(), lr=0.001, weight_decay=0.001)

    def forward(self):
        self.reset_optimizer()

        self.renderer.set_cameras(dist=self.dist_parameter.dist,
                                  elev=self.angle_parameters.angles[0],
                                  azim=self.angle_parameters.angles[1],
                                  at=(0, 0, 0),
                                  up=(0, 1, 0))
        predict_image = self.renderer.render_image()
        predict_image = self.refine_predict_image(predict_image)

        loss = self.get_loss(predict_image)
        loss.backward()

        self.update_optimizer()

        return loss.item()

    def get_loss(self, predict_image):
        # loss = L1Loss()(predict_image, self.target_image)
        # loss = MSELoss()(predict_image, self.target_image)
        loss = 1 - SSIM()(self.target_image, predict_image)

        return loss

    def reset_optimizer(self):
        self.dist_optimizer.zero_grad()
        self.angles_optimizer.zero_grad()

    def update_optimizer(self):
        self.dist_optimizer.step()
        self.angles_optimizer.step()

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
        return self.angle_parameters.angles[0].item()

    @property
    def azim(self):
        return self.angle_parameters.angles[1].item()


class DistParameter(nn.Module):
    def __init__(self, init_dist):
        super().__init__()
        dist_tensor = torch.tensor(init_dist, dtype=torch.float).to(config.cuda.device)
        self.dist = nn.Parameter(dist_tensor, requires_grad=True)


class AngleParameters(nn.Module):
    def __init__(self, init_elev, init_azim):
        super().__init__()
        angles_tensor = torch.tensor([init_elev, init_azim], dtype=torch.float).to(config.cuda.device)
        self.angles = nn.Parameter(angles_tensor, requires_grad=True)
