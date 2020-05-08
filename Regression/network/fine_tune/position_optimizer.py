import torch
import torch.nn as nn
from torch.nn import L1Loss
from ...config import config


class CameraPositionOptimizer(nn.Module):
    def __init__(self, renderer, target_image, dist, elev, azim):
        super().__init__()
        self.renderer = renderer
        self.device = config.cuda.device

        target_image = target_image.reshape(1, 400, 400, 3)
        self.register_buffer('target_image', target_image)

        camera_positions = torch.tensor([dist, elev, azim], dtype=torch.float).to(config.cuda.device)
        self.camera_position = nn.Parameter(camera_positions, requires_grad=True)

    def forward(self):
        self.renderer.set_cameras(dist=self.camera_position[0],
                                  elev=self.camera_position[1],
                                  azim=self.camera_position[2],
                                  at=(0, 0, 0),
                                  up=(0, 1, 0))
        predict_image = self.renderer.render_image()[..., :3]

        loss = L1Loss()(predict_image, self.target_image)

        return loss

    @property
    def dist(self):
        return self.camera_position[0].item()

    @property
    def elev(self):
        return self.camera_position[1].item()

    @property
    def azim(self):
        return self.camera_position[2].item()
