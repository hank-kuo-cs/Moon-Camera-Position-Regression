import torch
import torch.nn as nn
from torchvision import transforms
from torch.nn import L1Loss, MSELoss
from ...loss.fine_tune import SSIM
from ...config import config


class CameraPositionOptimizer(nn.Module):
    def __init__(self, renderer, target_image, dist, elev, azim):
        super().__init__()
        self.renderer = renderer
        self.img_size = config.generate.image_size
        self.device = config.cuda.device

        self.register_buffer('target_image', target_image)

        camera_positions = torch.tensor([dist, elev, azim], dtype=torch.float).to(config.cuda.device)
        self.camera_position = nn.Parameter(camera_positions, requires_grad=True)

    def forward(self):
        self.renderer.set_cameras(dist=self.camera_position[0],
                                  elev=self.camera_position[1],
                                  azim=self.camera_position[2],
                                  at=(0, 0, 0),
                                  up=(0, 1, 0))
        predict_image = self.renderer.render_image()
        predict_image = self.refine_predict_image(predict_image)

        # loss = L1Loss()(predict_image, self.target_image)
        # loss = MSELoss()(predict_image, self.target_image)
        loss = 1 - SSIM()(self.target_image, predict_image)

        return loss

    def refine_predict_image(self, predict_image):
        predict_image = predict_image[..., :3]
        predict_image = predict_image.reshape(3, self.img_size, self.img_size)
        predict_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(predict_image)

        return predict_image[None, ...]

    @property
    def dist(self):
        return self.camera_position[0].item()

    @property
    def elev(self):
        return self.camera_position[1].item()

    @property
    def azim(self):
        return self.camera_position[2].item()
