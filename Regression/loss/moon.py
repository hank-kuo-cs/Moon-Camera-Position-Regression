import torch
from .mse import get_mse_loss
from .image_comparison import get_image_comparison_loss
from ..generate.main import load_moon, Pytorch3DRenderer
from ..config import config


class MoonLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.renderer = None
        self.set_differential_renderer()

    def forward(self, predicts: torch.FloatTensor, labels: torch.FloatTensor) -> torch.Tensor:
        type_check = isinstance(predicts, torch.FloatTensor) and isinstance(labels, torch.FloatTensor)
        type_check_gpu = isinstance(predicts, torch.cuda.FloatTensor) and isinstance(labels, torch.cuda.FloatTensor)
        assert type_check or type_check_gpu

        mse_loss = get_mse_loss(predicts, labels)

        return mse_loss

    def set_differential_renderer(self):
        moon = load_moon()
        self.renderer = Pytorch3DRenderer(moon=moon)

        self.renderer.set_device()
        self.renderer.set_mesh()
        self.renderer.set_lights()
        self.renderer.set_raster_settings()
