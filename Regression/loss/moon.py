import torch
from torch.nn import L1Loss, MSELoss
from ..config import config
from .differential_renderer import DifferentialRenderer
from .mse import get_mse_loss


class MoonLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.renderer = DifferentialRenderer()

    def forward(self, predicts: torch.FloatTensor, labels: torch.FloatTensor) -> torch.Tensor:
        type_check = isinstance(predicts, torch.FloatTensor) and isinstance(labels, torch.FloatTensor)
        type_check_gpu = isinstance(predicts, torch.cuda.FloatTensor) and isinstance(labels, torch.cuda.FloatTensor)
        assert type_check or type_check_gpu

        img_compare_loss = self.get_img_compare_loss(predicts, labels)
        mse_loss = get_mse_loss(predicts, labels)

        l_mse = config.network.l_mse
        l_img = config.network.l_image_comparison

        return l_img * img_compare_loss + l_mse * mse_loss

    def get_img_compare_loss(self, predicts, labels):
        batch_size = config.network.batch_size
        tmp_loss_arr = []

        for i in range(batch_size):
            predict_img = self.renderer.render_image(predicts[i])
            gt_img = self.renderer.render_image(labels[i])

            loss = MSELoss()(predict_img, gt_img)
            loss = torch.div(loss, batch_size)
            tmp_loss_arr.append(loss)

        return torch.sum(torch.stack(tmp_loss_arr))
