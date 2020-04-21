import torch
from .mse import get_mse_loss
from ..config import config


class MoonLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicts: torch.FloatTensor, labels: torch.FloatTensor) -> torch.Tensor:
        type_check = isinstance(predicts, torch.FloatTensor) and isinstance(labels, torch.FloatTensor)
        type_check_gpu = isinstance(predicts, torch.cuda.FloatTensor) and isinstance(labels, torch.cuda.FloatTensor)
        assert type_check or type_check_gpu

        ground_truths = labels[:, :config.dataset.labels_num]
        assert predicts.size() == ground_truths.size()

        mse_loss = get_mse_loss(predicts, ground_truths.clone())

        return mse_loss
