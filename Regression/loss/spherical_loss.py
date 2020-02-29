import torch


class SphericalLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.predicts = None

    def forward(self, predicts: torch.FloatTensor, labels: torch.FloatTensor) -> torch.FloatTensor:
        pass
