import torch
from torch.nn import Module
from config import DEVICE


class TripletLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_features, p_features, n_features, margins):
        assert s_features.size() == p_features.size() == n_features.size()
        assert s_features.size(0) == margins.size(0)

        dist_p = torch.abs(s_features - p_features).mean(dim=1)
        dist_n = torch.abs(s_features - n_features).mean(dim=1)

        margins = margins.view(-1)

        assert margins.size() == dist_p.size() == dist_n.size()

        triplet_losses = margins + dist_p - dist_n
        compare_zeros = torch.zeros(triplet_losses.size(), dtype=torch.float, device=DEVICE)

        triplet_losses = torch.max(triplet_losses, compare_zeros)
        loss = triplet_losses.mean()

        return loss
