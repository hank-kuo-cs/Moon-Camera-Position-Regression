from torch.nn import Module, L1Loss


class TripletLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, s_features, p_features, n_features, margin):
        dist_p = L1Loss()(s_features, p_features)
        dist_n = L1Loss()(s_features, n_features)

        triplet_loss = margin + dist_p - dist_n

        return triplet_loss if triplet_loss > 0 else triplet_loss * 0
