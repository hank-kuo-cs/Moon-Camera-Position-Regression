class FineTuneConfig:
    def __init__(self,
                 dist_optimizer_lr: float,
                 dist_w_decay: float,
                 elev_optimizer_lr: float,
                 azim_optimizer_lr: float,
                 low_loss_bound: float,
                 epoch_num: int):
        self.dist_optimizer_lr = dist_optimizer_lr
        self.dist_w_decay = dist_w_decay
        self.elev_optimizer_lr = elev_optimizer_lr
        self.azim_optimizer_lr = azim_optimizer_lr
        self.low_loss_bound = low_loss_bound
        self.epoch_num = epoch_num

    def check_parameters(self):
        assert isinstance(self.dist_optimizer_lr, float)
        assert isinstance(self.dist_w_decay, float)
        assert isinstance(self.elev_optimizer_lr, float)
        assert isinstance(self.low_loss_bound, float)
        assert isinstance(self.epoch_num, int)
