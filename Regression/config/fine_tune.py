class FineTuneConfig:
    def __init__(self,
                 dist_optimizer_lr: float,
                 dist_w_decay: float,
                 angle_optimizer_lr: float,
                 angle_w_decay: float,
                 low_loss_bound: float,
                 epoch_num: int):
        self.dist_optimizer_lr = dist_optimizer_lr
        self.dist_w_decay = dist_w_decay
        self.angle_optimizer_lr = angle_optimizer_lr
        self.angle_w_decay = angle_w_decay
        self.low_loss_bound = low_loss_bound
        self.epoch_num = epoch_num

    def check_parameters(self):
        assert isinstance(self.dist_optimizer_lr, float)
        assert isinstance(self.dist_w_decay, float)
        assert isinstance(self.angle_optimizer_lr, float)
        assert isinstance(self.angle_w_decay, float)
        assert isinstance(self.low_loss_bound, float)
        assert isinstance(self.epoch_num, int)
