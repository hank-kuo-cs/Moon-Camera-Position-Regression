import os
from tensorboardX import SummaryWriter
from Regression.config import config


class LossWriter:
    def __init__(self, epoch: int = 0, step: int = 0, dataset_type: str = None, loss: float = 0.0):
        self.epoch = epoch
        self.step = step
        self.dataset_type = dataset_type
        self.loss = loss

    def set_parameters(self, **kwargs):
        if 'epoch' in kwargs:
            self.epoch = kwargs['epoch']
        if 'step' in kwargs:
            self.step = kwargs['step']
        if 'dataset_type' in kwargs:
            self.dataset_type = kwargs['dataset_type']
        if 'loss' in kwargs:
            self.loss = kwargs['loss']

    def write_loss_by_step(self):
        self.check_parameters()

        if self.step % config.tensorboard.loss_step != 0:
            return

        tag = '{0}/{1}/loss_by_step'.format(config.tensorboard.experiment_name, self.dataset_type)
        self.add_scalar_to_tensorboard(tag=tag, value=self.loss, global_step=self.step + self.steps_of_epochs)

    def write_loss_by_epoch(self):
        self.check_parameters()
        tag = '{0}/{1}/loss_by_epoch'.format(config.tensorboard.experiment_name, self.dataset_type)
        self.add_scalar_to_tensorboard(tag=tag, value=self.loss, global_step=self.epoch)

    def write_error_by_epoch(self, label_type):
        self.check_parameters()
        tag = '{0}/{1}/{2}_error'.format(config.tensorboard.experiment_name, self.dataset_type, label_type)
        self.add_scalar_to_tensorboard(tag=tag, value=self.loss, global_step=self.epoch)

    @staticmethod
    def add_scalar_to_tensorboard(tag: str, value: float, global_step: int):
        if not config.tensorboard.is_write_loss:
            return
        writer_path = os.path.join(config.tensorboard.tensorboard_path, config.tensorboard.experiment_name)
        writer = SummaryWriter(writer_path)
        writer.add_scalar(tag=tag, scalar_value=value, global_step=global_step)
        writer.close()

    @property
    def steps_of_epochs(self):
        return (self.epoch - 1) * config.dataset.get_dataset_num(self.dataset_type) // config.network.batch_size

    def check_parameters(self):
        assert isinstance(self.epoch, int)
        assert isinstance(self.step, int)
        assert isinstance(self.dataset_type, str)
        assert isinstance(self.loss, float)

        assert self.dataset_type in config.dataset.dataset_types

        assert self.epoch > 0
        assert 0 <= self.step <= self.step + self.steps_of_epochs
        assert self.loss > 0
