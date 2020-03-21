import numpy as np
from tensorboard.loss import LossWriter
from config import config


class TensorboardWriter:
    def __init__(self, dataset_type: str):
        self._dataset_type = dataset_type
        self.loss_writer = LossWriter(dataset_type=dataset_type)
        self._dataset_num = config.dataset.get_dataset_num(dataset_type)

        self.check_parameters()

    def write_avg_step_loss(self, step: int, epoch: int, avg_step_loss: float):
        self.loss_writer.set_parameters(step=step, epoch=epoch, loss=avg_step_loss)
        self.loss_writer.write_loss_by_step()

    def write_avg_epoch_loss(self, epoch: int, avg_epoch_loss: float):
        self.loss_writer.set_parameters(epoch=epoch, loss=avg_epoch_loss)
        self.loss_writer.write_loss_by_epoch()

    def write_avg_error(self, label_type: str, epoch: int, avg_error: float):
        self.loss_writer.set_parameters(epoch=epoch, loss=avg_error)
        self.loss_writer.write_error_by_epoch(label_type)

    def add_tsne(self, features):
        pass

    def check_parameters(self):
        assert isinstance(self._dataset_type, str)
        assert self._dataset_type in config.dataset.dataset_types
