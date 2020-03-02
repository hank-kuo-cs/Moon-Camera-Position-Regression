import numpy as np
from tensorboard.loss import LossWriter
from config import config


class TensorboardWriter:
    def __init__(self, dataset_type: str, step: int = 1, epoch: int = 1):
        self._dataset_type = dataset_type
        self._step = step
        self._epoch = epoch
        self._losses = []
        self.loss_writer = LossWriter(dataset_type=dataset_type)
        self._dataset_num = config.dataset.get_dataset_num(dataset_type)

        self.check_parameters()

    def add_loss(self, step_loss: float):
        self._losses.append(step_loss)

    def write_step_loss(self):
        step_loss = self._losses[-1]
        self.loss_writer.set_parameters(step=self._step, epoch=self._epoch, loss=step_loss)
        self.loss_writer.write_loss_by_step()

        self._step += 1

    def write_epoch_loss(self):
        losses = np.array(self._losses)
        epoch_loss = np.average(losses)
        self.loss_writer.set_parameters(loss=epoch_loss)
        self.loss_writer.write_loss_by_epoch()

        self.reset_loss()
        self._epoch += 1

    def reset_loss(self):
        self._losses.clear()
        self._step = 1

    def add_tsne(self, features):
        pass

    def check_parameters(self):
        assert isinstance(self._dataset_type, str)
        assert isinstance(self._step, int)
        assert isinstance(self._epoch, int)

        assert self._dataset_type in config.dataset.dataset_types
