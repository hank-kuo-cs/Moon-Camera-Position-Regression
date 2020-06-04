import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from config import EXPERIMENT_NAME


class TensorboardWriter:
    def __init__(self):
        writer_path = '/home/hank/Tensorboard/%s' % EXPERIMENT_NAME
        self.writer = SummaryWriter(writer_path)

    def add_scalar(self, tag: str, x: int, y: float):
        self.writer.add_scalar(tag=tag, scalar_value=y, global_step=x)
        self.writer.flush()

    def add_embedding(self, tag: str, features: np.ndarray, labels: np.ndarray):
        assert features.shape[0] == labels.shape[0]
        assert features.ndim == 2

        self.writer.add_embedding(mat=features, metadata=labels, tag=tag)
        self.writer.flush()

    def close(self):
        self.writer.close()