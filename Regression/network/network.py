import os
import torch


class Network:
    def __init__(self, config, model, loss_func, optimizer, tensorboard_writer, data_loader, is_train: bool = True, epoch: int = 1):
        self._config = config
        self._model = model
        self._loss_func = loss_func
        self._optimizer = optimizer
        self._tensorboard_writer = tensorboard_writer
        self._data_loader = data_loader
        self.is_train = is_train
        self.epoch = epoch

        self._features = None

    def run_one_epoch(self):
        for idx, (inputs, labels) in enumerate(self.get_data()):
            if self.is_train:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss_func(outputs, labels)

            if self.is_train:
                loss.backward()
                self.optimizer.step()

            self.tensorboard.add_loss(loss.item())

        self.epoch += 1

    def save_model(self):
        os.makedirs('checkpoint/', exist_ok=True)
        model_path = 'checkpoint/model_epoch%.3d.pth' % self.epoch
        torch.save(self.model.state_dict(), model_path)

    def get_data(self):
        for i, data in enumerate(self._data_loader):
            device = self.config.cuda.device
            inputs, labels = data[0].to(device), data[1].to(device)
            yield (inputs, labels)

    @property
    def config(self):
        if self._config is None:
            raise ValueError('Config of network is empty!')
        return self._config

    @property
    def model(self):
        if self._model is None:
            raise ValueError('Model of network is empty!')
        return self._model

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise ValueError('Optimizer of network is empty!')
        return self._optimizer

    @property
    def loss_func(self):
        if self._loss_func is None:
            raise ValueError('Loss function of network is empty!')
        return self._loss_func

    @property
    def data_loader(self):
        if self.data_loader is None:
            raise ValueError('Data loader of network is empty!')
        return self.data_loader

    @property
    def tensorboard(self):
        if self._tensorboard_writer is None:
            raise ValueError('Tensorboard of network is empty!')
        return self._tensorboard_writer
