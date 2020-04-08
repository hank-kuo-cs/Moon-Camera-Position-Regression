import os
import re
import torch
import logging
from glob import glob
from torch.utils.data import DataLoader

from .data import MoonDataset
from .loss import MoonLoss
from .network import ValidateNetwork, VGG19, ResNet18, ResNet34, ResNet50, DenseNet121, DenseNet161
from .tensorboard import TensorboardWriter
from .config import config


class Validating:
    def __init__(self, data_loader):
        self._epoch = 0

        self.network = None
        self.model = None
        self.data_loader = data_loader

    def validate(self):
        for i, model_path in enumerate(self.models_path):
            self._epoch = self.get_epoch_num(model_path)
            self.set_model(model_path)
            self.set_network()

            self.network.run_one_epoch()
        logging.info('Finish validating')

    def set_network(self):
        self.network = ValidateNetwork(data_loader=self.data_loader,
                                       model=self.model,
                                       loss_func=self.loss_func,
                                       tensorboard_writer=self.tensorboard_writer,
                                       epoch=self._epoch)

    def set_model(self, model_path):
        image_size = self.data_loader.dataset[0][0].size()[1]
        models = {'VGG19': VGG19,
                  'ResNet18': ResNet18, 'ResNet34': ResNet34, 'ResNet50': ResNet50,
                  'DenseNet121': DenseNet121, 'DenseNet161': DenseNet161}
        self.model = models[config.network.network_model](image_size=image_size)

        self.set_model_gpu()
        self.set_model_device()
        self.set_model_pretrain(model_path)

    def set_model_gpu(self):
        if config.cuda.is_parallel:
            gpu_ids = config.cuda.parallel_gpus
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)

    def set_model_device(self):
        self.model = self.model.to(config.cuda.device)

    def set_model_pretrain(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        logging.info('Use %s to validate' % model_path)

    @property
    def model_num(self):
        return len(self.models_path)

    @property
    def models_path(self):
        return sorted(glob('%s/model*' % self.checkpoint_path))

    @staticmethod
    def get_epoch_num(model_path: str):
        assert isinstance(model_path, str)

        epoch_num_str = re.findall(r'epoch(.+?)\.pth', model_path)
        if epoch_num_str:
            return int(epoch_num_str[0])
        raise ValueError('Cannot find epoch number in the model path: %s' % model_path)

    @property
    def checkpoint_path(self):
        file_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(file_path)
        return os.path.join(dir_path, 'checkpoint')

    @property
    def loss_func(self):
        return MoonLoss()

    @property
    def tensorboard_writer(self):
        return TensorboardWriter(dataset_type='validation')


def validate():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    config.print_config()

    validation_dataset = MoonDataset('validation')
    validation_data_loader = DataLoader(dataset=validation_dataset,
                                        batch_size=config.network.batch_size,
                                        shuffle=True,
                                        num_workers=4)

    validating = Validating(data_loader=validation_data_loader)
    validating.validate()
