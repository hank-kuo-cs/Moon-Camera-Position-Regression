import os
import re
import torch
import logging
import argparse
from glob import glob
from torch.utils.data import DataLoader

from .data import MoonDataset
from .loss import MoonLoss
from .network import TrainNetwork, VGG19, ResNet18, ResNet34, ResNet50, DenseNet121, DenseNet161
from .tensorboard import TensorboardWriter
from .config import config


class Training:
    def __init__(self, args, data_loader):
        self._is_scratch = False
        self._epoch_of_pretrain = ''
        self._epoch = 1
        self.set_arguments(args)

        self.network = None
        self.model = None
        self.data_loader = data_loader

    def train(self):
        self.set_model()
        self.set_network()

        for i in range(self._epoch - 1, config.network.epoch_num):
            logging.info('Start training epoch %d' % (i+1))
            self.network.run_one_epoch()

    def set_arguments(self, args):
        self._is_scratch = args.scratch
        self._epoch_of_pretrain = args.epoch_of_pretrain

    def set_network(self):
        self.network = TrainNetwork(model=self.model,
                                    data_loader=self.data_loader,
                                    loss_func=self.loss_func,
                                    optimizer=self.optimizer,
                                    tensorboard_writer=self.tensorboard_writer,
                                    epoch=self._epoch)

    def set_model(self):
        image_size = self.data_loader.dataset[0][0].size()[1]
        models = {'VGG19': VGG19,
                  'ResNet18': ResNet18, 'ResNet34': ResNet34, 'ResNet50': ResNet50,
                  'DenseNet121': DenseNet121, 'DenseNet161': DenseNet161}
        self.model = models[config.network.network_model](image_size=image_size)

        self.set_model_gpu()
        self.set_model_device()
        self.set_model_pretrain()

    def set_model_gpu(self):
        if config.cuda.is_parallel:
            gpu_ids = config.cuda.parallel_gpus
            self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)

    def set_model_device(self):
        self.model = self.model.to(config.cuda.device)

    def set_model_pretrain(self):
        if self._epoch_of_pretrain and self._is_scratch:
            raise ValueError('Cannot use both argument \'pretrain_model\' and \'scratch\'!')

        model_path = self.get_pretrain_model_path()
        if model_path and not self._is_scratch:
            self._epoch = self.get_epoch_num(model_path) + 1
            self.model.load_state_dict(torch.load(model_path))
            logging.info('Use pretrained model %s to continue training' % model_path)
        else:
            logging.info('Train from scratch')

    def get_pretrain_model_path(self):
        epoch_of_pretrain = self._epoch_of_pretrain

        if not epoch_of_pretrain:
            pretrain_model_paths = glob('%s/model*' % self.checkpoint_path)
            model_path = sorted(pretrain_model_paths)[-1] if pretrain_model_paths else None
        else:
            model_path = '%s/model_epoch%.3d.pth' % (self.checkpoint_path, int(epoch_of_pretrain))

        return model_path

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
    def optimizer(self):
        return torch.optim.SGD(params=self.model.parameters(),
                               lr=config.network.learning_rate,
                               momentum=config.network.momentum)

    @property
    def tensorboard_writer(self):
        return TensorboardWriter(dataset_type='train')


def train():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    config.print_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch_of_pretrain', type=str,
                        help='Use a pretrain model in checkpoint directory to continue training')
    parser.add_argument('-s', '--scratch', action='store_true',
                        help='Train model from scratch, do not use pretrain model')

    arguments = parser.parse_args()

    train_dataset = MoonDataset('train')
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=config.network.batch_size,
                                   shuffle=True,
                                   num_workers=4)

    trainer = Training(args=arguments, data_loader=train_data_loader)
    trainer.train()
