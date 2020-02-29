import torch
import logging
import argparse
from glob import glob
from torch.utils.data import DataLoader

from data import MoonDataset
from loss import MoonLoss
from network import Network, Resnet18, VGG19
from tensorboard import TensorboardWriter
from config import config


class Trainer:
    def __init__(self, args, data_loader):
        self.args = args
        self.network = None
        self.model = None
        self.data_loader = data_loader

    def run(self):
        self.set_model()
        self.set_network()

        for i in range(config.network.epoch_num):
            self.network.run_one_epoch()

    def set_network(self):
        self.network = Network(config=config,
                               model=self.model,
                               loss_func=self.loss_func,
                               optimizer=self.optimizer,
                               tensorboard_writer=self.tensorboard_writer,
                               data_loader=self.data_loader)

    def set_model(self):
        models = {'VGG19': VGG19, 'Resnet18': Resnet18}
        self.model = models[config.network.network_model]

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
        if self.args.pretrain_model and self.args.scratch:
            raise ValueError('Cannot use both argument \'pretrain_model\' and \'scratch\'!')

        model_path = self.get_pretrain_model_path()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))

    def get_pretrain_model_path(self):
        model_path = self.args.pretrain_model

        if not model_path:
            pretrain_model_paths = glob('./checkpoint/model*')
            model_path = sorted(pretrain_model_paths)[-1] if pretrain_model_paths else None

        return model_path

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
        return TensorboardWriter()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    config.print_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('-pm', '--pretrain_model', type=str, help='Use a pretrain model in checkpoint directory to continue training')
    parser.add_argument('-s', '--scratch', action='store_true', help='Train model from scratch, do not use pretrain model')

    arguments = parser.parse_args()

    train_dataset = MoonDataset().train_dataset
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=config.network.batch_size,
                                   shuffle=True,
                                   num_workers=4)

    trainer = Trainer(args=arguments, data_loader=train_data_loader)
    trainer.run()
