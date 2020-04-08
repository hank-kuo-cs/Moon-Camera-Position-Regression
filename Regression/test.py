import re
import torch
import logging
import argparse
from glob import glob
from torch.utils.data import DataLoader

from .data import MoonDataset
from .loss import MoonLoss
from .network import TestNetwork, VGG19, ResNet18, ResNet34, ResNet50, DenseNet121, DenseNet161
from .config import config


class Testing:
    def __init__(self, args, data_loader):
        self._model_path = ''
        self._epoch = 0
        self.set_arguments(args)

        self.network = None
        self.model = None
        self.data_loader = data_loader

    def test(self):
        self.set_model()
        self.set_network()

        logging.info('Start testing')
        self.network.run_one_epoch()
        logging.info('Finish testing')

    def set_arguments(self, args):
        self._model_path = args.model

    def set_network(self):
        self.network = TestNetwork(data_loader=self.data_loader,
                                   model=self.model,
                                   loss_func=self.loss_func,
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
        model_path = self._model_path
        if not model_path:
            pretrain_model_paths = glob('./checkpoint/model*')
            model_path = sorted(pretrain_model_paths)[-1] if pretrain_model_paths else None

        if not model_path:
            raise ValueError('Cannot find model to test!')

        self._epoch = self.get_epoch_num(model_path)
        self.model.load_state_dict(torch.load(model_path))
        logging.info('Use %s to test' % model_path)

    @staticmethod
    def get_epoch_num(model_path: str):
        assert isinstance(model_path, str)

        epoch_num_str = re.findall(r'epoch(.+?)\.pth', model_path)
        if epoch_num_str:
            return int(epoch_num_str[0])
        raise ValueError('Cannot find epoch number in the model path: %s' % model_path)

    @property
    def loss_func(self):
        return MoonLoss()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    config.print_config()

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='Use a certain model in checkpoint directory to test')

    arguments = parser.parse_args()

    test_dataset = MoonDataset('test')
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=config.network.batch_size,
                                  shuffle=True,
                                  num_workers=4)

    testing = Testing(args=arguments, data_loader=test_data_loader)
    testing.test()
