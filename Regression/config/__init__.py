import os
import json
from .cuda import CudaConfig
from .dataset import DatasetConfig
from .network import NetworkConfig
from .tensorboard import TensorboardConfig


# You have to comment below device setting code if you want to use parallel gpu.
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


class Config:
    def __init__(self):
        self.cuda = CudaConfig(device='cuda',
                               is_parallel=False,
                               cuda_device_number=0,
                               parallel_gpus=[0, 2, 3])

        self.dataset = DatasetConfig(dataset_path='',
                                     labels=['dist', 'c_theta', 'c_phi'],
                                     dataset_size={'train': 80000, 'test': 10000, 'validation': 10000},
                                     sub_dataset_size=10000,
                                     dist_range=15.0,   # km
                                     normalize_point_weight=2000.0)

        self.network = NetworkConfig(network_model='ResNet18',
                                     batch_size=10,
                                     epoch_num=300,
                                     learning_rate=0.001,
                                     momentum=0.9,
                                     l_mse=1.0,
                                     l_image_comparison=1.0)

        self.tensorboard = TensorboardConfig(tensorboard_path='',
                                             experiment_name='',
                                             loss_step=100,
                                             tsne_epoch_step=50,
                                             is_write_loss=True,
                                             is_write_tsne=False)

    def export_config_to_tensorboard_dir(self):
        file_out_path = os.path.join(self.tensorboard.tensorboard_path, self.tensorboard.experiment_name + '_config.txt')

        with open(file_out_path, 'w') as f:
            data = {'cuda': self.cuda.__dict__,
                    'dataset': self.dataset.__dict__,
                    'network': self.network.__dict__,
                    'tensorboard': self.tensorboard.__dict__}
            f.write(json.dumps(data))

    def print_config(self):
        print('Config Setting:')

        data = {'cuda': self.cuda.__dict__,
                'dataset': self.dataset.__dict__,
                'network': self.network.__dict__,
                'tensorboard': self.tensorboard.__dict__}

        for k1, v1 in data.items():
            print(k1)
            for k2, v2 in v1.items():
                print('\t{0}: {1}'.format(k2, v2))
            print()


config = Config()
