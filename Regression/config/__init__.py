import os
import json
from config.cuda import CudaConfig
from config.dataset import DatasetConfig
from config.network import NetworkConfig
from config.tensorboard import TensorboardConfig


class Config:
    def __init__(self):
        self.cuda = CudaConfig(device='cuda',
                               is_parallel=False,
                               cuda_device_number=2,
                               parallel_gpus=[0, 2, 3])

        self.dataset = DatasetConfig(dataset_path='/data/space/Dataset_fovy120_10k',
                                     labels=['dist'],
                                     dataset_size={'train': 80000, 'test': 10000, 'validation': 10000},
                                     sub_dataset_size=10000,
                                     dist_range=80.0,   # km
                                     normalize_point_weight=2000.0)

        self.network = NetworkConfig(network_model='VGG19',
                                     batch_size=10,
                                     epoch_num=300,
                                     learning_rate=0.001,
                                     momentum=0.9)

        self.tensorboard = TensorboardConfig(tensorboard_path='../Tensorboard',
                                             experiment_name='E1_old_dataset_80km',
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
