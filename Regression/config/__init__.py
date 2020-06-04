import os
import json
from .cuda import CudaConfig
from .dataset import DatasetConfig
from .generate import GenerateConfig
from .network import NetworkConfig
from .tensorboard import TensorboardConfig
from .fine_tune import FineTuneConfig


# If you want to use cpu or parallel gpus, please comment below code.
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Config:
    def __init__(self):
        self.cuda = CudaConfig(device='cuda',
                               is_parallel=False,
                               cuda_device_number=3,
                               parallel_gpus=[0, 2, 3])

        self.dataset = DatasetConfig(dataset_path='/data/space/pytorch3d/Dataset_15km_rgb_20w',
                                     labels=['dist', 'elev', 'azim'],
                                     dataset_size={'train': 160000, 'test': 20000, 'validation': 20000},
                                     sub_dataset_size=20000,
                                     normalize_point_weight=0.5)

        self.generate = GenerateConfig(moon_obj_path='/data/space/pytorch3d/moon_data/Moon_8K.obj',
                                       image_size=400,
                                       fov=120,
                                       znear=0.0001,
                                       zfar=1000.0,
                                       moon_radius_gl=1.7459008620440053,
                                       gl_to_km=1000.0,
                                       dist_between_moon_low_bound_km=1.0,
                                       dist_between_moon_high_bound_km=15.0,
                                       is_change_eye=True,
                                       is_change_at=True,
                                       is_change_up=True)

        self.network = NetworkConfig(network_model='VGG19',
                                     batch_size=10,
                                     epoch_num=300,
                                     learning_rate=0.001,
                                     momentum=0.9,
                                     l_mse=1.0,
                                     l_image_comparison=1.0,
                                     l_mse_dist=1.0,
                                     l_mse_elev=2.0,
                                     l_mse_azim=4.0,
                                     l_mse_p=1.0,
                                     l_mse_u=1.0)

        self.tensorboard = TensorboardConfig(tensorboard_path='/home/hank/Tensorboard',
                                             experiment_name='E5_20w_VGG19Pre_lambda_b10_lr1e3_sgd',
                                             loss_step=200,
                                             tsne_epoch_step=50,
                                             is_write_loss=True,
                                             is_write_tsne=False)

        self.fine_tune = FineTuneConfig(dist_optimizer_lr=0.001,
                                        dist_w_decay=0.001,
                                        angle_optimizer_lr=0.005,
                                        angle_w_decay=0.005,
                                        low_loss_bound=0.01,
                                        epoch_num=5)

    def export_config_to_tensorboard_dir(self):
        file_out_path = os.path.join(self.tensorboard.tensorboard_path, self.tensorboard.experiment_name + '_config.txt')

        with open(file_out_path, 'w') as f:
            data = {'cuda': self.cuda.__dict__,
                    'dataset': self.dataset.__dict__,
                    'network': self.network.__dict__,
                    'visualize': self.tensorboard.__dict__}
            f.write(json.dumps(data))

    def print_config(self):
        print('Config Setting:')

        data = {'cuda': self.cuda.__dict__,
                'dataset': self.dataset.__dict__,
                'network': self.network.__dict__,
                'visualize': self.tensorboard.__dict__}

        for k1, v1 in data.items():
            print(k1)
            for k2, v2 in v1.items():
                print('\t{0}: {1}'.format(k2, v2))
            print()


config = Config()
