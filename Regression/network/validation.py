import torch
import logging
import numpy as np
from tqdm import tqdm
from .network import Network
from ..config import config


class ValidateNetwork(Network):
    def __init__(self, model, loss_func, tensorboard_writer, data_loader, epoch: int = 1):
        super().__init__(model=model,
                         data_loader=data_loader,
                         loss_func=loss_func,
                         tensorboard_writer=tensorboard_writer,
                         epoch=epoch)
        self.predicts = None
        self.labels = None
        self.label_types = config.dataset.labels
        self.avg_epoch_loss = 0.0

    def run_one_epoch(self):
        self.model.eval()

        with torch.no_grad():
            for idx, (inputs, labels) in tqdm(enumerate(self.get_data())):
                predicts = self.model(inputs)

                predicts, labels = self.transform_spherical_angle_label(predicts, labels)

                self.add_predicts(predicts)
                self.add_labels(labels)

                loss = self.loss_func(predicts, labels)
                self.avg_epoch_loss += loss.item()

        self.write_epoch_loss()
        self.normalize_predicts_and_labels()
        self.write_avg_error()
        return self.avg_epoch_loss

    def write_epoch_loss(self):
        self.avg_epoch_loss /= (config.dataset.validation_dataset_num // config.network.batch_size)

        tag = 'validate/epoch_loss'
        self.tensorboard.add_scalar(tag=tag, x=self._epoch, y=self.avg_epoch_loss)

    def write_avg_error(self):
        self.write_distance_error()
        self.write_angle_error()
        self.write_point_error()

    def write_distance_error(self):
        dist_predicts = self.predicts[:, 0]
        dist_labels = self.labels[:, 0]

        dist_avg_km_error = np.average(np.abs((dist_predicts - dist_labels)))
        dist_avg_km_error = float(dist_avg_km_error)

        tag = 'validate/dist_km_error'

        self.tensorboard.add_scalar(tag=tag, x=self._epoch, y=dist_avg_km_error)

        print('Distance average error: ±%.3f km' % dist_avg_km_error)

    def write_angle_error(self):
        if len(self.label_types) < 3:
            return

        elev_avg_degree_error = np.average(np.abs(self.predicts[:, 1] - self.labels[:, 1]))
        elev_avg_degree_error = float(elev_avg_degree_error)

        tag = 'validate/elev_degree_error'
        self.tensorboard.add_scalar(tag=tag, x=self._epoch, y=elev_avg_degree_error)

        print('elev average error: ±%.3f degree' % elev_avg_degree_error)

        azim_avg_degree_error = np.average(np.abs(self.predicts[:, 2] - self.labels[:, 2]))
        azim_avg_degree_error = float(azim_avg_degree_error)

        tag = 'validate/azim_degree_error'
        self.tensorboard.add_scalar(tag=tag, x=self._epoch, y=azim_avg_degree_error)

        print('azim average error: ±%.3f degree' % azim_avg_degree_error)

    def write_point_error(self):
        if len(self.label_types) > 3:
            for i in range(3, len(self.label_types)):
                point_predicts = self.predicts[:, i]
                point_labels = self.labels[:, i]

                point_avg_km_error = np.average(np.abs(point_predicts - point_labels))
                point_avg_km_error = float(point_avg_km_error)

                tag = 'validate/%s_km_error' % self.label_types[i]
                self.tensorboard.add_scalar(tag=tag, x=self._epoch, y=point_avg_km_error)

                print('%s average error: ±%.3f' % (self.label_types[i], point_avg_km_error))

    def add_predicts(self, predicts):
        predicts = self.tensor_to_numpy(predicts)
        self.predicts = predicts if self.predicts is None else np.concatenate((self.predicts, predicts))

    def add_labels(self, labels):
        labels = self.tensor_to_numpy(labels)
        self.labels = labels if self.labels is None else np.concatenate((self.labels, labels))

    def normalize_predicts_and_labels(self):
        self.predicts[:, 0] *= config.generate.dist_between_moon_high_bound_km
        self.labels[:, 0] *= config.generate.dist_between_moon_high_bound_km

        if len(self.label_types) >= 3:
            self.predicts[:, 1: 3] *= 360
            self.labels[:, 1: 3] *= 360

        if len(self.label_types) > 3:
            point_gl_to_km_weight = config.generate.gl_to_km / config.dataset.normalize_point_weight
            self.predicts[:, 3:] *= point_gl_to_km_weight
            self.labels[:, 3:] *= point_gl_to_km_weight

    @staticmethod
    def transform_spherical_angle_label(predicts, labels):
        if len(config.dataset.labels) < 2:
            return predicts, labels

        tmp = torch.zeros((config.network.batch_size, 2), dtype=torch.float).to(config.cuda.device)

        predicts[:, 1: 3] = torch.remainder(predicts[:, 1: 3], 1)

        over_one_radius_indices = torch.abs(predicts[:, 1:3] - labels[:, 1:3]) > 0.5

        tmp[over_one_radius_indices & (labels[:, 1:3] < predicts[:, 1:3])] = 1
        tmp[over_one_radius_indices & (labels[:, 1:3] >= predicts[:, 1:3])] = -1

        labels[:, 1:3] += tmp

        return predicts, labels
