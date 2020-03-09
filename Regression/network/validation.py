import torch
import logging
import numpy as np
from tqdm import tqdm
from network.network import Network
from config import config


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

    def write_epoch_loss(self):
        self.avg_epoch_loss /= (config.dataset.train_dataset_num // config.network.batch_size)
        self.tensorboard.write_avg_epoch_loss(epoch=self._epoch, avg_epoch_loss=self.avg_epoch_loss)

    def write_avg_error(self):
        self.write_distance_error()
        self.write_angle_error()
        self.write_point_error()

    def write_distance_error(self):
        dist_predicts = self.predicts[:, 0]
        dist_labels = self.labels[:, 0]

        small_than_10km_indices = dist_labels <= 10

        dist_avg_km_error = np.average(np.abs((dist_predicts - dist_labels)))
        self.tensorboard.write_avg_error(label_type='dist', epoch=self._epoch, avg_error=dist_avg_km_error)
        print('Distance average error: ±%.3f km' % dist_avg_km_error)

        dist_small_than_10km_predicts = dist_predicts[small_than_10km_indices]
        dist_small_than_10km_labels = dist_labels[small_than_10km_indices]

        dist_avg_small_than_10km_error = np.average(np.abs(dist_small_than_10km_predicts - dist_small_than_10km_labels))
        self.tensorboard.write_avg_error(label_type='dist(<=10km)', epoch=self._epoch, avg_error=dist_avg_small_than_10km_error)
        print('Distance average error (<= 10km): ±%.3f km' % dist_avg_small_than_10km_error)

    def write_angle_error(self):
        if len(self.label_types) < 3:
            return

        theta_avg_degree_error = np.average(np.abs(self.predicts[:, 1] - self.labels[:, 1]))
        self.tensorboard.write_avg_error(label_type='c_theta', epoch=self._epoch, avg_error=theta_avg_degree_error)
        print('Camera theta average error: ±%.3f degree' % theta_avg_degree_error)

        phi_avg_degree_error = np.average(np.abs(self.predicts[:, 2] - self.labels[:, 2]))
        self.tensorboard.write_avg_error(label_type='c_phi', epoch=self._epoch, avg_error=phi_avg_degree_error)
        print('Camera phi average error: ±%.3f degree' % phi_avg_degree_error)

    def write_point_error(self):
        if len(self.label_types) > 3:
            for i in range(3, len(self.label_types)):
                point_predicts = self.predicts[:, i]
                point_labels = self.labels[:, i]

                point_avg_km_error = np.average(np.abs(point_predicts - point_labels))
                self.tensorboard.write_avg_error(label_type=self.label_types[i], epoch=self._epoch, avg_error=point_avg_km_error)
                print('%s average error: ±%.3f' % (self.label_types[i], point_avg_km_error))

    def add_predicts(self, predicts):
        predicts = self.tensor_to_numpy(predicts)
        self.predicts = predicts if self.predicts is None else np.concatenate((self.predicts, predicts))

    def add_labels(self, labels):
        labels = self.tensor_to_numpy(labels)
        self.labels = labels if self.labels is None else np.concatenate((self.labels, labels))

    def normalize_predicts_and_labels(self):
        self.predicts[:, 0] *= config.dataset.dist_range
        self.labels[:, 0] *= config.dataset.dist_range

        if len(self.label_types) >= 3:
            self.predicts[:, 1: 3] *= 360
            self.labels[:, 1: 3] *= 360

        if len(self.label_types) > 3:
            self.predicts[:, 3:] *= config.dataset.normalize_point_weight
            self.labels[:, 3:] *= config.dataset.normalize_point_weight

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
