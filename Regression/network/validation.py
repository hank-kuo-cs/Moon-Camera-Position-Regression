import torch
import logging
import numpy as np
from tqdm import tqdm
from .network import Network
from ..loss.mse import adjust_azim_labels_to_use_scmse
from ..config import config


class ValidateNetwork(Network):
    def __init__(self, model, loss_func, tensorboard_writer, data_loader, epoch: int = 1):
        super().__init__(model=model,
                         data_loader=data_loader,
                         loss_func=loss_func,
                         tensorboard_writer=tensorboard_writer,
                         epoch=epoch)
        self.predicts = []
        self.labels = []
        self.label_types = config.dataset.labels
        self.avg_epoch_loss = 0.0

    def run_one_epoch(self):
        self.model.eval()

        with torch.no_grad():
            for idx, (inputs, labels) in tqdm(enumerate(self.get_data())):
                predicts = self.model(inputs)

                self.add_predicts(predicts)
                self.add_labels(labels)

                loss = self.loss_func(predicts, labels)
                self.avg_epoch_loss += loss.item()

        self.normalize_predicts_and_labels()
        self.write_epoch_loss()
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
        dist_gts = self.labels[:, 0]

        assert isinstance(dist_predicts, np.ndarray) and isinstance(dist_gts, np.ndarray)

        dist_avg_km_error = self.get_average_error(dist_predicts, dist_gts)
        print('Distance average error: ±%.3f km' % dist_avg_km_error)

        tag = 'validate/dist_km_error'
        self.tensorboard.add_scalar(tag=tag, x=self._epoch, y=dist_avg_km_error)

    def write_angle_error(self):
        if len(self.label_types) < 3:
            return
        elev_predicts, elev_gts = self.predicts[:, 1], self.labels[:, 1]
        azim_predicts, azim_gts = self.predicts[:, 2], self.labels[:, 2]

        assert isinstance(elev_predicts, np.ndarray) and isinstance(elev_gts, np.ndarray)
        assert isinstance(azim_predicts, np.ndarray) and isinstance(azim_gts, np.ndarray)

        elev_avg_degree_error = self.get_average_error(elev_predicts, elev_gts)
        print('elev average error: ±%.3f degree' % elev_avg_degree_error)

        azim_avg_degree_error = self.get_average_error(azim_predicts, azim_gts, is_azim=True)
        print('azim average error: ±%.3f degree' % azim_avg_degree_error)

        tag = 'validate/elev_degree_error'
        self.tensorboard.add_scalar(tag=tag, x=self._epoch, y=elev_avg_degree_error)

        tag = 'validate/azim_degree_error'
        self.tensorboard.add_scalar(tag=tag, x=self._epoch, y=azim_avg_degree_error)

    def write_point_error(self):
        if len(self.label_types) <= 3:
            return

        for i in range(3, len(self.label_types)):
            point_predicts = self.predicts[:, i]
            point_gts = self.labels[:, i]

            assert isinstance(point_predicts, np.ndarray) and isinstance(point_gts, np.ndarray)

            point_avg_km_error = self.get_average_error(point_predicts, point_gts)

            tag = 'validate/%s_error' % self.label_types[i]
            self.tensorboard.add_scalar(tag=tag, x=self._epoch, y=point_avg_km_error)

            print('%s average error: ±%.3f' % (self.label_types[i], point_avg_km_error))

    def add_predicts(self, predicts):
        predicts = self.tensor_to_numpy(predicts)
        self.predicts.append(predicts)

    def add_labels(self, labels):
        labels = self.tensor_to_numpy(labels)
        labels = labels[:, :config.dataset.labels_num]
        self.labels.append(labels)

    def normalize_predicts_and_labels(self):
        self.predicts = np.concatenate(self.predicts)
        self.labels = np.concatenate(self.labels)

        self.predicts[:, 0] *= config.generate.dist_between_moon_high_bound_km
        self.labels[:, 0] *= config.generate.dist_between_moon_high_bound_km

        if len(self.label_types) >= 3:
            # elev
            self.predicts[:, 1] *= 90
            self.labels[:, 1] *= 90

            # azim
            self.predicts[:, 2] *= 360
            self.labels[:, 2] *= 360

        if len(self.label_types) > 3:
            self.predicts[:, 3:] /= config.dataset.normalize_point_weight
            self.labels[:, 3:] /= config.dataset.normalize_point_weight

    @staticmethod
    def get_average_error(predicts, ground_truths, is_azim=False):
        assert isinstance(predicts, np.ndarray) and isinstance(ground_truths, np.ndarray)
        assert predicts.shape == (config.network.batch_size, 1)
        assert ground_truths.shape == (config.network.batch_size, 1)

        if is_azim:
            ground_truths = adjust_azim_labels_to_use_scmse(predicts, ground_truths)

        return float(np.average(np.abs(predicts - ground_truths)))
