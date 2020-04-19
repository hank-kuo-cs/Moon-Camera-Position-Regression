import torch
import numpy as np
from .network import Network
from ..config import config
from tqdm import tqdm


class TestNetwork(Network):
    def __init__(self, model, data_loader, loss_func, epoch):
        super().__init__(model=model, data_loader=data_loader, loss_func=loss_func, epoch=epoch)
        self.predicts = None
        self.labels = None
        self.label_types = config.dataset.labels
        self.avg_loss = 0.0
        self.small_than_5km_indices = None

    def run_one_epoch(self):
        self.model.eval()

        with torch.no_grad():
            for idx, (inputs, labels) in tqdm(enumerate(self.get_data())):
                predicts = self.model(inputs)

                predicts, labels = self.transform_spherical_angle_label(predicts, labels)

                self.add_predicts(predicts)
                self.add_labels(labels)

                loss = self.loss_func(predicts.clone(), labels.clone())
                self.avg_loss += loss.item()

        self.avg_loss /= (config.dataset.test_dataset_num // config.network.batch_size)

        self.normalize_predicts_and_labels()
        self.show_some_results()
        self.show_avg_loss()
        self.show_avg_error()

    def show_some_results(self):
        for i in range(0, 10):
            print('dist and xyz (km), phi and theta (degree)')
            print('%d-th\tpredict\tlabel' % (i + 1))

            for j in range(len(self.label_types)):
                print('%s\t%.3f\t%.3f' % (self.label_types[j], self.predicts[i][j], self.labels[i][j]))
            print()

    def show_avg_loss(self):
        print('Average testing loss = %.6f' % self.avg_loss)

    def show_avg_error(self):
        self.show_distance_error()
        self.show_angle_error()
        self.show_point_error()

    def show_distance_error(self):
        dist_predicts = self.predicts[:, 0]
        dist_labels = self.labels[:, 0]

        self.small_than_5km_indices = dist_labels <= 5

        dist_avg_km_error = np.average(np.abs((dist_predicts - dist_labels)))
        print('Distance average error: ±%.3f km' % dist_avg_km_error)

        dist_small_than_5km_predicts = dist_predicts[self.small_than_5km_indices]
        dist_small_than_5km_labels = dist_labels[self.small_than_5km_indices]

        dist_avg_small_than_5km_error = np.average(np.abs(dist_small_than_5km_predicts - dist_small_than_5km_labels))
        print('Distance average error (<= 5km): ±%.3f km' % dist_avg_small_than_5km_error)

    def show_angle_error(self):
        if len(self.label_types) < 3:
            return
        elev_predicts, elev_labels = self.predicts[:, 1], self.labels[:, 1]
        azim_predicts, azim_labels = self.predicts[:, 2], self.labels[:, 2]

        elev_avg_degree_error = np.average(np.abs(elev_predicts - elev_labels))
        print('elev average error: ±%.3f degree' % elev_avg_degree_error)

        azim_avg_degree_error = np.average(np.abs(azim_predicts - azim_labels))
        print('azim average error: ±%.3f degree' % azim_avg_degree_error)

        elev_predicts_small_5km = elev_predicts[self.small_than_5km_indices]
        elev_labels_small_5km = elev_labels[self.small_than_5km_indices]

        elev_degree_error_small_5km = np.average(np.abs(elev_predicts_small_5km - elev_labels_small_5km))

        azim_predicts_small_5km = azim_predicts[self.small_than_5km_indices]
        azim_labels_small_5km = azim_labels[self.small_than_5km_indices]

        azim_degree_error_small_5km = np.average(np.abs(azim_predicts_small_5km - azim_labels_small_5km))

        print('elev average error (dist <= 5km): ±%.3fkm' % elev_degree_error_small_5km)
        print('azim average error (dist <= 5km): ±%.3fkm' % azim_degree_error_small_5km)

    def show_point_error(self):
        if len(self.label_types) > 3:
            for i in range(3, len(self.label_types)):
                point_predicts = self.predicts[:, i]
                point_labels = self.labels[:, i]

                point_avg_km_error = np.average(np.abs(point_predicts - point_labels))
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
            self.predicts[:, 3:] /= config.dataset.normalize_point_weight
            self.labels[:, 3:] /= config.dataset.normalize_point_weight

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
