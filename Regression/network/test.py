import numpy as np
from network.network import Network
from config import config
from tqdm import tqdm


class TestNetwork(Network):
    def __init__(self, model, data_loader, loss_func, epoch):
        super().__init__(model=model, data_loader=data_loader, loss_func=loss_func, epoch=epoch)
        self.predicts = None
        self.labels = None
        self.label_types = config.dataset.labels
        self.avg_loss = 0.0

    def run_one_epoch(self):
        for idx, (inputs, labels) in tqdm(enumerate(self.get_data())):
            predicts = self.model(inputs)

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
        for i in range(0, config.dataset.test_dataset_num, config.network.batch_size):
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

        dist_avg_km_error = np.average(np.abs((dist_predicts - dist_labels)))
        print('Distance average error: ±%.3f km' % dist_avg_km_error)

    def show_angle_error(self):
        if len(self.label_types) < 3:
            return

        tmp = np.zeros((config.network.batch_size, 2), dtype=np.float)

        over_one_radius_indices = np.abs(self.predicts[:, 1:3] - self.labels[:, 1:3]) > 0.5

        tmp[over_one_radius_indices & (self.predicts[:, 1:3] < self.labels[:, 1:3])] = 1
        tmp[over_one_radius_indices & (self.predicts[:, 1:3] >= self.labels[:, 1:3])] = -1

        self.labels[:, 1:3] += tmp

        theta_avg_degree_error = np.average(np.abs(self.predicts[:, 1] - self.labels[:, 1]))
        print('Camera theta average error: ±%.3f degree' % theta_avg_degree_error)

        phi_avg_degree_error = np.average(np.abs(self.predicts[:, 2] - self.labels[:, 2]))
        print('Camera phi average error: ±%.3f degree' % phi_avg_degree_error)

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
        self.predicts[:, 0] *= config.dataset.dist_range
        self.labels[:, 0] *= config.dataset.dist_range

        if len(self.label_types) >= 3:
            self.predicts[:, 1: 3] *= 360
            self.labels[:, 1: 3] *= 360

        if len(self.label_types) > 3:
            self.predicts[:, 3:] *= config.dataset.normalize_point_weight
            self.labels[:, 3:] *= config.dataset.normalize_point_weight
