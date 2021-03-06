import torch
import numpy as np
from .network import Network
from .fine_tune import FineTuner
from ..config import config
from tqdm import tqdm


class TestNetwork(Network):
    def __init__(self, model, data_loader, loss_func, epoch, is_fine_tune=False, small_dataset_size=0):
        super().__init__(model=model, data_loader=data_loader, loss_func=loss_func, epoch=epoch)
        self.is_fine_tune = is_fine_tune
        self.small_dataset_size = small_dataset_size

        self.predicts = []
        self.fine_tuned_predicts = []
        self.labels = []
        self.label_types = config.dataset.labels
        self.avg_loss = 0.0
        self.small_than_5km_indices = None
        self.fine_tuner = FineTuner()

    def run_one_epoch(self):
        self.model.eval()
        batch_size = config.network.batch_size

        for idx, (inputs, labels) in tqdm(enumerate(self.get_data())):
            with torch.no_grad():
                predicts = self.model(inputs)

            if 0 < self.small_dataset_size <= batch_size * idx:
                break

            self.add_predicts(predicts)
            self.add_labels(labels)

            loss = self.loss_func(predicts, labels)
            self.avg_loss += loss.item()

            if self.is_fine_tune:
                fine_tuned_predicts = self.fine_tuner.fine_tune_predict_positions(target_images=inputs,
                                                                                  predict_positions=predicts)
                self.add_fine_tuned_predicts(fine_tuned_predicts)

            del predicts, labels

        self.avg_loss /= (config.dataset.test_dataset_num // config.network.batch_size)

        self.normalize_predicts_and_labels()
        self.show_some_results()
        self.show_avg_loss()
        self.show_avg_error()

    def show_some_results(self):
        for i in range(0, config.network.batch_size):
            print('dist and xyz (km), phi and theta (degree)')
            print('%d-th\tlabel\tpredict\tfine tuned predict' % (i + 1))

            for j in range(len(self.label_types)):
                fine_tuned_predict = self.fine_tuned_predicts[i][j] if self.is_fine_tune else 0.0
                print('%s\t%.3f\t%.3f\t%.3f' %
                      (self.label_types[j], self.labels[i][j], self.predicts[i][j], fine_tuned_predict))
            print()

    def show_avg_loss(self):
        print('Average testing loss = %.6f' % self.avg_loss)

    def show_avg_error(self):
        dist_predicts = self.predicts[:, 0]
        dist_gts = self.labels[:, 0]
        self.show_distance_error(dist_predicts, dist_gts)

        elev_predicts, elev_gts = self.predicts[:, 1], self.labels[:, 1]
        azim_predicts, azim_gts = self.predicts[:, 2], self.labels[:, 2]
        self.show_angle_error(elev_predicts, elev_gts, azim_predicts, azim_gts)

        if not self.is_fine_tune:
            return

        print('\n========After fine tune========')
        dist_predicts = self.fine_tuned_predicts[:, 0]
        self.show_distance_error(dist_predicts, dist_gts)

        elev_predicts = self.fine_tuned_predicts[:, 1]
        azim_predicts = self.fine_tuned_predicts[:, 2]
        self.show_angle_error(elev_predicts, elev_gts, azim_predicts, azim_gts)

        # self.show_point_error()

    def show_distance_error(self, dist_predicts, dist_gts):
        assert isinstance(dist_predicts, np.ndarray) and isinstance(dist_gts, np.ndarray)

        self.small_than_5km_indices = dist_gts <= 5

        dist_avg_km_error = self.get_average_error(dist_predicts, dist_gts)
        print('Distance average error: ±%.3f km' % dist_avg_km_error)

        dist_avg_small_than_5km_error = self.get_average_error(dist_predicts[self.small_than_5km_indices],
                                                               dist_gts[self.small_than_5km_indices])
        print('Distance average error (<= 5km): ±%.3f km' % dist_avg_small_than_5km_error)

    def show_angle_error(self, elev_predicts, elev_gts, azim_predicts, azim_gts):
        if len(self.label_types) < 3:
            return
        elev_avg_degree_error = self.get_average_error(elev_predicts, elev_gts)
        print('elev average error: ±%.3f degree' % elev_avg_degree_error)

        azim_avg_degree_error = self.get_average_error(azim_predicts, azim_gts, is_azim=True)
        print('azim average error: ±%.3f degree' % azim_avg_degree_error)

        elev_degree_error_small_5km = self.get_average_error(elev_predicts[self.small_than_5km_indices],
                                                             elev_gts[self.small_than_5km_indices])
        print('elev average error (dist <= 5km): ±%.3f degree' % elev_degree_error_small_5km)

        azim_degree_error_small_5km = self.get_average_error(azim_predicts[self.small_than_5km_indices],
                                                             azim_gts[self.small_than_5km_indices],
                                                             is_azim=True)
        print('azim average error (dist <= 5km): ±%.3f degree' % azim_degree_error_small_5km)

    # def show_point_error(self):
    #     if len(self.label_types) <= 3:
    #         return
    #
    #     for i in range(3, len(self.label_types)):
    #         point_predicts = self.predicts[:, i]
    #         point_gts = self.labels[:, i]
    #
    #         point_avg_km_error = self.get_average_error(point_predicts, point_gts)
    #         print('%s average error: ±%.3f' % (self.label_types[i], point_avg_km_error))

    def add_predicts(self, predicts):
        predicts = self.tensor_to_numpy(predicts)
        self.predicts.append(predicts)

    def add_fine_tuned_predicts(self, fine_tuned_predicts):
        fine_tuned_predicts = self.tensor_to_numpy(fine_tuned_predicts)
        self.fine_tuned_predicts.append(fine_tuned_predicts)

    def add_labels(self, labels):
        labels = self.tensor_to_numpy(labels)
        labels = labels[:, :config.dataset.labels_num]
        self.labels.append(labels)

    def normalize_predicts_and_labels(self):
        self.predicts = np.concatenate(self.predicts)
        self.labels = np.concatenate(self.labels)

        self.predicts[:, 0] *= config.generate.dist_between_moon_high_bound_km
        self.labels[:, 0] *= config.generate.dist_between_moon_high_bound_km

        if self.is_fine_tune:
            self.fine_tuned_predicts = np.concatenate(self.fine_tuned_predicts)
            self.fine_tuned_predicts[:, 0] *= config.generate.dist_between_moon_high_bound_km

        if len(self.label_types) >= 3:
            # elev
            self.predicts[:, 1] *= 90
            self.labels[:, 1] *= 90

            # azim
            self.predicts[:, 2] *= 360
            self.labels[:, 2] *= 360

            if self.is_fine_tune:
                self.fine_tuned_predicts[:, 1] *= 90
                self.fine_tuned_predicts[:, 2] *= 360

        # if len(self.label_types) > 3:
        #     self.predicts[:, 3:] /= config.dataset.normalize_point_weight
        #     self.fine_tuned_predicts[:, 3:] /= config.dataset.normalize_point_weight
        #     self.labels[:, 3:] /= config.dataset.normalize_point_weight

    def get_average_error(self, predicts, ground_truths, is_azim=False):
        assert isinstance(predicts, np.ndarray) and isinstance(ground_truths, np.ndarray)
        assert predicts.shape[0] <= config.dataset.test_dataset_num and predicts.ndim == 1
        assert ground_truths.shape[0] <= config.dataset.test_dataset_num and ground_truths.ndim == 1

        if is_azim:
            ground_truths = self.adjust_azim_degrees_to_use_scmse(predicts, ground_truths)

        return float(np.average(np.abs(predicts - ground_truths)))
