import os
import cv2
import json
import numpy as np
from .label import LabelGenerator
from .camera import RandomCameraGenerator
from .light import RandomLightGenerator
from ...config import config


class DatasetWriter:
    def __init__(self):
        self._data_idx = 0

        self.camera_generator = RandomCameraGenerator()
        self.light_generator = RandomLightGenerator()
        self.label_generator = LabelGenerator()

        self.labels = []

        self._make_dataset_dir()

    @property
    def dataset_path(self):
        return config.dataset.dataset_path

    @property
    def data_num(self):
        return config.dataset.dataset_num

    @property
    def now_dataset_type(self):
        if self._data_idx < (self.data_num // 10) * 8:
            return 'train'
        elif self._data_idx < (self.data_num // 10) * 9:
            return 'test'
        elif self._data_idx < (self.data_num // 10) * 10:
            return 'validation'
        else:
            return 'end'

    @property
    def is_label_subset_end(self):
        return self._data_idx % (self.data_num // 10) == 0

    @property
    def now_label_subset_num(self):
        return self._data_idx // (self.data_num // 10)

    def write_data(self, image: np.ndarray):
        self._save_image(image)
        self._save_label()

        self._data_idx += 1

        if self.is_label_subset_end:
            self._export_label_to_json()

    def get_random_cameras(self):
        return self.camera_generator.get_random_camera()

    def get_random_light(self):
        pass

    def _save_image(self, image):
        self.check_image(image)
        image = self.pyr_down_image(image)

        save_image = {'train': self._save_train_image,
                      'test': self._save_test_image,
                      'validation': self._save_validation_image}
        save_image[self.now_dataset_type](image)

    def _save_train_image(self, image):
        sub_dir_num = str(self._data_idx // (self.data_num // 10))
        img_path = os.path.join(self.dataset_path, 'train/image', sub_dir_num, '%d.png' % self._data_idx)
        cv2.imwrite(img_path, image)

    def _save_test_image(self, image):
        img_path = os.path.join(self.dataset_path, 'test/image/0', '%d.png' % (self._data_idx % (self.data_num // 10)))
        cv2.imwrite(img_path, image)

    def _save_validation_image(self, image):
        img_path = os.path.join(self.dataset_path, 'validation/image/0', '%d.png' % (self._data_idx % (self.data_num // 10)))
        cv2.imwrite(img_path, image)

    def _save_label(self):
        dist = self.camera_generator.dist
        elev = self.camera_generator.elev
        azim = self.camera_generator.azim
        at = self.camera_generator.at
        up = self.camera_generator.up

        self.label_generator.set_view(dist=dist, elev=elev, azim=azim, at=at, up=up)
        label = self.label_generator.get_label()

        self.check_label(label)

        self.labels.append(label)

    def _export_label_to_json(self):
        if self.now_label_subset_num == 9:
            label_path = os.path.join(self.dataset_path, 'test/label', '0.json')
        elif self.now_label_subset_num == 10:
            label_path = os.path.join(self.dataset_path, 'validation/label', '0.json')
        else:
            label_path = os.path.join(self.dataset_path, 'train/label', '%d.json' % (self.now_label_subset_num-1))

        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f)

        self.labels.clear()

    def _make_dataset_dir(self):
        self.check_data_num()
        dataset_types = ['train', 'test', 'validation']

        for dataset_type in dataset_types:
            image_path = os.path.join(self.dataset_path, dataset_type, 'image')

            subdir_num = 8 if dataset_type == 'train' else 1
            for i in range(subdir_num):
                self.make_directories(path=os.path.join(image_path, str(i)))

            label_path = os.path.join(self.dataset_path, dataset_type, 'label')
            self.make_directories(path=label_path)

    def check_data_num(self):
        try:
            assert self.data_num % 10 == 0
        except AssertionError:
            raise AssertionError('Data Num must be multiplies of 10!')

    @staticmethod
    def make_directories(path):
        assert isinstance(path, str)

        os.makedirs(path, exist_ok=True)

    @staticmethod
    def pyr_down_image(image):
        if image.shape[0] > 600:
            image = cv2.pyrDown(image)
        return image

    @staticmethod
    def check_image(image):
        assert isinstance(image, np.ndarray)
        assert image.shape[0] == config.generate.image_size
        assert image.shape[1] == config.generate.image_size

    @staticmethod
    def check_label(label):
        assert isinstance(label, dict)

        keys = ['dist', 'elev', 'azim', 'p_x', 'p_y', 'p_z', 'u_x', 'u_y', 'u_z']
        for key in keys:
            assert key in label

        assert isinstance(label['dist'], float)
        assert isinstance(label['elev'], float)
        assert isinstance(label['azim'], float)
        assert isinstance(label['p_x'], float)
        assert isinstance(label['p_y'], float)
        assert isinstance(label['p_z'], float)
        assert isinstance(label['u_x'], float)
        assert isinstance(label['u_y'], float)
        assert isinstance(label['u_z'], float)
