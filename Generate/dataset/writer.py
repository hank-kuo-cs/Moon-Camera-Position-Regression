import os
import cv2
import json
import numpy as np
from config import DATASET_PATH, WINDOW_HEIGHT, WINDOW_WIDTH, DATA_NUM, PYR_DOWN_TIME
from model import Moon
from dataset.label import LabelGenerator
from dataset.view import RandomViewGenerator
from dataset.light import RandomLightGenerator


class DatasetWriter:
    def __init__(self, moon: Moon):
        self._data_idx = 0
        self.dataset_path = DATASET_PATH
        self.moon = moon

        self.view_generator = RandomViewGenerator()
        self.light_generator = RandomLightGenerator()
        self.label_generator = LabelGenerator(moon.obj.vertices)

        self.labels = []

        self._make_dataset_dir()

    @property
    def now_dataset_type(self):
        if self._data_idx < (DATA_NUM // 10) * 8:
            return 'train'
        elif self._data_idx < (DATA_NUM // 10) * 9:
            return 'test'
        elif self._data_idx < (DATA_NUM // 10) * 10:
            return 'validation'
        else:
            return 'end'

    @property
    def is_label_subset_end(self):
        return self._data_idx % (DATA_NUM // 10) == 0

    @property
    def now_label_subset_num(self):
        return self._data_idx // (DATA_NUM // 10)

    def write_data(self, image: np.ndarray, moon: Moon):
        self.moon = moon

        self._save_image(image)
        self._save_label()

        self._data_idx += 1

        if self.is_label_subset_end:
            self._export_label_to_json()

    def get_moon_view(self):
        return self.view_generator.get_moon_view()

    def get_random_light(self):
        pass

    def _save_image(self, image):
        self.check_image(image)
        image = self.resize_image(image)

        save_image = {'train': self._save_train_image,
                      'test': self._save_test_image,
                      'validation': self._save_validation_image}
        save_image[self.now_dataset_type](image)

    def _save_train_image(self, image):
        sub_dir_num = str(self._data_idx // (DATA_NUM // 10))
        img_path = os.path.join(DATASET_PATH, 'train/image', sub_dir_num, '%d.png' % self._data_idx)
        cv2.imwrite(img_path, image)

    def _save_test_image(self, image):
        img_path = os.path.join(DATASET_PATH, 'test/image/0', '%d.png' % (self._data_idx % (DATA_NUM // 10)))
        cv2.imwrite(img_path, image)

    def _save_validation_image(self, image):
        img_path = os.path.join(DATASET_PATH, 'validation/image/0', '%d.png' % (self._data_idx % (DATA_NUM // 10)))
        cv2.imwrite(img_path, image)

    def _save_label(self):
        self.label_generator.set_view(view=self.moon.view, spherical_eye=self.view_generator.spherical_eye)
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

    @staticmethod
    def check_data_num():
        try:
            assert DATA_NUM % 10 == 0
        except AssertionError:
            raise AssertionError('Data Num must be multiplies of 10!')

    @staticmethod
    def make_directories(path):
        assert isinstance(path, str)

        os.makedirs(path, exist_ok=True)

    @staticmethod
    def resize_image(image):
        for i in range(PYR_DOWN_TIME):
            image = cv2.pyrDown(image)
        return image

    @staticmethod
    def check_image(image):
        assert isinstance(image, np.ndarray)
        assert image.shape[0] == WINDOW_HEIGHT
        assert image.shape[1] == WINDOW_WIDTH

    @staticmethod
    def check_label(label):
        assert isinstance(label, dict)

        keys = ['dist', 'c_theta', 'c_phi', 'p_x', 'p_y', 'p_z', 'u_x', 'u_y', 'u_z']
        for key in keys:
            assert key in label

        assert isinstance(label['dist'], float)
        assert isinstance(label['c_theta'], float)
        assert isinstance(label['c_phi'], float)
        assert isinstance(label['p_x'], float)
        assert isinstance(label['p_y'], float)
        assert isinstance(label['p_z'], float)
        assert isinstance(label['u_x'], float)
        assert isinstance(label['u_y'], float)
        assert isinstance(label['u_z'], float)
