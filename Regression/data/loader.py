import os
import json
from glob import glob
from ..config import config


class DatasetLoader:
    def __init__(self, dataset_type):
        self._dataset_type = dataset_type
        self._dataset_path = None
        self.images_path = []
        self.labels = []

        self.set_dataset_path()
        self.load_dataset()

    def set_dataset_path(self):
        self._dataset_path = os.path.join(config.dataset.dataset_path, self._dataset_type)

    def load_dataset(self):
        self.load_images_path()
        self.load_labels()

    def load_images_path(self):
        sub_datset_size = config.dataset.sub_dataset_size
        sub_dirs_path = self.get_sub_dirs_path()
        for sub_num, sub_dir in enumerate(sub_dirs_path):
            self.check_sub_dir_image_num(sub_dir)
            images_path_in_one_sub_dir = ['%s/%d.png' % (sub_dir, sub_num * sub_datset_size + i) for i in range(sub_datset_size)]
            self.images_path += images_path_in_one_sub_dir

    def load_labels(self):
        labels_path = self.get_labels_path()
        for label_path in labels_path:
            labels = self.get_content_in_json(label_path)
            self.labels += labels

    def get_sub_dirs_path(self):
        image_dir_path = os.path.join(self._dataset_path, 'image/*')
        return sorted(glob(image_dir_path))

    def get_labels_path(self):
        labels_path = os.path.join(self._dataset_path, 'label/*')
        return sorted(glob(labels_path))

    @staticmethod
    def check_sub_dir_image_num(sub_dir):
        images_path = glob(sub_dir + '/*.png')
        try:
            assert len(images_path) == config.dataset.sub_dataset_size
        except AssertionError:
            raise AssertionError('Number of images (%d) in one subdirectory (%s) not fit the config setting (%d)'
                                 % (len(images_path), sub_dir, config.dataset.sub_dataset_size))

    @staticmethod
    def get_content_in_json(json_file):
        assert isinstance(json_file, str)
        assert json_file[-5:] == '.json'

        with open(json_file, 'r') as f:
            labels = json.load(f)

        return labels
