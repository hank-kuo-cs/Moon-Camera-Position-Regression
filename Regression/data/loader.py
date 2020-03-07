import os
import json
from glob import glob
from config import config


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
        sub_dirs_path = self.get_sub_dirs_path()
        for sub_dir in sub_dirs_path:
            images_path_in_one_sub_dir = ['%s/%d.png' % (sub_dir, i) for i in range(config.dataset.sub_dataset_size)]
            print(images_path_in_one_sub_dir)
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
    def get_content_in_json(json_file):
        assert isinstance(json_file, str)
        assert json_file[-5:] == '.json'

        with open(json_file, 'r') as f:
            labels = json.load(f)

        return labels
