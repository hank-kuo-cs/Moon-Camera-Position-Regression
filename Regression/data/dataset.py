import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from data.loader import DatasetLoader
from config import config


class MoonDataset(Dataset):
    def __init__(self):
        self.dataset_loader = None

    @property
    def train_dataset(self):
        self.dataset_loader = DatasetLoader('train')
        return self

    @property
    def test_dataset(self):
        self.dataset_loader = DatasetLoader('test')
        return self

    @property
    def validation_dataset(self):
        self.dataset_loader = DatasetLoader('validation')
        return self

    def __len__(self):
        assert len(self.dataset_loader.images_path) == len(self.dataset_loader.labels)
        return len(self.dataset_loader.images_path)

    def __getitem__(self, item):
        image = self.get_image(item)
        label = self.get_label(item)

        return image, label

    def get_image(self, item):
        image_path = self.dataset_loader.images_path[item]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        assert isinstance(image, np.ndarray)
        assert image.shape[0] > 0

        image = self.refine_image(image)

        return image

    def get_label(self, item):
        label = self.dataset_loader.labels[item]
        label = self.refine_label(label)

        return label

    @staticmethod
    def refine_image(image):
        image = cv2.pyrDown(image)
        image = cv2.equalizeHist(image)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        image = transform(image)

        assert isinstance(image, torch.FloatTensor)

        return image

    @classmethod
    def refine_label(cls, label):
        refined_label = []
        labels = config.dataset.labels

        for label_type in labels:
            value = label[label_type]
            refined_label += cls.normalize_label(label_type, value)

        label = np.array(refined_label)
        label = torch.from_numpy(label).float()

        assert isinstance(label, torch.FloatTensor)

        return label

    @classmethod
    def normalize_label(cls, label_type, value):
        if label_type == 'dist':
            normalize_func = cls.normalize_dist
        elif label_type == 'c_theta' or label_type == 'c_phi':
            normalize_func = cls.normalize_angle
        elif label_type == 'p_xyz' or label_type == 'u_xyz':
            normalize_func = cls.normalize_vec
        else:
            raise ValueError('Cannot tell this label type \'%s\'' % label_type)
        return normalize_func(value)

    @staticmethod
    def normalize_dist(dist):
        return [dist / config.dataset.dist_range]

    @staticmethod
    def normalize_angle(angle):
        return [angle / (np.pi * 2)]

    @staticmethod
    def normalize_vec(v_xyz):
        assert isinstance(v_xyz, list)
        assert len(v_xyz) == 3

        for value in v_xyz:
            value /= 1000

        return v_xyz
