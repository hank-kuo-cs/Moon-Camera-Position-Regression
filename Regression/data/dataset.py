import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from .loader import DatasetLoader
from ..config import config


class MoonDataset(Dataset):
    def __init__(self, data_type: str):
        assert data_type in ['train', 'test', 'validation']

        self.dataset_loader = DatasetLoader(data_type)

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

        assert isinstance(image, torch.FloatTensor)
        return image

    def get_label(self, item):
        label = self.dataset_loader.labels[item]
        label = self.refine_label(label)

        assert isinstance(label, torch.FloatTensor)
        return label

    @staticmethod
    def refine_image(image):
        image_size = config.generate.image_size

        if image.shape[0] > image_size:
            image = cv2.pyrDown(image, dstsize=(image_size, image_size))

        image = cv2.equalizeHist(image)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        image = transform(image)

        return image

    @classmethod
    def refine_label(cls, label: dict):
        refined_label = []
        for key in label.keys():
            refined_label.append(cls.normalize_label(key, label[key]))

        refined_label = np.array(refined_label)
        refined_label = torch.from_numpy(refined_label).float()

        return refined_label

    @classmethod
    def normalize_label(cls, label_type, value):
        normalize_func_dict = {'dist': cls.normalize_dist,
                               'elev': cls.normalize_elev,
                               'azim': cls.normalize_azim,
                               'p_x': cls.normalize_point,
                               'p_y': cls.normalize_point,
                               'p_z': cls.normalize_point,
                               'u_x': cls.normalize_point,
                               'u_y': cls.normalize_point,
                               'u_z': cls.normalize_point}

        normalize_func = normalize_func_dict[label_type]
        return normalize_func(value)

    @staticmethod
    def normalize_dist(dist):
        dist = (dist - config.generate.moon_radius_gl)
        dist /= config.generate.dist_between_moon_high_bound_km * config.generate.km_to_gl
        if dist < 0 or dist > 1:
            raise ValueError('dist must be normalized to [0, 1]')
        return dist

    @staticmethod
    def normalize_elev(elev):
        elev = elev / (np.pi / 2)
        if elev < -1 or elev > 1:
            raise ValueError('elev must be normalized to [-1, 1]')
        return elev

    @staticmethod
    def normalize_azim(azim):
        azim = azim / (np.pi * 2)
        if azim < 0 or azim > 1:
            raise ValueError('azim must be normalized to [0, 1]')
        return azim

    @staticmethod
    def normalize_point(point):
        weight = config.dataset.normalize_point_weight
        return point * weight
