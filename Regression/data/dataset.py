import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from .loader import DatasetLoader
from ..generate.config import GL_UNIT_TO_KM, KM_TO_GL_UNIT, MOON_MAX_RADIUS_IN_GL_UNIT
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
        if image.shape[0] > 400:
            image = cv2.pyrDown(image)
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
                               'c_theta': cls.normalize_angle,
                               'c_phi': cls.normalize_angle,
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
        return dist * KM_TO_GL_UNIT + MOON_MAX_RADIUS_IN_GL_UNIT

    @staticmethod
    def normalize_angle(angle):
        return angle

    @staticmethod
    def normalize_point(point):
        return point * KM_TO_GL_UNIT
