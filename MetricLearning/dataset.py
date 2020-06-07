import cv2
import torch
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from config import MARGIN


class MetricDataset(Dataset):
    def __init__(self, dataset_type):
        super().__init__()
        self.dataset_type = dataset_type
        self.combination = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        dataset_size = {
            'train': 16000,
            'test': 2000,
            'valid': 2000
        }[self.dataset_type]

        return dataset_size * 10

    def __getitem__(self, item):
        dir_num = int(item / 10)
        if self.dataset_type == 'test':
            dir_num += 16000
        elif self.dataset_type == 'valid':
            dir_num += 18000

        combination = self.combination[item % 10]

        dataset_path = '/data/space/metric_dataset/%s/%d' % (self.dataset_type, dir_num)

        images_path = sorted(glob(dataset_path + '/*.png'))

        s_img = self.load_img(images_path[0])
        p_img = self.load_img(images_path[combination[0]])
        n_img = self.load_img(images_path[combination[1]])
        margin = MARGIN * (combination[1] - combination[0])
        margin = torch.tensor([margin], dtype=torch.float)
        if self.dataset_type == 'test':
            return s_img, p_img, n_img, combination
        return s_img, p_img, n_img, margin

    def load_img(self, img_path):
        img = cv2.imread(img_path)
        assert img.shape == (400, 400, 3)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)

        return img





