import re
import torch
import logging
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import TripletAngleFeatureExtractor
from visualize import TensorboardWriter
from dataset import MetricDataset
from config import DEVICE, BATCH_SIZE


def get_correct_num(s_features, p_features, n_features):
    d_p = abs(s_features - p_features).mean(dim=1)
    d_n = abs(s_features - n_features).mean(dim=1)

    correct_num = (d_p < d_n).sum().item()

    return correct_num


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    tensorboard_writer = TensorboardWriter()

    valid_dataset = MetricDataset('valid')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model_paths = sorted(glob('checkpoint/model_epoch*'))

    model = TripletAngleFeatureExtractor()
    model.to(DEVICE)

    for model_path in model_paths:
        model.load_state_dict(torch.load(model_path))
        epoch_num = int(re.findall(r'epoch(.+?)\.pth', model_path)[0])
        logging.info('Test epoch %d' % epoch_num)

        correct_num = 0

        with torch.no_grad():
            model.eval()
            for i, data in tqdm(enumerate(valid_dataloader)):
                s_imgs, p_imgs, n_imgs, margins = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE), data[3].to(DEVICE)
                s_features, p_features, n_features = model(s_imgs, p_imgs, n_imgs)

                correct_num += get_correct_num(s_features, p_features, n_features)

        accuracy = correct_num / len(valid_dataset) * 100
        logging.info('Epoch %d, Accuracy = %.3f' % (epoch_num, accuracy))
        tensorboard_writer.add_scalar(tag='validate/accuracy', x=epoch_num, y=accuracy)
