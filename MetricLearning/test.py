import re
import torch
import logging
import argparse
from glob import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import TripletAngleFeatureExtractor
from dataset import MetricDataset
from config import DEVICE, BATCH_SIZE


def get_correct_num(s_features, p_features, n_features):
    d_p = abs(s_features - p_features).mean(dim=1)
    d_n = abs(s_features - n_features).mean(dim=1)

    correct_num = (d_p < d_n).sum().item()

    return correct_num


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epoch_of_pretrain', type=str,
                        help='Use a pretrain model in checkpoint directory to test')
    arguments = parser.parse_args()

    test_dataset = MetricDataset('test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    epoch_to_test = int(arguments.epoch_of_pretrain)
    model_path = sorted(glob('checkpoint/model_epoch*'))[-1] if not epoch_to_test else 'checkpoint/model_epoch%.3d.pth' % epoch_to_test
    epoch_num = re.findall(r'epoch(.+?)\.pth', model_path)[0]

    model = TripletAngleFeatureExtractor()
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_path))

    correct_num = 0
    metric_correct_num = {}

    logging.info('Test epoch %d' % int(epoch_num))
    with torch.no_grad():
        model.eval()
        for i, data in tqdm(enumerate(test_dataloader)):
            s_imgs, p_imgs, n_imgs, combinations = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE), data[3]
            print(combinations.size())
            s_features, p_features, n_features = model(s_imgs, p_imgs, n_imgs)

            combinations = combinations.tolist()

            for b in range(s_features.size(0)):
                if combinations[b] not in metric_correct_num:
                    metric_correct_num[combinations[b]] = 0

                d_p = abs(s_features[b] - p_features[b]).mean()
                d_n = abs(s_features[b] - n_features[b]).mean()

                if d_p < d_n:
                    metric_correct_num[combinations[b]] += 1

            correct_num += get_correct_num(s_features, p_features, n_features)

    accuracy = correct_num / len(test_dataset) * 100
    logging.info('Accuracy = %.3f' % accuracy)

    print('Each interval accuracy:')
    for k, v in metric_correct_num.items():
        print(k, v / len(test_dataset) * 10)
