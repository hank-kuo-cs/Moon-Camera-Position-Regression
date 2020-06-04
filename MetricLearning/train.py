import os
import torch
import logging
from torch.optim import SGD
from torch.utils.data import DataLoader
from model import TripletAngleFeatureExtractor
from loss import TripletLoss
from visualize import TensorboardWriter
from dataset import MetricDataset
from config import EPOCH_NUM, LEARNING_RATE, BATCH_SIZE, DEVICE


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    os.makedirs('checkpoint', exist_ok=True)

    train_dataset = MetricDataset('train')
    train_dataloader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=4, batch_size=BATCH_SIZE)

    model = TripletAngleFeatureExtractor()
    model = model.to(DEVICE)

    loss_func = TripletLoss()
    optimizer = SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    avg_loss = 0.0
    epoch_loss = 0.0
    tensorboard_writer = TensorboardWriter()

    for epoch in range(EPOCH_NUM):
        n = 0
        logging.info('Epoch %d' % (epoch + 1))
        model.train()

        for i, data in enumerate(train_dataloader):
            n += 1
            optimizer.zero_grad()
            s_imgs, p_imgs, n_imgs, margins = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE), data[3].to(DEVICE)
            batch_size = s_imgs.size(0)
            s_features, p_features, n_features = model(s_imgs, p_imgs, n_imgs)

            loss = loss_func(s_features, p_features, n_features, margins)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            epoch_loss += loss.item()

            if (i + 1) % 100 == 0:
                avg_loss /= 100
                logging.info('Loss step %d = %.6f' % (i+1, avg_loss))
                epoch_steps = (epoch + 1) * (len(train_dataset) / BATCH_SIZE)
                tensorboard_writer.add_scalar(tag='train/step_loss', x=epoch_steps+i+1, y=avg_loss)
                avg_loss = 0
        epoch_loss /= n
        logging.info('Epoch Loss = %.6f' % epoch_loss)
        tensorboard_writer.add_scalar(tag='train/epoch_loss', x=epoch+1, y=epoch_loss)
        epoch_loss = 0.0

        model_path = 'checkpoint/model_epoch%.3d.pth' % (epoch + 1)
        torch.save(model.state_dict(), model_path)
