import logging
import numpy as np
from .network import Network
from ..config import config


class TrainNetwork(Network):
    def __init__(self,
                 model,
                 loss_func,
                 optimizer,
                 tensorboard_writer,
                 data_loader,
                 epoch: int = 1):
        super().__init__(model=model,
                         data_loader=data_loader,
                         loss_func=loss_func,
                         optimizer=optimizer,
                         tensorboard_writer=tensorboard_writer,
                         epoch=epoch)

        self.avg_step_loss = 0.0
        self.avg_epoch_loss = 0.0
        self.features_list = []
        self.labels_list = []
        self.batch_size = config.network.batch_size
        self.tsne_epoch_step = config.tensorboard.tsne_epoch_step
        self.is_write_tsne = config.tensorboard.is_write_tsne

    def run_one_epoch(self):
        self.model.train()

        for idx, (inputs, labels) in enumerate(self.get_data()):
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            # write tsne
            if idx < (100 // self.batch_size) and self._epoch % self.tsne_epoch_step == 0 and self.is_write_tsne:
                self.features_list.append(self.model.features)
                self.labels_list.append(labels.clone().numpy())

            loss = self.loss_func(outputs, labels)

            loss.backward()
            self.optimizer.step()

            self.show_step_loss(loss=loss.item(), step=idx+1)

        self.save_model()
        self.show_epoch_loss()
        self.show_tsne()
        self._epoch += 1

    def show_step_loss(self, loss, step):
        self.avg_step_loss += loss
        self.avg_epoch_loss += loss

        if step % config.tensorboard.loss_step == 0:
            self.avg_step_loss /= config.tensorboard.loss_step
            logging.info('epoch %d, %d step, loss = %.6f' % (self._epoch, step, self.avg_step_loss))

            tag = 'train/step_loss'
            x = step + self.steps_of_an_epoch * (self._epoch - 1)
            y = self.avg_step_loss
            self.tensorboard.add_scalar(tag=tag, x=x, y=y)

            self.avg_step_loss = 0.0

    def show_epoch_loss(self):
        self.avg_epoch_loss /= self.steps_of_an_epoch

        logging.info('Writing epoch loss...')

        tag = 'train/epoch_loss'
        x = self._epoch
        y = self.avg_epoch_loss
        self.tensorboard.add_scalar(tag=tag, x=x, y=y)

        self.avg_epoch_loss = 0.0

    def show_tsne(self):
        if not len(self.features_list):
            return

        logging.info('Writing tsne...')

        label_names = config.dataset.labels
        features = np.concatenate(self.features_list)
        labels_list = np.concatenate(self.labels_list)

        for i, label_name in enumerate(label_names):
            tag = 'epoch%.3d/%s' % (self._epoch, label_name)
            labels = labels_list[:, i]
            self.tensorboard.add_embedding(tag=tag, features=features, labels=labels)

        self.features_list.clear()
        self.labels_list.clear()

    @property
    def steps_of_an_epoch(self):
        return config.dataset.train_dataset_num // config.network.batch_size
