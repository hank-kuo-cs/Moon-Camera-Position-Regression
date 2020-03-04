import logging
from network.network import Network
from config import config


class TrainNetwork(Network):
    def __init__(self, model, loss_func, optimizer, tensorboard_writer, data_loader, epoch: int = 1):
        super().__init__(model=model,
                         data_loader=data_loader,
                         loss_func=loss_func,
                         optimizer=optimizer,
                         tensorboard_writer=tensorboard_writer,
                         epoch=epoch)
        self.avg_step_loss = 0.0
        self.avg_epoch_loss = 0.0

    def run_one_epoch(self):
        self.model.train()

        for idx, (inputs, labels) in enumerate(self.get_data()):
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss_func(outputs, labels)

            loss.backward()
            self.optimizer.step()

            self.show_step_loss(loss=loss.item(), step=idx+1)

        self.save_model()
        self.show_epoch_loss()
        self._epoch += 1

    def show_step_loss(self, loss, step):
        self.avg_step_loss += loss
        self.avg_epoch_loss += loss

        if step % config.tensorboard.loss_step == 0:
            self.avg_step_loss /= config.tensorboard.loss_step
            logging.info('epoch %d, %d step, loss = %.6f' % (self._epoch, step, self.avg_step_loss))
            self.tensorboard.write_avg_step_loss(step=step, epoch=self._epoch, avg_step_loss=self.avg_step_loss)

            self.avg_step_loss = 0.0

    def show_epoch_loss(self):
        self.avg_epoch_loss /= (config.dataset.train_dataset_num // config.network.batch_size)
        self.tensorboard.write_avg_epoch_loss(epoch=self._epoch, avg_epoch_loss=self.avg_epoch_loss)
        self.avg_epoch_loss = 0.0
