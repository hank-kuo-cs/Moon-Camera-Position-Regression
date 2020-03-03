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
        self.show_loss = 0.0

    def run_one_epoch(self):
        for idx, (inputs, labels) in enumerate(self.get_data()):
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.loss_func(outputs, labels)

            loss.backward()
            self.optimizer.step()

            self.show_step_loss(loss=loss.item(), step=idx+1)
            self.tensorboard.add_loss(loss.item())
            self.tensorboard.write_step_loss()

        self.save_model()
        self.tensorboard.write_epoch_loss()
        self._epoch += 1

    def show_step_loss(self, loss, step):
        self.show_loss += loss
        if step % config.tensorboard.loss_step == 0:
            avg_step_loss = self.show_loss / config.tensorboard.loss_step
            logging.info('epoch %d, %d step, loss = %.6f' % (self._epoch, step, avg_step_loss))
            self.show_loss = 0
