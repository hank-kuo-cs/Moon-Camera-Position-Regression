from network import VGG19, Resnet18


class NetworkConfig:
    def __init__(self,
                 network_model: str,
                 batch_size: int = 20,
                 epoch_num: int = 300,
                 learning_rate: float = 0.001,
                 momentum: float = 0.9):

        self._network_model = network_model
        self._epoch_num = epoch_num
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._momentum = momentum

        self.check_parameters()

    @property
    def network_model(self):
        return self._network_model

    @property
    def epoch_num(self):
        return self._epoch_num

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def momentum(self):
        return self._momentum

    def check_parameters(self):
        assert isinstance(self._network_model, str)
        assert isinstance(self._batch_size, int)
        assert isinstance(self._epoch_num, int)
        assert isinstance(self._learning_rate, float)
        assert isinstance(self._momentum, float)
