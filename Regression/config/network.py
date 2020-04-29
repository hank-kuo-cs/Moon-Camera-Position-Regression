class NetworkConfig:
    def __init__(self,
                 network_model: str,
                 batch_size: int = 20,
                 epoch_num: int = 300,
                 learning_rate: float = 0.001,
                 momentum: float = 0.9,
                 l_mse: float = 1.0,
                 l_image_comparison: float = 1.0,
                 l_mse_dist: float = 1.0,
                 l_mse_elev: float = 1.0,
                 l_mse_azim: float = 2.0,
                 l_mse_p: float = 1.0,
                 l_mse_u: float = 1.0):

        self._network_model = network_model
        self._epoch_num = epoch_num
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._l_mse = l_mse
        self._l_image_comparison = l_image_comparison
        self.l_mse_dist = l_mse_dist
        self.l_mse_elev = l_mse_elev
        self.l_mse_azim = l_mse_azim
        self.l_mse_p = l_mse_p
        self.l_mse_u = l_mse_u

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

    @property
    def l_mse(self):
        return self._l_mse

    @property
    def l_image_comparison(self):
        return self._l_image_comparison

    def check_parameters(self):
        assert isinstance(self._network_model, str)
        assert isinstance(self._batch_size, int)
        assert isinstance(self._epoch_num, int)
        assert isinstance(self._learning_rate, float)
        assert isinstance(self._momentum, float)
        assert isinstance(self._l_mse, float)
        assert isinstance(self._l_image_comparison, float)
