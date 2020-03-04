import os
import torch


class CudaConfig:
    def __init__(self,
                 device: str = 'cuda',
                 is_parallel: bool = False,
                 cuda_device_number: int = 0,
                 parallel_gpus: list = None):

        self._device = device
        self._is_parallel = is_parallel
        self._cuda_device_number = cuda_device_number
        self._parallel_gpus = parallel_gpus
        self._cuda_num = torch.cuda.device_count()

        self.check_parameters()
        self.set_cuda_device()

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, new_device: str):
        assert new_device == 'cuda' or 'cpu'
        self._device = new_device

    @property
    def is_parallel(self) -> bool:
        return self._is_parallel

    @is_parallel.setter
    def is_parallel(self, new_is_parallel: bool):
        assert isinstance(new_is_parallel, bool)
        self._is_parallel = new_is_parallel

    @property
    def cuda_device_number(self):
        return self._cuda_device_number

    @cuda_device_number.setter
    def cuda_device_number(self, new_cuda_device_number: int):
        assert isinstance(new_cuda_device_number, int)
        assert new_cuda_device_number < self._cuda_num
        self._cuda_device_number = new_cuda_device_number

    @property
    def parallel_gpus(self):
        return self._parallel_gpus

    def set_cuda_device(self):
        if self.device == 'cuda':
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.cuda_device_number)

    def check_parameters(self):
        assert self.device == 'cuda' or 'cpu'
        assert isinstance(self.is_parallel, bool)
        assert isinstance(self.cuda_device_number, int)

        if self.device == 'cuda':
            assert torch.cuda.is_available()
