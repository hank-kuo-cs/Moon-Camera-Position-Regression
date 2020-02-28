class DatasetConfig:
    def __init__(self,
                 dataset_path: str,
                 labels: list,
                 dataset_size: dict,
                 sub_dataset_size: int):

        self._dataset_path = dataset_path
        self._labels = labels
        self._dataset_size = dataset_size
        self._sub_dataset_size = sub_dataset_size
        self._dataset_type = ['train', 'test', 'validation']

        self.check_parameters()

    @property
    def train_dataset_num(self) -> int:
        return self._dataset_size['train']

    @property
    def test_dataset_num(self) -> int:
        return self._dataset_size['test']

    @property
    def validation_dataset_num(self) -> int:
        return self._dataset_size['validation']

    @property
    def dataset_num(self) -> int:
        return self.train_dataset_num + self.test_dataset_num + self.validation_dataset_num

    def check_parameters(self):
        assert isinstance(self._dataset_path, str)
        assert isinstance(self._labels, list)
        assert isinstance(self._dataset_size, dict)
        assert isinstance(self._sub_dataset_size, int)

        labels = ['dist', 'c_theta', 'c_phi', 'p_xyz', 'u_xyz']

        for l in self._labels:
            assert l in labels

            if l == 'p_xyz' or l == 'u_xyz':
                assert isinstance(self._labels[l], list)
                assert len(self._labels[l]) == 3

        for k, v in self._dataset_size.items():
            assert k in self._dataset_type
            assert v >= self._sub_dataset_size
            assert v % self._sub_dataset_size == 0
