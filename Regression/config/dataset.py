class DatasetConfig:
    def __init__(self,
                 dataset_path: str,
                 labels: list,
                 dataset_size: dict,
                 sub_dataset_size: int,
                 dist_range: float,
                 normalize_point_weight: float):

        self._dataset_path = dataset_path
        self._labels = labels
        self._dataset_size = dataset_size
        self._sub_dataset_size = sub_dataset_size
        self._dataset_types = ['train', 'test', 'validation']
        self._dist_range = dist_range   # km
        self._normalize_point_weight = normalize_point_weight

        self.check_parameters()

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

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

    @property
    def dataset_types(self):
        return self._dataset_types

    @property
    def dist_range(self) -> float:
        return self._dist_range

    @property
    def labels(self) -> list:
        return self._labels

    @property
    def normalize_point_weight(self) -> float:
        return self._normalize_point_weight

    def get_dataset_num(self, dataset_type=None) -> int:
        if dataset_type is None:
            return self.dataset_num

        assert dataset_type in self._dataset_types
        return self._dataset_size[dataset_type]

    def check_parameters(self):
        assert isinstance(self._dataset_path, str)
        assert isinstance(self._labels, list)
        assert isinstance(self._dataset_size, dict)
        assert isinstance(self._sub_dataset_size, int)
        assert isinstance(self._dist_range, float)
        assert isinstance(self._normalize_point_weight, float)

        labels = ['dist', 'c_theta', 'c_phi', 'p_x', 'p_y', 'p_z', 'u_x', 'u_y', 'u_z']

        for l in self._labels:
            assert l in labels

        for k, v in self._dataset_size.items():
            assert k in self._dataset_types
            assert v >= self._sub_dataset_size
            assert v % self._sub_dataset_size == 0
