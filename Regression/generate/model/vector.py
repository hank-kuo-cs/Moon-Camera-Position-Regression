import numpy as np


class Cartesian3DVector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._vec = np.array([x, y, z], dtype=np.float)

    def __neg__(self):
        self._vec = -self._vec

    def __repr__(self):
        return str(self._vec)

    @property
    def x(self):
        return self._vec[0]

    @x.setter
    def x(self, new_x):
        assert isinstance(new_x, float)
        self._vec[0] = new_x

    @property
    def y(self):
        return self._vec[1]

    @y.setter
    def y(self, new_y):
        assert isinstance(new_y, float)
        self._vec[1] = new_y

    @property
    def z(self):
        return self._vec[2]

    @z.setter
    def z(self, new_z):
        assert isinstance(new_z, float)
        self._vec[2] = new_z

    @property
    def length(self):
        return np.linalg.norm(self._vec)

    def to_list(self):
        return self._vec.tolist()

    def to_numpy(self):
        return self._vec

    @staticmethod
    def from_numpy(np_vec):
        assert len(np_vec) == 3

        return Cartesian3DVector(x=np_vec[0], y=np_vec[1], z=np_vec[2])

    def normalize(self):
        if self.length:
            self._vec = self._vec / self.length

        return Cartesian3DVector(x=self.x, y=self.y, z=self.z)

    def check_parameters(self):
        assert isinstance(self._vec, np.ndarray)
        assert isinstance(self.x, float)
        assert isinstance(self.y, float)
        assert isinstance(self.z, float)
