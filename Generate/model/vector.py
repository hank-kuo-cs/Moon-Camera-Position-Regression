import numpy as np
from model.point import Spherical3DPoint, Cardassian3DPoint


class Cardassian3DVector:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._vec = np.array([x, y, z], dtype=np.float)

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

    def normalize(self):
        if not self.length:
            raise ValueError('Vector length is 0, cannot be normalized!')

        self._vec = self._vec / self.length
        return Cardassian3DVector(x=self.x, y=self.y, z=self.z)

    def check_parameters(self):
        assert isinstance(self._vec, np.ndarray)
        assert isinstance(self.x, float)
        assert isinstance(self.y, float)
        assert isinstance(self.z, float)
