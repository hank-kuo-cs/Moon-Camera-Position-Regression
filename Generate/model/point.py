import numpy as np


class Spherical3DPoint:
    def __init__(self, gamma: float = 0.0, theta: float = 0.0, phi: float = 0.0):
        self._point = np.array([gamma, theta, phi], dtype=np.float)

    @property
    def gamma(self):
        return self._point[0]

    @gamma.setter
    def gamma(self, new_gamma):
        self._check_gamma(new_gamma)
        self._point[0] = new_gamma

    @property
    def theta(self):
        return self._point[1]

    @theta.setter
    def theta(self, new_theta):
        self._check_theta(new_theta)
        self._point[1] = new_theta

    @property
    def phi(self):
        return self._point[2]

    @phi.setter
    def phi(self, new_phi):
        self._check_phi(new_phi)
        self._point[2] = new_phi

    def to_list(self):
        return self._point.tolist()

    @staticmethod
    def from_list(point_list):
        return Spherical3DPoint(gamma=point_list[0], theta=point_list[1], phi=point_list[2])

    def check_parameters(self):
        self._check_gamma(self.gamma)
        self._check_theta(self.theta)
        self._check_phi(self.phi)

    @staticmethod
    def _check_gamma(gamma):
        assert isinstance(gamma, float)
        assert 0 <= gamma

    @staticmethod
    def _check_theta(theta):
        assert isinstance(theta, float)
        assert 0 <= theta <= np.pi * 2

    @staticmethod
    def _check_phi(phi):
        assert isinstance(phi, float)
        assert 0 <= phi <= np.pi


class Cardassian3DPoint:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self._point = np.array([x, y, z], dtype=np.float)

    @property
    def x(self):
        return self._point[0]

    @x.setter
    def x(self, new_x):
        self._check_x(new_x)
        self._point[0] = new_x

    @property
    def y(self):
        return self._point[1]

    @y.setter
    def y(self, new_y):
        self._check_y(new_y)
        self._point[1] = new_y

    @property
    def z(self):
        return self._point[2]

    @z.setter
    def z(self, new_z):
        self._check_z(new_z)
        self._point[2] = new_z

    @property
    def length_with_origin(self):
        return np.linalg.norm(self._point)

    @staticmethod
    def from_spherical_point(sph_point: Spherical3DPoint):
        x = sph_point.gamma * np.sin(sph_point.theta) * np.cos(sph_point.phi)
        y = sph_point.gamma * np.sin(sph_point.theta) * np.sin(sph_point.phi)
        z = sph_point.gamma * np.cos(sph_point.theta)

        return Cardassian3DPoint(x=x, y=y, z=z)

    def to_list(self):
        return self._point.tolist()

    @staticmethod
    def from_list(point_list):
        return Cardassian3DPoint(x=point_list[0], y=point_list[1], z=point_list[2])

    def check_parameters(self):
        self._check_x(self.x)
        self._check_y(self.y)
        self._check_z(self.z)

    @staticmethod
    def _check_x(x):
        assert isinstance(x, float)

    @staticmethod
    def _check_y(y):
        assert isinstance(y, float)

    @staticmethod
    def _check_z(z):
        assert isinstance(z, float)
