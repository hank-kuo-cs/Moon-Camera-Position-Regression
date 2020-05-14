import numpy as np
from copy import deepcopy
from ...config import config


class RandomCameraGenerator:
    def __init__(self):
        self.dist, self.elev, self.azim = 0.0, 0.0, 0.0
        self.at = [0.0, 0.0, 0.0]
        self.up = [0.0, 1.0, 0.0]

    def get_random_camera(self):
        self.reset()

        self.set_eye()
        self.set_at()
        self.set_up()

        dist, elev, azim = self.dist, self.elev, self.azim
        at = deepcopy(self.at)
        up = deepcopy(self.up)

        return dist, elev, azim, at, up

    def reset(self):
        self.dist, self.elev, self.azim = 0.0, 0.0, 0.0
        self.at = [0.0, 0.0, 0.0]
        self.up = [0.0, 1.0, 0.0]

    @property
    def eye_cartesian_point(self) -> np.ndarray:
        return self.spherical_to_cartesian(self.dist, self.elev, self.azim)

    @property
    def at_vec(self) -> np.ndarray:
        eye = self.eye_cartesian_point
        at = np.array(self.at)
        return at-eye

    def set_eye(self):
        if not config.generate.is_change_eye:
            return

        dist_high = config.generate.dist_high_gl
        dist_low = config.generate.dist_low_gl

        self.dist = np.random.uniform(low=dist_low, high=dist_high)
        self.elev = self.get_random_elev()
        self.azim = self.get_random_azim()

    def set_at(self):
        if not config.generate.is_change_at:
            return

        at = np.random.normal(loc=0, scale=0.01, size=3)

        self.at = at.tolist()

    def set_up(self):
        while True:
            if not config.generate.is_change_up:
                up = np.random.uniform(0, 1, 3)
            else:
                up = deepcopy(self.up)

            at_vec = self.at_vec

            up = np.cross(at_vec, up)
            up = np.cross(up, at_vec)

            up_length = np.linalg.norm(up)
            if up_length != 0:
                up /= up_length
                break

        self.up = up.tolist()

    @staticmethod
    def get_random_elev():
        return np.pi * 0.5 - np.arccos(1 - 2 * np.random.uniform(0, 1))

    @staticmethod
    def get_random_azim():
        return np.random.uniform(0, 1) * 2 * np.pi

    @staticmethod
    def spherical_to_cartesian(dist, elev, azim) -> np.ndarray:
        x = dist * np.cos(elev) * np.sin(azim)
        y = dist * np.sin(elev)
        z = dist * np.cos(elev) * np.cos(azim)

        return np.array([x, y, z])
