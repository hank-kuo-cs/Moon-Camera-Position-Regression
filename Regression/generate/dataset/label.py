import numpy as np
from copy import deepcopy
from ...config import config


class LabelGenerator:
    def __init__(self):
        self.dist = None
        self.elev = None
        self.azim = None
        self.at = None
        self.up = None

        self.label = {'dist': 0.0,      # gl unit
                      'elev': 0.0,      # rad
                      'azim': 0.0,      # rad
                      'p_x': 0.0,       # gl unit
                      'p_y': 0.0,       # gl unit
                      'p_z': 0.0,       # gl unit
                      'u_x': 0.0,       # gl unit
                      'u_y': 0.0,       # gl unit
                      'u_z': 0.0}       # gl unit

    def set_view(self, dist: float, elev: float, azim: float, at: list, up: list):
        self.dist = dist
        self.elev = elev
        self.azim = azim
        self.at = at
        self.up = up

        self.check_parameters()

    def get_label(self):
        self.label['dist'] = self.dist

        self.set_eye()
        self.set_p()
        self.set_u()

        label = deepcopy(self.label)

        return label

    def set_eye(self):
        self.label['dist'] = self.dist
        self.label['elev'] = self.elev
        self.label['azim'] = self.azim

    def set_p(self):
        self.label['p_x'] = self.at[0]
        self.label['p_y'] = self.at[1]
        self.label['p_z'] = self.at[2]

    def set_u(self):
        self.label['u_x'] = self.up[0]
        self.label['u_y'] = self.up[1]
        self.label['u_z'] = self.up[2]

    @staticmethod
    def get_project_vector_length(v1, v2):
        assert isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)
        assert v1.shape == (3,) and v2.shape == (3,)

        return np.linalg.norm(np.dot(v1, v2) / np.linalg.norm(v2) * v2 / np.linalg.norm(v2))

    def check_parameters(self):
        assert isinstance(self.dist, float)
        assert isinstance(self.elev, float)
        assert isinstance(self.azim, float)
        assert isinstance(self.at, list)
        assert isinstance(self.up, list)

        dist_high_bound = config.generate.moon_radius_gl + config.generate.dist_between_moon_high_bound_km * config.generate.km_to_gl

        assert self.dist <= dist_high_bound
        assert -0.5 * np.pi <= self.elev <= 0.5 * np.pi
        assert 0 <= self.azim <= np.pi * 2
