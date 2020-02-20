import numpy as np
from copy import deepcopy
from config import VIEW, GAMMA_RANGE, MOON_AVG_RADIUS_IN_GL_UNIT, GL_UNIT_TO_KM


class RandomViewGenerator:
    def __init__(self):
        self.gamma_range = [0, 0]  # OpenGL Unit
        self.view = deepcopy(VIEW)
        self.label = {'dist': 0.0, 'eye': [0.0, 0.0, 0.0], 'at': [0.0, 0.0, 0.0], 'up': [0.0, 0.0, 0.0]}
        self.set_gamma_range()

    def set_gamma_range(self):
        gamma_range_in_km = GAMMA_RANGE
        self.gamma_range = [MOON_RADIUS_IN_GL_UNIT, MOON_RADIUS_IN_GL_UNIT]

        for i in range(2):
            self.gamma_range[i] += gamma_range_in_km[i] / GL_UNIT_TO_KM

    def get_eye(self):
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        theta = self.get_random_theta()
        phi = self.get_random_phi()

        eye = [gamma, theta, phi]

        return eye

    def get_at(self):
        gamma = 0
        theta = self.get_random_theta()
        phi = self.get_random_phi()

        at = [gamma, theta, phi]

    def transform_view_to_xyz(self):
        self.view.eye = self.spherical_coordinate_to_xyz_coordinate(self.view.eye)
        self.view.at = self.spherical_coordinate_to_xyz_coordinate(self.view.at)

        self.calculate_up_vec()

    def calculate_up_vec(self):
        eye_vec = np.array(self.view.eye)
        at_vec = np.array(self.view.at)

        up_vec = np.cross(at_vec-eye_vec, self.view.up)
        up_vec = np.cross(up_vec, at_vec-eye_vec)

        self.view.up = self.normalize_vector(up_vec)

    @staticmethod
    def spherical_coordinate_to_xyz_coordinate(sc_vec):
        x = sc_vec[0] * np.sin(sc_vec[1]) * np.cos(sc_vec[2])
        y = sc_vec[0] * np.sin(sc_vec[1]) * np.sin(sc_vec[2])
        z = sc_vec[0] * np.cos(sc_vec[1])

        return [x, y, z]

    @staticmethod
    def get_vec_length(vec):
        length = np.linalg.norm(vec)

        return length if length > 0 else 1

    @staticmethod
    def normalize_vector(vec):
        length = ViewSetting.get_vec_length(vec)
        vec = vec / length

        return vec.tolist() if type(vec) == np.ndarray else vec





    @staticmethod
    def get_random_theta():
        return math.acos(1 - 2 * random.uniform(0, 1))

    @staticmethod
    def get_random_phi():
        return 2 * math.pi * random.uniform(0, 1)


    def get_label(self):
        pass

    def set_view(self, view: dict):
        for key, value in view.items():
            if key not in self.view:
                raise KeyError('View do not have key "%s"' % key)
            self.view[key] = value

    def set_eye(self, eye: list):
        if len(eye) != 3:
            raise ValueError('eye need to be 3 arguments')
        self.view['eye'] = eye

    def set_at(self, at: list):
        if len(at) != 3:
            raise ValueError('at need to be 3 arguments')
        self.view['at'] = at

    def set_up(self, up: list):
        if len(up) != 3:
            raise ValueError('up need to be 3 arguments')
        self.view['up'] = up

