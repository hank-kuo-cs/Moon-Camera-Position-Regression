import numpy as np
from ..loader import load_view
from ..model import Cartesian3DPoint, Spherical3DPoint, MoonView, Cartesian3DVector
from ..config import GAMMA_RANGE, MOON_MAX_RADIUS_IN_GL_UNIT, KM_TO_GL_UNIT, IS_CHANGE_EYE, IS_CHANGE_UP, IS_CHANGE_AT


class RandomViewGenerator:
    def __init__(self):
        self.moon_view = load_view()
        self.spherical_eye = None

    def get_moon_view(self) -> MoonView:
        self.reset()

        self.set_eye()
        self.set_at()
        self.set_up()

        self.normalize_view()

        return self.moon_view

    def reset(self):
        self.moon_view = load_view()
        self.spherical_eye = None

    def set_eye(self):
        if IS_CHANGE_EYE:
            self.moon_view.eye = self.get_random_eye()

    def set_at(self):
        if IS_CHANGE_AT:
            self.moon_view.at = self.get_random_at()

    def set_up(self):
        if IS_CHANGE_UP:
            self.moon_view.up = self.get_random_up()

    def get_random_eye(self) -> Cartesian3DPoint:
        gamma_gl_range = [MOON_MAX_RADIUS_IN_GL_UNIT + GAMMA_RANGE[i] * KM_TO_GL_UNIT for i in range(2)]

        gamma = np.random.uniform(low=gamma_gl_range[0], high=gamma_gl_range[1])
        theta = self.get_random_theta()
        phi = self.get_random_phi()

        self.spherical_eye = Spherical3DPoint(gamma=gamma, theta=theta, phi=phi)
        eye = Cartesian3DPoint.from_spherical_point(self.spherical_eye)

        return eye

    def get_random_at(self) -> Cartesian3DPoint:
        gamma_gl_range = [0.0, 0.5 * MOON_MAX_RADIUS_IN_GL_UNIT]

        gamma = np.random.uniform(low=gamma_gl_range[0], high=gamma_gl_range[1])
        theta = self.get_random_theta()
        phi = self.get_random_phi()

        at = Spherical3DPoint(gamma=gamma, theta=theta, phi=phi)
        at = Cartesian3DPoint.from_spherical_point(at)

        return at

    def get_random_up(self) -> Cartesian3DVector:
        up_vec = np.random.uniform(0, 1, 3)
        up_vec = Cartesian3DVector.from_numpy(up_vec)

        return up_vec

    def normalize_view(self):
        self.normalize_at()
        self.normalize_up()

    def normalize_at(self):
        eye = self.moon_view.eye

        at_vec = self.moon_view.at - eye
        at_vec = at_vec.normalize()

        self.moon_view.at = eye + at_vec

    def normalize_up(self):
        up_vec = self.moon_view.up.to_numpy()
        at_vec = self.moon_view.at - self.moon_view.eye
        at_vec = at_vec.to_numpy()

        up_vec = np.cross(at_vec, up_vec)
        up_vec = np.cross(up_vec, at_vec)
        up_vec = self.normalize_vector(up_vec)
        up_vec = Cartesian3DPoint.from_list(up_vec)

        self.moon_view.up = up_vec

    @staticmethod
    def normalize_vector(vec: np.ndarray) -> list:
        assert isinstance(vec, np.ndarray)
        length = np.linalg.norm(vec)
        vec = vec / length if length > 0 else vec

        return vec.tolist() if type(vec) == np.ndarray else vec

    @staticmethod
    def get_random_theta() -> float:
        return np.arccos(1 - 2 * np.random.uniform(0, 1))

    @staticmethod
    def get_random_phi() -> float:
        return 2 * np.pi * np.random.uniform(0, 1)
