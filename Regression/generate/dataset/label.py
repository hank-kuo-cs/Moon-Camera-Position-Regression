import numpy as np
from copy import deepcopy
from ..model import MoonView, Spherical3DPoint
from ..config import GL_UNIT_TO_KM, MOON_MAX_RADIUS_IN_GL_UNIT


class LabelGenerator:
    def __init__(self, vertices: list):
        self.view = None
        self.spherical_eye = None
        self.vertices = np.array(vertices)

        self.label = {'dist': 0.0,      # km
                      'c_theta': 0.0,   # rad
                      'c_phi': 0.0,     # rad
                      'p_x': 0.0,       # km
                      'p_y': 0.0,       # km
                      'p_z': 0.0,       # km
                      'u_x': 0.0,       # km
                      'u_y': 0.0,       # km
                      'u_z': 0.0}       # km

    def set_view(self, view: MoonView, spherical_eye: Spherical3DPoint):
        self.view = view
        self.spherical_eye = spherical_eye

    def get_label(self):
        self.set_distance()
        self.set_c()
        self.set_p()
        self.set_u()

        label = deepcopy(self.label)

        return label

    def set_distance(self):
        dist = (self.spherical_eye.gamma - MOON_MAX_RADIUS_IN_GL_UNIT) * GL_UNIT_TO_KM
        self.label['dist'] = dist

        return self.label

    def set_c(self):
        self.label['c_theta'] = self.spherical_eye.theta
        self.label['c_phi'] = self.spherical_eye.phi

    def set_p(self):
        p_xyz = (self.view.at * GL_UNIT_TO_KM).to_list()
        self.label['p_x'] = p_xyz[0]
        self.label['p_y'] = p_xyz[1]
        self.label['p_z'] = p_xyz[2]

    def set_u(self):
        u_xyz = (self.view.up * GL_UNIT_TO_KM).to_list()
        self.label['u_x'] = u_xyz[0]
        self.label['u_y'] = u_xyz[1]
        self.label['u_z'] = u_xyz[2]

    @staticmethod
    def get_project_vector_length(v1, v2):
        assert isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)
        assert v1.shape == (3,) and v2.shape == (3,)

        return np.linalg.norm(np.dot(v1, v2) / np.linalg.norm(v2) * v2 / np.linalg.norm(v2))
