import numpy as np
from model import MoonView, Cartesian3DPoint, Spherical3DPoint


class LabelGenerator:
    def __init__(self, vertices: list):
        self.view = None
        self.spherical_eye = None
        self.vertices = np.array(vertices)
        self.label = {'dist': 0.0,
                      'c_theta': 0.0,
                      'c_phi': 0.0,
                      'p_xyz': [0.0, 0.0, 0.0],
                      'u_xyz': [0.0, 0.0, 0.0]}

    def set_view(self, view: MoonView, spherical_eye: Spherical3DPoint):
        self.view = view
        self.spherical_eye = spherical_eye

    def get_label(self):
        self.set_distance()
        self.set_c()
        self.set_p()
        self.set_u()

        return self.label

    def set_distance(self):
        eye = self.view.eye.to_numpy()
        dist_all = np.linalg.norm(self.vertices - eye, axis=1)
        self.label['dist'] = np.min(dist_all)

    def set_c(self):
        self.label['c_theta'] = self.spherical_eye.theta
        self.label['c_phi'] = self.spherical_eye.phi

    def set_p(self):
        self.label['p_xyz'] = self.view.at.to_list()

    def set_u(self):
        self.label['u_xyz'] = self.view.up.to_list()
