import numpy as np
from model import MoonView, Spherical3DPoint
from config import GL_UNIT_TO_KM


class LabelGenerator:
    def __init__(self, vertices: list):
        self.view = None
        self.spherical_eye = None
        self.vertices = np.array(vertices)
        self.label = {'dist': 0.0,      # km
                      'c_theta': 0.0,   # rad
                      'c_phi': 0.0,     # rad
                      'p_xyz': [0.0, 0.0, 0.0],     # km
                      'u_xyz': [0.0, 0.0, 0.0]}     # km

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

        dist_with_vertices = np.linalg.norm(self.vertices - eye, axis=1)
        nearest_vertex_idx = np.argmin(dist_with_vertices)

        nearest_vertex = self.vertices[nearest_vertex_idx]

        dist_with_moon_surface = np.dot(nearest_vertex - eye, -eye) / np.linalg.norm(eye)
        self.label['dist'] = dist_with_moon_surface * GL_UNIT_TO_KM

    def set_c(self):
        self.label['c_theta'] = self.spherical_eye.theta
        self.label['c_phi'] = self.spherical_eye.phi

    def set_p(self):
        self.label['p_xyz'] = self.view.at.to_list() * GL_UNIT_TO_KM

    def set_u(self):
        self.label['u_xyz'] = self.view.up.to_list() * GL_UNIT_TO_KM
