import numpy as np
from loader.model import MoonView
from renderer.scene.lib import *


class ViewSetting:
    def __init__(self, view: MoonView):
        self.view = view

    def set_view(self):
        self.set_projection()
        self.set_model_view()

    def set_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        width = self.view.viewport[0]
        height = self.view.viewport[1]

        gluPerspective(self.view.fov, float(width / height), self.view.d_near, self.view.d_far)

    def set_model_view(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        self.transform_view_to_xyz()

        gluLookAt(*self.view.eye, *self.view.at, *self.view.up)

    def transform_view_to_xyz(self):
        self.view.eye = self.spherical_coordinate_to_xyz_coordinate(self.view.eye)
        self.view.at = self.spherical_coordinate_to_xyz_coordinate(self.view.at)

        self.calculate_up_vec()

    def calculate_up_vec(self):
        eye_vec = np.array(self.view.eye)
        at_vec = np.array(self.view.at)

        up_vec = np.cross(at_vec-eye_vec, self.view.up)
        up_vec = np.cross(up_vec, at_vec-eye_vec)
        length = np.linalg.norm(up_vec)
        up_vec = up_vec / length if length > 0 else up_vec
        up_vec = up_vec.tolist()

        self.view.up = up_vec

    @staticmethod
    def spherical_coordinate_to_xyz_coordinate(sc_vec):
        x = sc_vec[0] * np.sin(sc_vec[1]) * np.cos(sc_vec[2])
        y = sc_vec[0] * np.sin(sc_vec[1]) * np.sin(sc_vec[2])
        z = sc_vec[0] * np.cos(sc_vec[1])

        return [x, y, z]
