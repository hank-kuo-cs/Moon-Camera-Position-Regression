from model import MoonView, Cardassian3DPoint
from renderer.scene.lib import *


class ViewSetting:
    def __init__(self, view: MoonView):
        self.view = view
        self.check_point_type()

    def set_view(self):
        self.set_projection()
        self.set_model_view()

    def set_projection(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        width = self.view.viewport[0]
        height = self.view.viewport[1]

        gluPerspective(self.view.fov, float(width / height), self.view.znear, self.view.zfar)

    def set_model_view(self):
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        eye = self.view.eye.to_list()
        at = self.view.at.to_list()
        up = self.view.up.to_list()

        gluLookAt(*eye, *at, *up)

    def check_point_type(self):
        assert isinstance(self.view.eye, Cardassian3DPoint)
        assert isinstance(self.view.at, Cardassian3DPoint)
        assert isinstance(self.view.up, Cardassian3DPoint)
