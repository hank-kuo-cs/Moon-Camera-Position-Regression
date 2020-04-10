from ....model import MoonLight
from .lib import *


class LightSetting:
    def __init__(self, light: MoonLight):
        self.light = light

    def set_light(self):
        self.enable_light()
        self.set_position_light()
        self.set_ambient_light()
        self.set_diffuse_light()

    def set_position_light(self):
        glLightfv(GL_LIGHT0, GL_POSITION, self.light.position_light)

    def set_ambient_light(self):
        glLightfv(GL_LIGHT0, GL_AMBIENT, self.light.ambient_light)

    def set_diffuse_light(self):
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.light.diffuse_light)

    @staticmethod
    def enable_light():
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
