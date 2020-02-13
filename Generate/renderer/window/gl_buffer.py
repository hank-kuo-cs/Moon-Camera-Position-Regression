from renderer.window.lib import *


class GLBufferSetting:
    @classmethod
    def initialize_gl_buffer(cls):
        cls.remove_object_behind_scene()
        cls.smooth_object()
        cls.enable_material()

    @staticmethod
    def remove_object_behind_scene():
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)

    @staticmethod
    def enable_material():
        glEnable(GL_COLOR_MATERIAL)

    @staticmethod
    def smooth_object():
        glShadeModel(GL_SMOOTH)
