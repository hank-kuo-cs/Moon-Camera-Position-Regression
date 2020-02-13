from renderer.window.lib import *
from renderer.window.gl_buffer import GLBufferSetting
from config import WINDOW_NAME, WINDOW_POSITION, WINDOW_HEIGHT, WINDOW_WIDTH


class GLUTWindowSetting:
    def __init__(self):
        self.display_window = DisplayWindow(name=WINDOW_NAME,
                                            width=WINDOW_WIDTH,
                                            height=WINDOW_HEIGHT,
                                            position=WINDOW_POSITION)

    def set_display_window(self):
        self.initialize_glut_window()
        self.create_glut_hidden_window()

        GLBufferSetting.initialize_gl_buffer()

    def initialize_glut_window(self):
        glutInit()
        glutInitWindowSize(self.display_window.width, self.display_window.height)
        glutInitWindowPosition(self.display_window.position[0], self.display_window.position[1])
        glutInitDisplayMode(GLUT_RGBA)

    def create_glut_hidden_window(self):
        glutCreateWindow(self.display_window.name)
        # glutHideWindow()


class DisplayWindow:
    def __init__(self, name: str, width: int, height: int, position: list):
        self.name = name
        self.width = width
        self.height = height
        self.position = position

    def check_parameters(self):
        assert type(self.name) == str
        assert type(self.width) == int
        assert type(self.height) == int
        assert type(self.position) == list
        assert len(self.position) == 2
        assert type(self.position[0]) == str and type(self.position[1]) == str
