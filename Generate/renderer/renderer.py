from config import *
from loader.moon import Moon
from renderer.window import GLUTWindowSetting
from renderer.scene import LightSetting, ViewSetting
from renderer.material import MoonSetting, TextureSetting
from renderer.io import ImageDecoder
from OpenGL.GL import glClear, glCallList, glColor, glFlush, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
from OpenGL.GLUT import glutDisplayFunc, glutSwapBuffers, glutMainLoop, glutWireTeapot


list_id = 0


class Renderer:
    def __init__(self, moon: Moon):
        self.moon = moon
        logging.info('Declare glut window')
        self.window_setting = GLUTWindowSetting()
        logging.info('Declare light')
        self.light_setting = LightSetting(self.moon.light)
        logging.info('Declare view')
        self.view_setting = ViewSetting(self.moon.view)
        logging.info('Declare texture')
        self.texture_setting = TextureSetting(self.moon.texture)
        logging.info('Declare moon')
        self.moon_setting = MoonSetting(self.moon)

    def render_moon(self):
        logging.info('set window')
        self.set_window()
        logging.info('set light')
        self.set_light()
        logging.info('set view')
        self.set_view()
        logging.info('set texture')
        self.set_texture()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glColor(120.0, 120.0, 120.0)
        logging.info('set moon')
        self.set_moon()
        logging.info('draw moon')
        self.draw_moon()

    def set_window(self):
        self.window_setting.set_display_window()

    def set_light(self):
        self.light_setting.set_light()

    def set_view(self):
        self.view_setting.set_view()

    def set_texture(self):
        self.texture_setting.set_texture()

    def set_moon(self):
        self.moon_setting.set_moon()

    def draw_moon(self):
        global list_id
        list_id = self.moon_setting.polygon_list_id

        glutDisplayFunc(display)
        glutMainLoop()



    @staticmethod
    def export_image(image_path):
        image_decoder = ImageDecoder(image_path)
        image_decoder.get_image_from_gl_buffer()
        image_decoder.save_image()


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glColor(120.0, 120.0, 120.0)
    # glutWireTeapot(0.6)
    # self.light_setting.set_light()
    # self.view_setting.set_view()
    global list_id
    glCallList(list_id)
    glutSwapBuffers()