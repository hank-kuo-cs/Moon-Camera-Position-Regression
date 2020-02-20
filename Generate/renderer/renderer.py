import cv2
import pygame
import numpy as np
from config import WINDOW_WIDTH, WINDOW_HEIGHT
from model.moon import Moon
from renderer.scene import LightSetting, ViewSetting
from renderer.material import MoonSetting, TextureSetting
from OpenGL.GL import glClear, glCallList, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT


class Renderer:
    def __init__(self, moon: Moon):
        self.moon = moon
        self.polygon_list_id = None
        self.surface = None
        self.setting_state = {'window': False, 'light': False, 'view': False, 'texture': False, 'moon': False}

    def set_window(self):
        self.setting_state['window'] = True

        pygame.init()
        self.surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)

    def set_light(self):
        self.check_window_state()
        self.setting_state['light'] = True

        LightSetting(self.moon.light).set_light()

    def set_view(self):
        self.check_window_state()
        self.setting_state['view'] = True

        ViewSetting(self.moon.view).set_view()

    def set_texture(self):
        self.check_window_state()
        self.setting_state['texture'] = True

        TextureSetting(self.moon.texture).set_texture()

    def set_moon(self):
        self.check_window_state()
        self.setting_state['moon'] = True

        self.polygon_list_id = MoonSetting(self.moon).set_moon()

    def render_moon(self):
        self.check_setting_state()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glCallList(self.polygon_list_id)

    def save_image(self, image_path):
        pygame.image.save(self.surface, image_path)

    def get_image(self):
        image_bytes = pygame.image.tostring(self.surface, 'RGBA')
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = np.reshape(image, (WINDOW_HEIGHT, WINDOW_WIDTH, 4))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        return image

    def check_setting_state(self):
        for key, value in self.setting_state.items():
            if not value:
                raise ValueError('Not yet set up %d' % key)

    def check_window_state(self):
        if not self.setting_state['window']:
            raise ValueError('Window need to be set up first')
