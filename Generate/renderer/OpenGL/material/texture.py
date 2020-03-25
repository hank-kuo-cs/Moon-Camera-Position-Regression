from model import MoonTexture
from renderer.OpenGL.material.lib import *


class TextureSetting:
    def __init__(self, texture: MoonTexture):
        self.texture = texture

    def set_texture(self):
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_TEXTURE_2D)
        self.texture.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture.texture_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        glTexImage2D(GL_TEXTURE_2D,
                     0,
                     GL_RGBA,
                     self.texture.width,
                     self.texture.height,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     self.texture.texture_bytes)
