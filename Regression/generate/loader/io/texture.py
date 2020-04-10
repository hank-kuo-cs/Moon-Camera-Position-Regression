import cv2
import numpy as np
from ...model import MoonTexture


class TextureEncoder:
    def __init__(self, texture_path):
        self.texture_path = texture_path
        self.texture_image = None
        self.moon_texture = MoonTexture()

    def load_texture(self) -> MoonTexture:
        self.load_4channel_image()
        self.convert_color_from_BGRA_to_RGBA()
        self.flip_column_value()

        self.set_height_and_width()
        self.set_texture_bytes()

        return self.moon_texture

    def load_4channel_image(self):
        self.texture_image = cv2.imread(self.texture_path, cv2.IMREAD_UNCHANGED)

    def convert_color_from_BGRA_to_RGBA(self):
        self.texture_image = cv2.cvtColor(self.texture_image, cv2.COLOR_BGRA2RGBA)

    def flip_column_value(self):
        self.texture_image = np.flip(self.texture_image, axis=0)

    def set_height_and_width(self):
        self.moon_texture.height = self.texture_image.shape[0]
        self.moon_texture.width = self.texture_image.shape[1]

    def set_texture_bytes(self):
        self.moon_texture.texture_bytes = self.texture_image.tostring()
