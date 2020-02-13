import cv2
import numpy as np
from OpenGL.GL import glReadPixels, GL_BGR, GL_UNSIGNED_BYTE
from config import WINDOW_WIDTH, WINDOW_HEIGHT


class ImageDecoder:
    def __init__(self, save_path: str):
        self.save_path = save_path
        self.image = np.array([])

    def get_image_from_gl_buffer(self):
        pixels = glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE)
        self.image = np.frombuffer(pixels, dtype=np.uint8)
        self.reshape_image()

    def reshape_image(self):
        self.image = self.image.reshape((WINDOW_HEIGHT, WINDOW_WIDTH, 3))
        self.image = np.flip(self.image, axis=0)

    def save_image(self):
        cv2.imwrite(self.save_path, self.image)