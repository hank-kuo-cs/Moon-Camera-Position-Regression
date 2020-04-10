import os
import cv2
import torch
import numpy as np
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, TexturedSoftPhongShader
from ...model import Moon
from .mesh import load_mesh
from .scene import load_lights, load_cameras, load_rasterization_setting
from ...config import OBJECT_PATH


class Pytorch3DRenderer:
    def __init__(self, moon: Moon):
        self.moon = moon
        self.device = None
        self.mesh = None
        self.cameras = None
        self.raster_settings = None
        self.lights = None
        self.mesh_renderer = None

    def set_device(self):
        self.device = torch.device('cuda')

    def set_mesh(self):
        if not os.path.exists(OBJECT_PATH):
            raise FileNotFoundError('Cannot find moon object from \'%s\'' % OBJECT_PATH)

        self.mesh = load_mesh(obj_path=OBJECT_PATH)

    def set_cameras(self, moon_view=None):
        if moon_view:
            self.moon.view = moon_view
        self.cameras = load_cameras(self.moon.view)

    def set_raster_settings(self):
        self.raster_settings = load_rasterization_setting()

    def set_lights(self, moon_light=None):
        if moon_light:
            self.moon.light = moon_light
        self.lights = load_lights(self.moon.light)

    def render_image(self) -> np.ndarray:
        rasterizer = MeshRasterizer(cameras=self.cameras,
                                    raster_settings=self.raster_settings)

        shader = TexturedSoftPhongShader(device=self.device,
                                         cameras=self.cameras,
                                         lights=self.lights)

        self.mesh_renderer = MeshRenderer(rasterizer=rasterizer,
                                          shader=shader)

        image = self.mesh_renderer(self.mesh)
        return image

    @staticmethod
    def refine_image_to_data(image) -> np.ndarray:
        img_data = image.cpu().numpy().squeeze() * 255
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2GRAY)
        return img_data
