import cv2
import torch
import numpy as np
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, TexturedSoftPhongShader
from .mesh import load_mesh
from .scene import load_lights, load_cameras, load_rasterization_setting
from ....config import config


class Pytorch3DRenderer:
    def __init__(self):
        self.device = None
        self.mesh = None
        self.cameras = None
        self.raster_settings = None
        self.lights = None
        self.mesh_renderer = None

        self.initialize()

    def initialize(self):
        self.set_device()
        self.set_mesh()
        self.set_lights()
        self.set_raster_settings()

    def set_device(self):
        self.device = torch.device(config.cuda.device)

    def set_mesh(self):
        self.mesh = load_mesh()

    def set_cameras(self, dist, elev, azim, at, up):
        self.cameras = load_cameras(dist=dist, elev=elev, azim=azim, at=at, up=up)

    def set_raster_settings(self):
        self.raster_settings = load_rasterization_setting()

    def set_lights(self):
        self.lights = load_lights()

    def render_image(self) -> np.ndarray:
        if self.cameras is None:
            raise ValueError('cameras is None in pytorch3D renderer!')

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
        if config.cuda.device != 'cpu':
            image = image.cpu()
        img_data = image.numpy().squeeze() * 255
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        return img_data
