import cv2
import torch
import numpy as np
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, TexturedSoftPhongShader, SoftSilhouetteShader, BlendParams
from .mesh import load_mesh
from .scene import load_lights, load_perspective_cameras, load_rasterization_setting, load_camera_positions
from ....config import config


class Pytorch3DRenderer:
    def __init__(self):
        self.device = None
        self.mesh = None
        self.cameras = None
        self.raster_settings = None
        self.lights = None
        self.mesh_renderer = None
        self.R = None
        self.T = None

        self.initialize()

    def initialize(self):
        self._set_device()
        self._set_mesh()
        self._set_perspective_cameras()
        self._set_lights()
        self._set_raster_settings()
        self._set_renderer()

    def _set_device(self):
        self.device = torch.device(config.cuda.device)

    def _set_mesh(self):
        self.mesh = load_mesh()

    def _set_perspective_cameras(self):
        self.cameras = load_perspective_cameras()

    def _set_raster_settings(self):
        self.raster_settings = load_rasterization_setting()

    def _set_lights(self):
        self.lights = load_lights()

    def set_cameras(self, dist, elev, azim, at, up, is_degree=False):
        if is_degree:
            elev *= (np.pi / 180)
            azim *= (np.pi / 180)

        R, T = load_camera_positions(dist, elev, azim, at, up)
        self.R = R
        self.T = T

    def _set_renderer(self):
        if self.cameras is None:
            raise ValueError('cameras is None in pytorch3D renderer!')

        rasterizer = MeshRasterizer(cameras=self.cameras,
                                    raster_settings=self.raster_settings)

        texture_shader = TexturedSoftPhongShader(device=self.device,
                                                 cameras=self.cameras,
                                                 lights=self.lights)

        silhouette_shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=1e-4, gamma=1e-4))

        self.mesh_renderer = MeshRenderer(rasterizer=rasterizer,
                                          shader=texture_shader)

        self.mask_renderer = MeshRenderer(rasterizer=rasterizer,
                                          shader=silhouette_shader)

    def render_image(self) -> np.ndarray:
        assert self.R is not None and self.T is not None

        image = self.mesh_renderer(self.mesh.clone(), R=self.R, T=self.T)
        return image

    def render_mask_image(self) -> np.ndarray:
        assert self.R is not None and self.T is not None

        image = self.mask_renderer(self.mesh.clone(), R=self.R, T=self.T)
        return image

    @staticmethod
    def refine_image_to_data(image) -> np.ndarray:
        if config.cuda.device != 'cpu':
            image = image.cpu()
        img_data = image.numpy().squeeze() * 255
        img_data = img_data.astype(np.uint8)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        return img_data

    @staticmethod
    def refine_mask(mask_image) -> np.ndarray:
        if config.cuda.device != 'cpu':
            mask_image = mask_image.cpu()
        img_data = mask_image.numpy().squeeze()[..., 3] * 255
        img_data[img_data > 0] = 255
        img_data = img_data.astype(np.uint8)
        return img_data
