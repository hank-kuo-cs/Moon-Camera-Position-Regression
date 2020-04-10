import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Textures
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    DirectionalLights, TexturedSoftPhongShader
)
from ..generate.config import OBJECT_PATH, VIEW
from ..config import config


class DifferentialRenderer:
    def __init__(self):
        self.cuda_device = config.cuda.device
        self.moon_mesh = None
        self.renderer = None

        self.set_moon_mesh()
        self.set_renderer()

    def render_image(self, cameras):
        # x, y, z = self.transform_spherical_to_cartesian(cameras[0], cameras[1], cameras[2])

        # R, T = look_at_view_transform(eye=((x, y, z),),
        #                               at=((cameras[3], cameras[4], cameras[5]),),
        #                               up=((cameras[6], cameras[7], cameras[8]),),
        #                               device=self.cuda_device)
        R, T = look_at_view_transform(dist=cameras[0],
                                      elev=cameras[1],
                                      azim=cameras[2],
                                      at=((cameras[3], cameras[4], cameras[5]),),
                                      up=((cameras[6], cameras[7], cameras[8]),),
                                      device=self.cuda_device)

        img = self.renderer(meshes_world=self.moon_mesh.clone(), R=R, T=T)
        return img

    @staticmethod
    def transform_spherical_to_cartesian(gamma, theta, phi):
        x = gamma * torch.sin(theta) * torch.cos(phi)
        y = gamma * torch.sin(theta) * torch.sin(phi)
        z = gamma * torch.cos(theta)

        return x, y, z

    def set_moon_mesh(self):
        vertices, faces, aux = load_obj(OBJECT_PATH)
        faces_idx = faces.verts_idx.to(self.cuda_device)
        vertices = vertices.to(self.cuda_device)

        verts_uvs = aux.verts_uvs[None, ...].to(self.cuda_device)
        faces_uvs = faces.textures_idx[None, ...].to(self.cuda_device)
        tex_maps = aux.texture_images
        texture_image = list(tex_maps.values())[0]
        texture_image = texture_image[None, ...].to(self.cuda_device)

        tex = Textures(verts_uvs=verts_uvs,
                       faces_uvs=faces_uvs,
                       maps=texture_image)

        self.moon_mesh = Meshes(verts=[vertices],
                                faces=[faces_idx],
                                textures=tex)

    def set_renderer(self):
        cameras = OpenGLPerspectiveCameras(device=self.cuda_device,
                                           degrees=True,
                                           fov=VIEW['fov'],
                                           znear=VIEW['znear'],
                                           zfar=VIEW['zfar'])

        raster_settings = RasterizationSettings(image_size=VIEW['viewport'][0],
                                                blur_radius=0.0,
                                                faces_per_pixel=1,
                                                bin_size=0)

        lights = DirectionalLights(device=self.cuda_device,
                                   direction=((-40, 200, 100),),
                                   ambient_color=((0.5, 0.5, 0.5),),
                                   diffuse_color=((0.5, 0.5, 0.5),),
                                   specular_color=((0.0, 0.0, 0.0),), )

        self.renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras,
                                                               raster_settings=raster_settings),
                                     shader=TexturedSoftPhongShader(device=self.cuda_device,
                                                                    cameras=cameras,
                                                                    lights=lights))
