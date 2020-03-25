import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, Textures
from config import DEVICE_NUM


def load_mesh(obj_path):
    device = torch.device('cuda:%s' % DEVICE_NUM)
    torch.cuda.set_device(device)
    vertices, faces, aux = load_obj(obj_path)

    vertices_uvs = aux.verts_uvs[None, ...].to(device)
    faces_uvs = faces.textures_idx[None, ...].to(device)

    texture_maps = aux.texture_images
    texture_maps = list(texture_maps.values())[0]
    texture_maps = texture_maps[None, ...].to(device)

    textures = Textures(verts_uvs=vertices_uvs,
                        faces_uvs=faces_uvs,
                        maps=texture_maps,)

    vertices = vertices.to(device)
    faces = faces.verts_idx.to(device)

    mesh = Meshes(verts=[vertices],
                  faces=[faces],
                  textures=textures)

    return mesh
