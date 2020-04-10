from ..config import *
from ..model import Moon
from .io import ObjectEncoder, MaterialEncoder, TextureEncoder, LightEncoder, ViewEncoder


def load_moon() -> Moon:
    moon_obj = load_object()
    moon_mtl = load_material()
    moon_texture = load_texture()
    moon_light = load_light()
    moon_view = load_view()

    return Moon(obj=moon_obj,
                mtl=moon_mtl,
                texture=moon_texture,
                light=moon_light,
                view=moon_view)


def load_object():
    if not os.path.exists(OBJECT_PATH):
        raise FileNotFoundError('Cannot find moon object from \'%s\'' % OBJECT_PATH)

    object_encoder = ObjectEncoder(OBJECT_PATH)
    return object_encoder.load_object()


def load_material():
    material_encoder = MaterialEncoder(MATERIAL_PATH)
    return material_encoder.load_material()


def load_texture():
    texture_encoder = TextureEncoder(TEXTURE_PATH)
    return texture_encoder.load_texture()


def load_light():
    light_encoder = LightEncoder()
    return light_encoder.load_light()


def load_view():
    view_encoder = ViewEncoder()
    return view_encoder.load_view()
