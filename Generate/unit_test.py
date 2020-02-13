from loader.io import ObjectEncoder, MaterialEncoder, TextureEncoder
from config import *


def load_obj():
    logging.info('Load moon object')
    object_encoder = ObjectEncoder(OBJECT_PATH)
    moon_obj = object_encoder.load_object()
    moon_obj.check_parameters()


def load_mtl():
    logging.info('Load moon material')
    material_encoder = MaterialEncoder(MATERIAL_PATH)
    moon_material = material_encoder.load_material()
    moon_material.check_parameters()


def load_texture():
    logging.info('Load moon texture')
    texture_encoder = TextureEncoder(TEXTURE_PATH)
    moon_texture = texture_encoder.load_texture()
    moon_texture.check_parameters()


def load_moon():
    load_obj()
    load_mtl()
    load_texture()


if __name__ == '__main__':
    logging.info('Check loading data...')
    load_moon()
    logging.info('Load data success')

    print()
