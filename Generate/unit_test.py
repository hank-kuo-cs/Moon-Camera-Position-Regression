import logging
from loader import load_object, load_texture, load_material, load_view, load_light
from model import Moon


def load_moon() -> Moon:
    moon_obj = load_object()
    moon_obj.check_parameters()

    moon_mtl = load_material()
    moon_mtl.check_parameters()

    moon_texture = load_texture()
    moon_texture.check_parameters()

    moon_light = load_light()
    moon_light.check_parameters()

    moon_view = load_view()
    moon_view.check_parameters()

    return Moon(obj=moon_obj,
                mtl=moon_mtl,
                texture=moon_texture,
                light=moon_light,
                view=moon_view)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    logging.info('Check loading data...')
    moon = load_moon()
    moon.check_parameters()
