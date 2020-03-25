import logging
import numpy as np
from model import Cartesian3DPoint, Cartesian3DVector, MoonView
from loader import load_moon
from renderer import Renderer


def generate_one_image():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    logging.info('Load moon file...')
    moon = load_moon()

    logging.info('Set up renderer...')
    renderer = Renderer(moon=moon)

    vertices = moon.obj.vertices

    renderer.set_window()
    renderer.set_texture()
    renderer.set_moon()
    renderer.set_light()
    renderer.set_view()

    logging.info('Render image...')
    renderer.render_moon()
    renderer.save_image('test.png')


def config_experiment():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    logging.info('Load moon file...')
    moon = load_moon()

    moon_view = moon.view

    logging.info('Set up renderer...')


    znear_list = [0.001, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in range(1, 30):
        gamma = 0 + 0.1 * i
        moon_view.eye.z = -float(gamma)
        print(moon_view.eye)
        renderer = Renderer(moon=moon)

        renderer.set_window()
        renderer.set_texture()
        renderer.set_moon()
        renderer.set_light()
        renderer.set_view(moon_view)
        renderer.render_moon()
        renderer.save_image('fov120_znear1e-4_dist%dkm.png' % (int(1000 * gamma)))


def make_gif():
    pass


def normalize_up(moon_view: MoonView, up_vec: np.ndarray) -> Cartesian3DPoint:
    assert isinstance(up_vec, np.ndarray)
    at_vec = moon_view.at - moon_view.eye
    at_vec = at_vec.to_numpy()

    up_vec = np.cross(at_vec, up_vec)
    up_vec = np.cross(up_vec, at_vec)
    up_vec = Cartesian3DVector.from_numpy(up_vec)
    up_vec = up_vec.normalize()

    return moon_view.eye + up_vec


if __name__ == '__main__':
    generate_one_image()
