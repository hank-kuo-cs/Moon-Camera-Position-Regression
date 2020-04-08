import logging
from tqdm import tqdm
from .loader import load_moon
from .dataset import DatasetWriter
from .renderer import OpenGLRenderer
from .renderer import Pytorch3DRenderer
from .config import DATA_NUM


def run_opengl(moon, dataset_writer):
    logging.info('Set up renderer...')
    renderer = OpenGLRenderer(moon=moon)

    renderer.set_window()
    renderer.set_texture()
    renderer.set_moon()
    renderer.set_light()

    logging.info('Render moon image with uniform view setting...')
    for i in tqdm(range(DATA_NUM)):
        renderer.set_view(dataset_writer.get_moon_view())
        renderer.render_moon()
        image = renderer.get_image()

        dataset_writer.write_data(image=image, moon=renderer.moon)


def run_pytorch3d(moon, dataset_writer):
    logging.info('Set up renderer...')
    renderer = Pytorch3DRenderer(moon=moon)

    renderer.set_device()
    renderer.set_mesh()
    renderer.set_lights()
    renderer.set_raster_settings()

    logging.info('Render moon image with uniform view setting...')
    for i in tqdm(range(DATA_NUM)):
        renderer.set_cameras(dataset_writer.get_moon_view())
        image = renderer.render_image()
        dataset_writer.write_data(image=image, moon=renderer.moon)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    logging.info('Load moon file...')
    moon = load_moon()

    logging.info('Set up dataset writer...')
    dataset_writer = DatasetWriter(moon=moon)
    run_opengl(moon, dataset_writer)
    # run_pytorch3d(moon, dataset_writer)
