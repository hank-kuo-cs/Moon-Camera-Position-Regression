import logging
from tqdm import tqdm
from .dataset import DatasetWriter
from .renderer import Pytorch3DRenderer
from ..config import config


def generate_dataset():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    logging.info('Load moon file...')

    logging.info('Set up dataset writer...')
    dataset_writer = DatasetWriter()

    logging.info('Set up renderer...')
    renderer = Pytorch3DRenderer()

    logging.info('Render moon images...')
    for i in tqdm(range(config.dataset.dataset_num)):
        dist, elev, azim, at, up = dataset_writer.get_random_cameras()

        renderer.set_cameras(dist, elev, azim, at, up)
        image = renderer.render_image()
        img_data = renderer.refine_image_to_data(image)

        dataset_writer.write_data(image=img_data)
