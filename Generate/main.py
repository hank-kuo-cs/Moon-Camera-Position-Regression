import logging
from tqdm import tqdm
from loader import load_moon
from dataset import DatasetWriter
from renderer import Renderer
from config import DATA_NUM


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    logging.info('Load moon file...')
    moon = load_moon()

    logging.info('Set up dataset writer...')
    dataset_writer = DatasetWriter(moon=moon)

    logging.info('Set up renderer...')
    renderer = Renderer(moon=moon)

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

    # renderer.set_view()
    # renderer.render_moon()
    # renderer.save_image('test.png')
