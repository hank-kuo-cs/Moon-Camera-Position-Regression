import logging
from loader import load_moon
from renderer import Renderer


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    a = 1.74223 + 1.739946 + 1.745721
    print(a / 3)

    moon = load_moon()
    renderer = Renderer(moon=moon)

    renderer.set_window()
    renderer.set_texture()
    renderer.set_moon()

    renderer.set_light()
    renderer.set_view()

    renderer.render_moon()

    renderer.save_image('test.png')
