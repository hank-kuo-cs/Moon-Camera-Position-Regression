from loader import load_moon
from renderer import Renderer
from config import *
import cv2


if __name__ == '__main__':
    moon = load_moon()
    renderer = Renderer(moon=moon)

    renderer.set_window()
    renderer.set_texture()
    renderer.set_moon()

    renderer.set_light()
    renderer.set_view()

    renderer.render_moon()

    renderer.save_image('test.png')
    image = renderer.get_image()
    cv2.imwrite('yo.png', image)
