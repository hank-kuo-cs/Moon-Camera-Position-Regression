import pygame
from loader import load_moon
from renderer import Renderer
from config import *


if __name__ == '__main__':
    logging.info('Load Moon')
    moon = load_moon()
    logging.info('Render Moon')
    pygame.init()
    surface = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    renderer = Renderer(moon)
    renderer.render_moon()
    pygame.image.save(surface, 'test.png')
    # renderer.export_image('test.png')
