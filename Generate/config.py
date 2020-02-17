import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

# Data
OBJECT_PATH = 'data/Moon_8K.obj'
MATERIAL_PATH = 'data/Moon_8K.mtl'
TEXTURE_PATH = 'data/Diffuse_8K.png'

# Window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# View
VIEW = {
    'viewport': [800, 600],
    'fov': 120.0,
    'znear': 1.0,
    'zfar': 100.0,
    'eye': [4, 0, 0],
    'at': [0, 0, 0],
    'up': [0, 1, 0]
}


# Light
LIGHT = {
    'position': (-40, 200, 100, 0.0),
    'ambient': (0.2, 0.2, 0.2, 1.0),
    'diffuse': (0.5, 0.5, 0.5, 1.0)
}

# Unit
MOON_RADIUS_IN_GL_UNIT = 1.742887
GL_UNIT_TO_KM = 996.679647

# Dataset
DATASET_PATH = '/data/space/dataset'
DATA_NUM = 100000
GAMMA_RANGE = [0.2, 10]  # km
IS_CHANGE_EYE = True
IS_CHANGE_AT = True
IS_CHANGE_UP = False
