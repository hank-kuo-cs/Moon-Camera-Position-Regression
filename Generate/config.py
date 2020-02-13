import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

# Data
OBJECT_PATH = 'data/Moon_8K.obj'
MATERIAL_PATH = 'data/Moon_8K.mtl'
TEXTURE_PATH = 'data/Diffuse_8K.png'

# Window
WINDOW_NAME = 'Generate Moon Data'
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_POSITION = [0, 0]

# View
VIEWPORT = [800, 600]
FOV = 120.0
DNEAR = 1.0
DFAR = 100.0
EYE = [0, 0, 0]
AT = [0, 0, 0]
UP = [0, 0, 0]
IS_CHANGE_EYE = True
IS_CHANGE_AT = True
IS_CHANGE_UP = False

# Light
POSITION_LIGHT = (-40, 200, 100, 0.0)
AMBIENT_LIGHT = (0.2, 0.2, 0.2, 1.0)
DIFFUSE_LIGHT = (0.5, 0.5, 0.5, 1.0)

# Unit
MOON_RADIUS_IN_GL_UNIT = 1.742887
GL_UNIT_TO_KM = 996.679647

# Dataset
DATASET_PATH = '/data/space/dataset'
DATA_NUM = 100000
