# Window
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Data
OBJECT_PATH = 'data/Moon_8K.obj'
MATERIAL_PATH = 'data/Moon_8K.mtl'
TEXTURE_PATH = 'data/Diffuse_8K.png'

# View
VIEW = {
    'viewport': [800, 600],
    'fov': 120.0,
    'znear': 1.0,
    'zfar': 100.0,
    'eye': [0, 0, 1],  # Cardassian Coordinate
    'at': [0, 0, 0],  # Cardassian Coordinate
    'up': [0, 1, 0]  # Cardassian Coordinate
}

# Light
LIGHT = {
    'position': (-40.0, 200.0, 100.0, 0.0),
    'ambient': (0.2, 0.2, 0.2, 1.0),
    'diffuse': (0.5, 0.5, 0.5, 1.0)
}

# Unit
MOON_AVG_RADIUS_IN_GL_UNIT = 1.74263233333
GL_UNIT_TO_KM = 996.825301
KM_TO_GL_UNIT = 0.0010031848

# Dataset
DATASET_PATH = '../Dataset'
DATA_NUM = 10000
GAMMA_RANGE = [0.2, 80]  # km
IS_CHANGE_EYE = True
IS_CHANGE_AT = True
IS_CHANGE_UP = True
