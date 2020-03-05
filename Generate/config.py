# Window
"""
Window size equals to your image size.
"""
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800

# Data
OBJECT_PATH = '../../data/Moon_8K.obj'
MATERIAL_PATH = '../../data/Moon_8K.mtl'
TEXTURE_PATH = '../../data/Diffuse_8K.png'

# View
"""
You don't have to change this setting unless you want to generate single image,
then you can change the eye, at, and up to your own setting,
note that eye, at, and up are in cartesian coordinate.
"""
VIEW = {
    'viewport': [800, 800],
    'fov': 120.0,
    'znear': 1.0,
    'zfar': 100.0,
    'eye': [1.745721, 0, 0],
    'at': [1, 1, 1],
    'up': [0, 1, 0]
}

# Light
LIGHT = {
    'position': (-40.0, 200.0, 100.0, 0.0),
    'ambient': (0.2, 0.2, 0.2, 1.0),
    'diffuse': (0.5, 0.5, 0.5, 1.0)
}

# Unit
MOON_MAX_RADIUS_IN_GL_UNIT = 1.745721
GL_UNIT_TO_KM = 1000
KM_TO_GL_UNIT = 0.001

# Dataset
DATASET_PATH = '../../Dataset_fix_up_100km'
DATA_NUM = 100000
GAMMA_RANGE = [0.0, 100]  # km
IS_CHANGE_EYE = True
IS_CHANGE_AT = True
IS_CHANGE_UP = False
PYR_DOWN_TIME = 1
