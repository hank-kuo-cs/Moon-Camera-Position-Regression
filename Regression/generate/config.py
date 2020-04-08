import os
from .model import Cartesian3DPoint, Spherical3DPoint
# Window
"""
Window size equals to your image size.
"""
WINDOW_WIDTH = 100
WINDOW_HEIGHT = 100

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
    'fov': 120,
    'znear': 0.0001,
    'zfar': 1000.0,
    'eye': [2.0, 0.0, 0.0],
    'at': [0.0, 0.0, 0.0],
    'up': [0.0, 1.0, 0.0]
}

# Light
LIGHT = {
    'position': (-40.0, 200.0, 100.0, 0.0),
    'ambient': (0.5, 0.5, 0.5, 1.0),
    'diffuse': (0.5, 0.5, 0.5, 1.0)
}

# Unit
MOON_MAX_RADIUS_IN_GL_UNIT = 1.7459008620440053
# GL_UNIT_TO_KM = 1737.1 / MOON_MAX_RADIUS_IN_GL_UNIT
# KM_TO_GL_UNIT = 1 / GL_UNIT_TO_KM
GL_UNIT_TO_KM = 1000
KM_TO_GL_UNIT = 0.001

# Dataset
DATASET_PATH = '../../Dataset_test'
DATA_NUM = 100
GAMMA_RANGE = [1.0, 15.0]  # km
IS_CHANGE_EYE = True
IS_CHANGE_AT = False
IS_CHANGE_UP = False

# GPU
DEVICE_NUM = 0

os.environ['CUDA_VISIBLE_DEVICES'] = str(DEVICE_NUM)
