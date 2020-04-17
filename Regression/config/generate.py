class GenerateConfig:
    def __init__(self,
                 moon_obj_path: str,
                 image_size: int,
                 fov: int,
                 znear: float,
                 zfar: float,
                 moon_radius_gl: float,
                 gl_to_km: float,
                 dist_between_moon_low_bound_km: float,
                 dist_between_moon_high_bound_km: float,
                 is_change_eye: bool,
                 is_change_at: bool,
                 is_change_up: bool):
        self.moon_obj_path = moon_obj_path
        self.image_size = image_size
        self.fov = fov
        self.znear = znear
        self.zfar = zfar
        self.moon_radius_gl = moon_radius_gl
        self.gl_to_km = gl_to_km
        self.dist_between_moon_low_bound_km = dist_between_moon_low_bound_km
        self.dist_between_moon_high_bound_km = dist_between_moon_high_bound_km
        self.is_change_eye = is_change_eye
        self.is_change_at = is_change_at
        self.is_change_up = is_change_up

    @property
    def km_to_gl(self) -> float:
        return 1 / self.gl_to_km

    @property
    def dist_low_gl(self) -> float:
        return self.moon_radius_gl + self.dist_between_moon_low_bound_km * self.km_to_gl

    @property
    def dist_high_gl(self) -> float:
        return self.moon_radius_gl + self.dist_between_moon_high_bound_km * self.km_to_gl

    def check_parameters(self):
        assert isinstance(self.moon_obj_path, str) and len(self.moon_obj_path) > 0
        assert isinstance(self.image_size, int) and self.image_size > 0
        assert isinstance(self.fov, int) and self.fov > 0
        assert isinstance(self.znear, float) and self.znear > 0
        assert isinstance(self.zfar, float) and self.zfar > 0
        assert isinstance(self.moon_radius_gl, float) and self.moon_radius_gl > 0
        assert isinstance(self.gl_to_km, float) and self.gl_to_km > 0
        assert isinstance(self.dist_between_moon_low_bound_km, float) and self.dist_between_moon_low_bound_km > 0
        assert isinstance(self.dist_between_moon_high_bound_km, float) and self.dist_between_moon_high_bound_km > 0
        assert isinstance(self.is_change_eye, bool)
        assert isinstance(self.is_change_at, bool)
        assert isinstance(self.is_change_up, bool)


# View
"""
You don't have to change this setting unless you want to generate single image,
then you can change the eye, at, and up to your own setting,
note that eye, at, and up are in cartesian coordinate.
"""
# VIEW = {
#     'viewport': [800, 800],
#     'fov': 120,
#     'znear': 0.0001,
#     'zfar': 1000.0,
#     'eye': [2.0, 0.0, 0.0],
#     'at': [0.0, 0.0, 0.0],
#     'up': [0.0, 1.0, 0.0]
# }
#
# # Light
# LIGHT = {
#     'position': (-40.0, 200.0, 100.0, 0.0),
#     'ambient': (0.5, 0.5, 0.5, 1.0),
#     'diffuse': (0.5, 0.5, 0.5, 1.0)
# }
#
# # Unit
# MOON_MAX_RADIUS_IN_GL_UNIT = 1.7459008620440053
# GL_UNIT_TO_KM = 1000
# KM_TO_GL_UNIT = 0.001
#
# # Dataset
# DATASET_PATH = '/data/space/Dataset'
# DATA_NUM = 200000
# GAMMA_RANGE = [1.0, 15.0]  # km
# IS_CHANGE_EYE = True
# IS_CHANGE_AT = False
# IS_CHANGE_UP = False
#
# # GPU
# DEVICE_NUM = 0
# CUDA_DEVICE = 'cuda'
#
# if CUDA_DEVICE == 'cuda':
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(DEVICE_NUM)