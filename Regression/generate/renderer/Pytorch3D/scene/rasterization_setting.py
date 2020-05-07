from pytorch3d.renderer import RasterizationSettings
from .....config import config


def load_rasterization_setting():
        return RasterizationSettings(image_size=config.generate.image_size,
                                     blur_radius=0.0,
                                     faces_per_pixel=1,
                                     bin_size=0)
