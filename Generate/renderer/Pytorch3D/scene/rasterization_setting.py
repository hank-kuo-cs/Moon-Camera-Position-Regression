from ....pytorch3d.renderer import RasterizationSettings
from ....config import WINDOW_WIDTH, WINDOW_HEIGHT


def load_rasterization_setting():
        assert WINDOW_HEIGHT == WINDOW_WIDTH
        return RasterizationSettings(image_size=WINDOW_WIDTH,
                                     blur_radius=0.0,
                                     faces_per_pixel=1,
                                     bin_size=0)
