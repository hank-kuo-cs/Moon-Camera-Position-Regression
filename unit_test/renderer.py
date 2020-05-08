import cv2
import numpy as np
from Regression.generate.renderer import Pytorch3DRenderer
from Regression.config import config


def render_one_image(dist, elev, azim, at=(0, 0, 0), up=(0, 1, 0), degree=False, img_name=None):
    if degree:
        elev *= (np.pi / 180)
        azim *= (np.pi / 180)

    moon_radius = config.generate.moon_radius_gl
    km2gl = config.generate.km_to_gl

    dist = dist * km2gl + moon_radius

    renderer = Pytorch3DRenderer()
    renderer.set_cameras(dist, elev, azim, at, up)
    image = renderer.render_image()
    image = renderer.refine_image_to_data(image)

    if img_name is None:
        img_name = 'result.png'
    cv2.imwrite(img_name, image)


if __name__ == '__main__':
    # dist (km), elev (degree or rad), azim (degree or rad)

    dists = [1]
    elevs = [0, 0.1]
    azims = [0, 0.1]

    for dist in dists:
        for elev in elevs:
            for azim in azims:
                render_one_image(dist, elev, azim, degree=True, img_name='result_%dkm_%f_%f.png' % (dist, elev, azim))

