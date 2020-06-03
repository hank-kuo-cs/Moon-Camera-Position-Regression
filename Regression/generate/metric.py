import os
import cv2
import logging
import numpy as np
from tqdm import tqdm
from .dataset.camera import RandomCameraGenerator
from .renderer import Pytorch3DRenderer
from ..config import config


def get_random_sign():
    a = np.random.rand()
    return 1 if a > 0.5 else -1


def transform_spherical_to_cartesian(dist, elev, azim):
    x = dist * np.cos(elev) * np.sin(azim)
    y = dist * np.sin(elev)
    z = dist * np.cos(elev) * np.cos(azim)

    return x, y, z


def get_km_between_moon_from_gl(dist):
    moon_radius_gl = config.generate.moon_radius_gl
    gl_to_km = config.generate.gl_to_km

    return (dist - moon_radius_gl) * gl_to_km


def get_gl_from_km_between_moon(dist):
    moon_radius_gl = config.generate.moon_radius_gl
    km_to_gl = config.generate.km_to_gl

    return dist * km_to_gl + moon_radius_gl


def get_random_dist_based_on_dist(dist):
    dist_high = config.generate.dist_high_gl
    dist_low = config.generate.dist_low_gl
    dist_km = get_km_between_moon_from_gl(dist)

    km_offset = np.random.uniform(0, dist_km * 0.5)
    result = dist + km_offset * config.generate.km_to_gl * get_random_sign()
    result = np.clip(result, dist_low, dist_high)

    return result


def get_random_elev_azim_offset_with_ellipse_uniform(elev_low, elev_high, azim_low, azim_high):
    theta = np.random.rand() * np.pi * 2

    if elev_low == 0 and azim_low == 0:
        r_low = 0.0
    else:
        r_low = np.sqrt(elev_low ** 2 * azim_low ** 2 / (elev_low ** 2 * np.cos(theta) ** 2 + azim_low ** 2 * np.sin(theta) ** 2))
    r_high = np.sqrt(elev_high ** 2 * azim_high ** 2 / (elev_high ** 2 * np.cos(theta) ** 2 + azim_high ** 2 * np.sin(theta) ** 2))

    r = np.random.uniform(r_low, r_high)

    elev_offset = r * np.sin(theta)
    azim_offset = r * np.cos(theta)

    return elev_offset, azim_offset


def generate_metric_dataset():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')
    logging.info('Set up dataset writer...')
    random_camera_generator = RandomCameraGenerator()

    logging.info('Set up renderer...')
    renderer = Pytorch3DRenderer()

    elev_bins = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
    azim_bins = [0.0, 1.0, 3.0, 5.0, 8.0, 12.0]

    xs = []
    ys = []
    zs = []

    for i in tqdm(range(20000)):
        if i < 16000:
            dir_path = '/data/space/metric_dataset/train'
        elif i < 18000:
            dir_path = '/data/space/metric_dataset/test'
        else:
            dir_path = '/data/space/metric_dataset/valid'

        dist, elev, azim, at, up = random_camera_generator.get_random_camera()

        print('\n0:', get_km_between_moon_from_gl(dist), elev / np.pi * 180, azim / np.pi * 180)

        # a sample
        renderer.set_cameras(dist, elev, azim, at, up)
        image = renderer.render_image()
        img_data = renderer.refine_image_to_data(image)

        for j in range(len(elev_bins) - 1):
            random_dist = get_random_dist_based_on_dist(dist)
            elev_offset, azim_offset = get_random_elev_azim_offset_with_ellipse_uniform(elev_bins[j], elev_bins[j + 1],
                                                                                        azim_bins[j], azim_bins[j + 1])
            random_elev = elev + elev_offset * np.pi / 180
            random_azim = azim + azim_offset * np.pi / 180

            random_at = random_camera_generator.get_random_at()
            random_up = random_camera_generator.get_random_up(random_dist, random_elev, random_azim, random_at)

            renderer.set_cameras(random_dist, random_elev, random_azim, random_at, random_up)

            print('%d:' % (j + 1), get_km_between_moon_from_gl(random_dist), random_elev / np.pi * 180, random_azim / np.pi * 180, 'offset=', elev_offset, azim_offset)

            # x, y, z = transform_spherical_to_cartesian(random_dist, random_elev, random_azim)

            # random_image = renderer.render_image()
            # random_img_data = renderer.refine_image_to_data(random_image)

            # img_path = os.path.join(dir_path, '%d' % i, '%d.png' % j)
            # os.makedirs(img_path, exist_ok=True)

            # cv2.imwrite(img_path, random_img_data)
