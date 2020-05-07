from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from Regression.data import MoonDataset
from Regression.config import config


high_km = config.generate.dist_between_moon_high_bound_km
moon_radius = config.generate.moon_radius_gl * config.generate.gl_to_km


def plot_3d_points(xs, ys, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # plt.savefig('camera_point_distribution.png')
    plt.show()


def transform_spherical_to_cartesian(dist, elev, azim):
    x = dist * np.cos(elev) * np.sin(azim)
    y = dist * np.sin(elev)
    z = dist * np.cos(elev) * np.cos(azim)

    return x, y, z


def load_dataset(dataset):
    dist_list = []
    elev_list = []
    azim_list = []

    for img, label in tqdm(dataset):
        dist = label[0] * high_km + moon_radius
        elev = label[1] * np.pi / 2
        azim = label[2] * np.pi * 2

        dist_list.append(dist)
        elev_list.append(elev)
        azim_list.append(azim)

    return dist_list, elev_list, azim_list


def dataset_test():
    train_datatset = MoonDataset('train')
    test_dataset = MoonDataset('test')
    valid_dataset = MoonDataset('validation')

    dists = []
    elevs = []
    azims = []
    xs = []
    ys = []
    zs = []

    dataset_types = ['train', 'test', 'validation']

    for dataset_type in dataset_types:
        print('loading %s dataset...' % dataset_type)
        moon_dataset = MoonDataset(dataset_type)

        dist_list, elev_list, azim_list = load_dataset(moon_dataset)
        dists += dist_list
        elevs += elev_list
        azims += azim_list

    print('transform all spherical points to cartesian coordinate...')
    for i in tqdm(range(len(dists))):
        x, y, z = transform_spherical_to_cartesian(dists[i], elevs[i], azims[i])
        xs.append(x)
        ys.append(y)
        zs.append(z)

    plot_3d_points(xs, ys, zs)
