from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Regression.data import MoonDataset


def plot_3d_points(xs, ys, zs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig('camera_point_distribution.png')
    # plt.show()

def transform_spherical_to_cartesian(dist, elev, azim):
    x = dist * np.cos(elev) * np.sin(azim)
    y = dist * np.sin(elev)
    z = dist * np.cos(elev) * np.cos(azim)

    return x, y, z


def dataset_test():
    train_datatset = MoonDataset('train')
    test_dataset = MoonDataset('test')
    valid_dataset = MoonDataset('validation')

    dist_list = []
    elev_list = []
    azim_list = []

    for img, label in train_datatset:
        dist = label[0]
        elev = label[1]
        azim = label[2]

        dist_list.append(dist)
        elev_list.append(elev)
        azim_list.append(azim)

    for img, label in test_dataset:
        dist = label[0]
        elev = label[1]
        azim = label[2]

        dist_list.append(dist)
        elev_list.append(elev)
        azim_list.append(azim)

    for img, label in valid_dataset:
        dist = label[0]
        elev = label[1]
        azim = label[2]

        dist_list.append(dist)
        elev_list.append(elev)
        azim_list.append(azim)

    # plt.xlim((0, 1))
    # sns.distplot(dist_list)
    # plt.show()
    #
    # plt.xlim((-1, 1))
    # sns.distplot(elev_list)
    # plt.show()
    #
    # plt.xlim((0, 1))
    # sns.distplot(azim_list)
    # plt.show()
    #
    # plt.close()

    xs = []
    ys = []
    zs = []

    for i in range(len(dist_list)):
        x, y, z = transform_spherical_to_cartesian(dist_list[i], elev_list[i], azim_list[i])
        xs.append(x)
        ys.append(y)
        zs.append(z)

    plot_3d_points(xs, ys, zs)
