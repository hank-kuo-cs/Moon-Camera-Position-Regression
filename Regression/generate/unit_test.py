import cv2
import numpy as np
import json
from glob import glob
import logging
from .loader import load_object, load_texture, load_material, load_view, load_light
from .model import Moon


def load_moon() -> Moon:
    moon_obj = load_object()
    moon_obj.check_parameters()

    moon_mtl = load_material()
    moon_mtl.check_parameters()

    moon_texture = load_texture()
    moon_texture.check_parameters()

    moon_light = load_light()
    moon_light.check_parameters()

    moon_view = load_view()
    moon_view.check_parameters()

    return Moon(obj=moon_obj,
                mtl=moon_mtl,
                texture=moon_texture,
                light=moon_light,
                view=moon_view)


def dataset_test():
    image_path = '00.png'
    test_image_path = 'test2.png'

    img_a = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    img_c = abs(img_a - img_b)
    img_c = cv2.equalizeHist(img_c)

    cv2.imwrite('dist.png', img_c)

    # print(img_a.shape)
    # print(img_b.shape)

    dist = np.sum(np.abs(img_a - img_b))
    print(dist)


def label_test(dataset_path):
    labels_path = glob(dataset_path + '/test/label/*.json')
    for label in labels_path:
        with open(label, 'r') as f:
            data = json.load(f)
            print(data[0])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

    # logging.info('Check loading data...')
    # moon = load_moon()
    # moon.check_parameters()
    # dataset_test()
    #
    # image_path = '0.png'
    # test_image_path = 'test0.png'
    #
    # img_a = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img_b = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    # img_c = abs(img_a - img_b)
    # img_c = cv2.equalizeHist(img_c)
    #
    # cv2.imwrite('dist.png', img_c)
    #
    # # print(img_a.shape)
    # # print(img_b.shape)
    #
    # dist = np.sum(np.abs(img_a - img_b))
    # print(dist)

    # label_test('../../Dataset_300km')

    moon = load_moon()
    vertices = np.array(moon.obj.vertices)
    np.max(np.linalg.norm(vertices, axis=0))
    # renderer = Renderer(moon=moon)
    # renderer.set_window()
    # renderer.set_texture()
    # renderer.set_light()
    # renderer.set_view()