import cv2
import torch
from Regression.network.fine_tune import FineTuner
import numpy as np
from torchvision import transforms
from Regression.generate.renderer import Pytorch3DRenderer
from Regression.config import config


def load_image(image_path):
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)

    return image


def render_one_image(dist, elev, azim, at=(0, 0, 0), up=(0, 1, 0), degree=False, image_name='result.png'):
    if degree:
        elev *= (np.pi / 180)
        azim *= (np.pi / 180)

    moon_radius = config.generate.moon_radius_gl
    km2gl = config.generate.km_to_gl

    dist = dist * km2gl + moon_radius

    renderer = Pytorch3DRenderer()
    renderer.set_cameras(dist, elev, azim, at, up)
    predict_image = renderer.render_image()

    refine_image = renderer.refine_image_to_data(predict_image)
    cv2.imwrite(image_name, refine_image)


def fine_tune():
    # dist (km)
    # render_one_image(dist=11, elev=1, azim=1, degree=True, image_name='init3.png')
    # render_one_image(dist=10, elev=0, azim=0, degree=True, image_name='target.png')
    fine_tuner = FineTuner()
    target_image = load_image('target.png')

    target_images = [target_image]
    gt_positions = torch.tensor([[10 / 15, 0 / 180 * 2, 0 / 180 / 2]])
    predict_positions = torch.tensor([[11 / 15, 1 / 180 * 2, 1 / 180 / 2]])
    print('target prediction:', gt_positions)
    print('coarse prediction:', predict_positions)

    fine_tuned_positions = fine_tuner.fine_tune_predict_positions(target_images, predict_positions.clone())

    print('target prediction:', gt_positions)
    print('coarse prediction:', predict_positions)
    print('fine tune prediction:', fine_tuned_positions)
