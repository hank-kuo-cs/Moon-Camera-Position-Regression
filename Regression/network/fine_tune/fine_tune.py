import cv2
import torch
import numpy as np
from tqdm import tqdm
from ...generate.renderer import Pytorch3DRenderer
from .renderer_model import RendererModel
from ...config import config


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


class FineTuner:
    def __init__(self):
        self.renderer = Pytorch3DRenderer()

    def fine_tune_predict_positions(self, target_images, predict_positions):
        fine_tuned_predicts = []

        for i in range(len(predict_positions)):
            target_image = target_images[i]
            predict_position = predict_positions[i]

            fine_tuned_predict = self._fine_tune_one_position(target_image, predict_position)
            fine_tuned_predicts.append(self.position2regression(fine_tuned_predict))

        fine_tuned_predicts = torch.tensor(fine_tuned_predicts)

        assert fine_tuned_predicts.size() == predict_positions.size()
        return fine_tuned_predicts

    def _fine_tune_one_position(self, target_image, predict_position):
        dist, elev, azim = self.regression2position(predict_position)
        target_image = self.normalize_target_image(target_image)

        model = RendererModel(self.renderer, target_image, dist, elev, azim).to(config.cuda.device)

        best_position = [0.0, 0.0, 0.0]
        best_loss = 10000.0

        for i in range(config.fine_tune.epoch_num):
            now_loss = model()
            fine_tune_prediction = self.position2regression([model.dist, model.elev, model.azim])

            print('\nepoch %d' % (i + 1))
            print('loss = %.6f' % now_loss)
            print('prediction:', fine_tune_prediction)

            if now_loss < best_loss:
                best_loss = now_loss
                best_position = [model.dist, model.elev, model.azim]

            if best_loss < config.fine_tune.low_loss_bound:
                break

        return best_position

    def regression2position(self, predict):
        predict = self.tensor_to_numpy(predict)
        dist = predict[0] * config.generate.dist_between_moon_high_bound_gl + config.generate.moon_radius_gl
        elev = predict[1] * np.pi / 2
        azim = predict[2] * np.pi * 2

        return dist, elev, azim

    @staticmethod
    def normalize_target_image(target_image):
        return target_image[None, ...]

    @staticmethod
    def position2regression(position):
        position[0] = (position[0] - config.generate.moon_radius_gl) / config.generate.dist_between_moon_high_bound_gl
        position[1] /= (np.pi / 2)
        position[2] /= (np.pi * 2)

        return position

    @staticmethod
    def tensor_to_numpy(tensor_array):
        assert isinstance(tensor_array, torch.Tensor)

        tensor_array = tensor_array.clone()
        if tensor_array.requires_grad:
            tensor_array = tensor_array.detach()
        if config.cuda.device != 'cpu':
            tensor_array = tensor_array.cpu()

        numpy_array = tensor_array.numpy()
        return numpy_array
