import torch
import numpy as np
from tqdm import tqdm
from ...generate.renderer import Pytorch3DRenderer
from .renderer_model import RendererModel
from ...config import config


class FineTuner:
    def __init__(self):
        self.renderer = Pytorch3DRenderer()

    def fine_tune_predict_positions(self, target_images, predict_positions):
        fine_tuned_predicts = []

        batch_size = config.network.batch_size
        for i in range(1):
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

        epochs = 10
        lr = 0.001
        loss_low_bound = 0.02

        # optimizer = Adam(model.parameters(), lr=lr)
        best_position = [0.0, 0.0, 0.0]
        best_loss = 100

        for i in tqdm(range(epochs)):
            # optimizer.zero_grad()

            now_loss = model()

            # loss.backward()
            # optimizer.step()

            # now_loss = loss.item()

            if now_loss < best_loss:
                best_loss = now_loss
                best_position = [model.dist, model.elev, model.azim]

            if best_loss < loss_low_bound:
                break

        return best_position

    def regression2position(self, predict):
        predict = self.tensor_to_numpy(predict)
        dist = predict[0] * config.generate.dist_between_moon_high_bound_gl + config.generate.moon_radius_gl
        elev = predict[1] * np.pi / 2
        azim = predict[2] * np.pi * 2

        return dist, elev, azim

    def normalize_target_image(self, target_image):
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
