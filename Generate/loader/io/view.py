from loader.model import MoonView
from config import VIEWPORT, FOV, DNEAR, DFAR, EYE, AT, UP


class ViewEncoder:
    def __init__(self):
        self.view = MoonView()

    def load_view(self) -> MoonView:
        self.view.viewport = VIEWPORT
        self.view.fov = FOV
        self.view.d_near = DNEAR
        self.view.d_far = DFAR
        self.view.eye = EYE
        self.view.at = AT
        self.view.up = UP

        return self.view
