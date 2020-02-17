from loader.model import MoonView
from config import VIEW


class ViewEncoder:
    def __init__(self):
        self.view = MoonView()

    def load_view(self) -> MoonView:
        self.view.viewport = VIEW['viewport']
        self.view.fov = VIEW['fov']
        self.view.znear = VIEW['znear']
        self.view.zfar = VIEW['zfar']
        self.view.eye = VIEW['eye']
        self.view.at = VIEW['at']
        self.view.up = VIEW['up']

        return self.view
