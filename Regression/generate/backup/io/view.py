from ...model import MoonView, Cartesian3DPoint
from ...config import VIEW


class ViewEncoder:
    def __init__(self):
        self.view = MoonView()

    def load_view(self) -> MoonView:
        self.view.viewport = VIEW['viewport']
        self.view.fov = VIEW['fov']
        self.view.znear = VIEW['znear']
        self.view.zfar = VIEW['zfar']
        self.view.eye = Cartesian3DPoint.from_list(VIEW['eye'])
        self.view.at = Cartesian3DPoint.from_list(VIEW['at'])
        self.view.up = Cartesian3DPoint.from_list(VIEW['up'])

        return self.view
