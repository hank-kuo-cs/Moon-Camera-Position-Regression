from .point import Cartesian3DPoint


class MoonView:
    def __init__(self):
        self.viewport = [0, 0]
        self.fov = 0.0
        self.znear = 0.0
        self.zfar = 0.0
        self.eye = Cartesian3DPoint.from_list([0.0, 0.0, 0.0])
        self.at = Cartesian3DPoint.from_list([0.0, 0.0, 0.0])
        self.up = Cartesian3DPoint.from_list([0.0, 0.0, 0.0])

    def check_parameters(self):
        assert isinstance(self.viewport, list)
        assert isinstance(self.eye, Cartesian3DPoint)
        assert isinstance(self.at, Cartesian3DPoint)
        assert isinstance(self.up, Cartesian3DPoint)
        assert isinstance(self.fov, float)
        assert isinstance(self.znear, float)
        assert isinstance(self.zfar, float)

        assert len(self.viewport) == 2

