from model.point import Cardassian3DPoint


class MoonView:
    def __init__(self):
        self.viewport = [0, 0]
        self.fov = 0.0
        self.znear = 0.0
        self.zfar = 0.0
        self.eye = Cardassian3DPoint.from_list([0.0, 0.0, 0.0])
        self.at = Cardassian3DPoint.from_list([0.0, 0.0, 0.0])
        self.up = Cardassian3DPoint.from_list([0.0, 0.0, 0.0])

    def check_parameters(self):
        assert isinstance(self.viewport, list)
        assert isinstance(self.eye, list)
        assert isinstance(self.at, list)
        assert isinstance(self.up, list)
        assert isinstance(self.fov, float)
        assert isinstance(self.znear, float)
        assert isinstance(self.zfar, float)

        assert len(self.viewport) == 2
        assert len(self.eye) == 3
        assert len(self.at) == 3
        assert len(self.up) == 3
