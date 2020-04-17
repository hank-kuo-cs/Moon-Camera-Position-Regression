class MoonView:
    def __init__(self):
        self.viewport = [0, 0]
        self.fov = 0.0
        self.znear = 0.0
        self.zfar = 0.0

    def check_parameters(self):
        assert isinstance(self.viewport, list)
        assert isinstance(self.fov, float)
        assert isinstance(self.znear, float)
        assert isinstance(self.zfar, float)

        assert len(self.viewport) == 2

