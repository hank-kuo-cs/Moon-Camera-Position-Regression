class MoonLight:
    def __init__(self):
        self._position_light = (0.0, 0.0, 0.0, 0.0)
        self._ambient_light = (0.0, 0.0, 0.0, 0.0)
        self._diffuse_light = (0.0, 0.0, 0.0, 0.0)

    @property
    def position_light(self):
        return self._position_light

    @position_light.setter
    def position_light(self, new_position_light):
        self.check_light(new_position_light)
        self._position_light = new_position_light

    @property
    def ambient_light(self):
        return self._ambient_light

    @ambient_light.setter
    def ambient_light(self, new_ambient_light):
        self.check_light(new_ambient_light)
        self._ambient_light = new_ambient_light

    @property
    def diffuse_light(self):
        return self._diffuse_light

    @diffuse_light.setter
    def diffuse_light(self, new_diffuse_light):
        self.check_light(new_diffuse_light)
        self._diffuse_light = new_diffuse_light

    @staticmethod
    def check_light(light):
        assert isinstance(light, tuple)
        assert len(light) == 4
        for i in range(len(light)):
            assert isinstance(light[i], float)

    def check_parameters(self):
        assert isinstance(self.position_light, tuple)
        assert isinstance(self.ambient_light, tuple)
        assert isinstance(self.diffuse_light, tuple)

        assert len(self.position_light) == 4
        assert len(self.ambient_light) == 4
        assert len(self.diffuse_light) == 4
