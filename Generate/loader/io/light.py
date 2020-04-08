from ...config import LIGHT
from ...model import MoonLight


class LightEncoder:
    def __init__(self):
        self.light = MoonLight()

    def load_light(self) -> MoonLight:
        self.light.position_light = LIGHT['position']
        self.light.ambient_light = LIGHT['ambient']
        self.light.diffuse_light = LIGHT['diffuse']

        return self.light
