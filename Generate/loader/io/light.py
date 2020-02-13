from config import POSITION_LIGHT, AMBIENT_LIGHT, DIFFUSE_LIGHT
from loader.model import MoonLight


class LightEncoder:
    def __init__(self):
        self.light = MoonLight()

    def load_light(self) -> MoonLight:
        self.light.position_light = POSITION_LIGHT
        self.light.ambient_light = AMBIENT_LIGHT
        self.light.diffuse_light = DIFFUSE_LIGHT

        return self.light
