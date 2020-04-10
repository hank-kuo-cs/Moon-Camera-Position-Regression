from .light import MoonLight
from .material import MoonMaterial
from .object import MoonObject
from .texture import MoonTexture
from .view import MoonView


class Moon:
    def __init__(self, obj: MoonObject, mtl: MoonMaterial, texture: MoonTexture, light: MoonLight, view: MoonView):
        self.obj = obj
        self.mtl = mtl
        self.texture = texture
        self.light = light
        self.view = view

    def set_view(self, moon_view: MoonView):
        self.check_parameters()
        self.view = moon_view

    def set_light(self, moon_light: MoonLight):
        self.check_parameters()
        self.light = moon_light

    def check_parameters(self):
        assert type(self.obj) == MoonObject
        assert type(self.mtl) == MoonMaterial
        assert type(self.texture) == MoonTexture
        assert type(self.light) == MoonLight
        assert type(self.view) == MoonView
