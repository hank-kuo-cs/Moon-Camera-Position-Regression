import re
from loader.model import MoonMaterial


class MaterialEncoder:
    def __init__(self, mtl_path):
        self.mtl_path = mtl_path
        self.moon_material = MoonMaterial()
        self.plain_file = None

    def load_material(self) -> MoonMaterial:
        with open(self.mtl_path, 'r') as f:
            self.plain_file = f.read()

        self.parse_material()

        return self.moon_material

    def parse_material(self):
        diffuse_color_str = re.findall(r'Kd ([0-9.]+ [0-9.]+ [0-9.]+) *\n', self.plain_file)[0]
        self.moon_material.diffuse_color = self.color_str_to_float_vec(diffuse_color_str)

    @staticmethod
    def color_str_to_float_vec(color_str: str) -> list:
        return list(map(lambda x: float(x), color_str.split()))
