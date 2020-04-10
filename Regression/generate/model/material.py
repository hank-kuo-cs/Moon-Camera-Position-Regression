class MoonMaterial:
    def __init__(self):
        self.ambient_color = None
        self.diffuse_color = None
        self.specular_color = None
        self.emission_color = None
        self.Ns = None
        self.Ni = None
        self.d = 1
        self.illum = 2
        self.diffuse_map = None
        self.ambient_map = None

    def check_parameters(self):
        assert len(self.diffuse_color) == 3
        assert type(self.diffuse_color[0]) == float
