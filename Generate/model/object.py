class MoonObject:
    def __init__(self):
        self.vertices = []
        self.tex_vertices = []
        self.normals = []
        self.faces = []

    def check_parameters(self):
        assert len(self.vertices) == len(self.normals)


class MoonFaceQuadMesh:
    def __init__(self):
        self.vertex_indices = []
        self.texture_indices = []
        self.normal_indices = []

    def check_parameters(self):
        assert len(self.vertex_indices) == 4
        assert type(self.vertex_indices[0]) == int

        if len(self.texture_indices) > 0:
            assert type(self.texture_indices[0]) == int
        if len(self.normal_indices) > 0:
            assert type(self.normal_indices[0]) == int
