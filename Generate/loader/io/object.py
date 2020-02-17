import re
from loader.model import MoonObject, MoonFaceQuadMesh


class ObjectEncoder:
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.moon_object = MoonObject()
        self.plain_file = ''

    def load_object(self) -> MoonObject:
        with open(self.obj_path, 'r') as f:
            self.plain_file = f.read()

        self.parse_vertices()
        self.parse_texture_vertices()
        self.parse_normal_vectors()
        self.parse_faces()

        return self.moon_object

    def parse_vertices(self):
        plain_vertices = re.findall(r'v ([-.0-9]+ [-0-9.]+ [-0-9.]+ *)\n', self.plain_file)
        self.moon_object.vertices = [self.float_str_to_float_list(plain_vertex) for plain_vertex in plain_vertices]

    def parse_texture_vertices(self):
        plain_textures = re.findall(r'vt ([-.0-9]+ [-0-9.]+ *)\n', self.plain_file)
        self.moon_object.tex_vertices = [self.float_str_to_float_list(plain_texture) for plain_texture in plain_textures]

    def parse_normal_vectors(self):
        plain_normals = re.findall(r'vn ([-.0-9]+ [-0-9.]+ [-0-9.]+ *)\n', self.plain_file)
        self.moon_object.normals = [self.float_str_to_float_list(plain_normal) for plain_normal in plain_normals]

    def parse_faces(self):
        plain_faces = re.findall(r'f ([ 0-9/]+)\n', self.plain_file)
        self.moon_object.faces = [self.face_str_to_face_list(plain_face) for plain_face in plain_faces]

    @staticmethod
    def float_str_to_float_list(vec_str: str) -> list:
        vector = list(map(lambda x: float(x), vec_str.split()))
        if len(vector) == 3:
            vector = [vector[0], vector[2], vector[1]]

        return vector

    @staticmethod
    def face_str_to_face_list(face_vec: str) -> MoonFaceQuadMesh:
        face = MoonFaceQuadMesh()

        for face_str in face_vec.split():
            indices = face_str.split('/')
            face.vertex_indices.append(int(indices[0]) - 1)
            face.texture_indices.append(int(indices[1]) - 1)
            face.normal_indices.append(int(indices[2]) - 1)

        return face
