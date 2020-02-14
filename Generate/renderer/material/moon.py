from loader.moon import Moon
from renderer.material.lib import *
from OpenGL.GL import glDisable, GL_TEXTURE_2D
from tqdm import tqdm


class MoonSetting:
    def __init__(self, moon: Moon):
        self.moon = moon
        self.polygon_list_id = glGenLists(1)

    def set_moon(self):
        glFrontFace(GL_CCW)
        glNewList(self.polygon_list_id, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)

        for face in tqdm(self.moon.obj.faces):
            glBegin(GL_POLYGON)
            face.check_parameters()

            for i in range(4):
                vertex = self.moon.obj.vertices[face.vertex_indices[i]]
                normal = self.moon.obj.normals[face.normal_indices[i]]
                texture = self.moon.obj.tex_vertices[face.texture_indices[i]]

                glVertex3fv(vertex)
                glNormal3fv(normal)
                glTexCoord2fv(texture)

            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()
