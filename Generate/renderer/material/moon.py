from loader.moon import Moon
from renderer.material.lib import *


class MoonSetting:
    def __init__(self, moon: Moon):
        self.moon = moon

    def set_moon(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glShadeModel(GL_SMOOTH)
        glFrontFace(GL_CCW)

        polygon_list_id = self.draw_polygon_list()
        return polygon_list_id

    def draw_polygon_list(self):
        polygon_list_id = glGenLists(1)

        glNewList(polygon_list_id, GL_COMPILE)

        for face in self.moon.obj.faces:
            glBegin(GL_POLYGON)

            for i in range(4):
                normal = self.moon.obj.normals[face.normal_indices[i]]
                texture = self.moon.obj.tex_vertices[face.texture_indices[i]]
                vertex = self.moon.obj.vertices[face.vertex_indices[i]]

                glNormal3fv(normal)
                glTexCoord2fv(texture)
                glVertex3fv(vertex)

            glEnd()
        glEndList()

        return polygon_list_id
