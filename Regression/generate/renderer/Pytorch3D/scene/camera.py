import torch
from pytorch3d.renderer import look_at_view_transform, OpenGLPerspectiveCameras
from ....model import MoonView


def load_cameras(moon_view: MoonView):
    device = torch.device('cuda')

    eye = moon_view.eye
    at = moon_view.at
    up = moon_view.up

    R, T = look_at_view_transform(eye=((eye[0], eye[1], eye[2]),),
                                  at=((at[0], at[1], at[2]),),
                                  up=((up[0], up[1], up[2]),))

    return OpenGLPerspectiveCameras(device=device,
                                    R=R,
                                    T=T,
                                    degrees=True,
                                    fov=moon_view.fov,
                                    znear=moon_view.znear,
                                    zfar=moon_view.zfar)
