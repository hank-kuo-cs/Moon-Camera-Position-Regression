import torch
from pytorch3d.renderer import look_at_view_transform, OpenGLPerspectiveCameras
from model import MoonView


def load_cameras(moon_view: MoonView):
    device = torch.device('cuda')

    eye = moon_view.eye.to_list()
    at = moon_view.at.to_list()
    up = moon_view.up.to_list()

    R, T = look_at_view_transform(eye=((*eye),),
                                  at=((*at),),
                                  up=((*up),))

    return OpenGLPerspectiveCameras(device=device,
                                    R=R,
                                    T=T,
                                    degrees=True,
                                    fov=moon_view.fov,
                                    znear=moon_view.znear,
                                    zfar=moon_view.zfar)
