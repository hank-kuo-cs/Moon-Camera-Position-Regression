import torch
from pytorch3d.renderer import look_at_view_transform, OpenGLPerspectiveCameras
from .....config import config


def load_cameras(dist, elev, azim, at, up):
    device = torch.device('cuda')

    R, T = look_at_view_transform(dist=dist,
                                  elev=elev,
                                  azim=azim,
                                  at=((at[0], at[1], at[2]),),
                                  up=((up[0], up[1], up[2]),))

    return OpenGLPerspectiveCameras(device=device,
                                    R=R,
                                    T=T,
                                    degrees=True,
                                    fov=config.generate.fov,
                                    znear=config.generate.znear,
                                    zfar=config.generate.zfar)
