import torch
from pytorch3d.renderer import PointLights, DirectionalLights
from model import MoonLight


def load_lights(moon_light: MoonLight):
    device = torch.device('cuda')
    direction = moon_light.position_light[:3]
    ambient_color = moon_light.ambient_light[: 3]
    diffuse_color = moon_light.diffuse_light[: 3]
    specular_color = (0.0, 0.0, 0.0)

    return DirectionalLights(device=device,
                             direction=(direction,),
                             ambient_color=(ambient_color,),
                             diffuse_color=(diffuse_color,),
                             specular_color=(specular_color,))
