import numpy as np
import numpy.linalg as LA


def normalize(vec: np.array, eps: float = 1e-6):
    return vec / (LA.norm(vec) + eps)


def blend(color1: np.array, color2: np.array, t: float):
    return (1.0 - t) * color1 + t * color2


class Ray:
    def __init__(self, origin: np.array, direction: np.array) -> None:
        self.origin = origin
        self.direction = normalize(direction)

    def at(self, t: float) -> np.array:
        return self.origin + t * self.direction


def ray_color(ray: Ray) -> np.array:
    t = 0.5 * (ray.direction[1] + 1.0)
    color1 = np.array([1.0, 1.0, 1.0])
    color2 = np.array([0.5, 0.7, 1.0])
    return blend(color1=color1, color2=color2, t=t)


def compute_ray_color(
    uv, camera_origin, horizontal_vec, vertical_vec, lower_left_corner
):

    (u, v) = uv
    ray = Ray(
        origin=camera_origin,
        direction=(lower_left_corner + u * horizontal_vec + v * vertical_vec)
        - camera_origin,
    )

    return ray_color(ray)
