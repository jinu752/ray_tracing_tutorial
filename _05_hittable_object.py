from __future__ import annotations
from typing import List
import sys

import numpy as np
import numpy.linalg as LA

infinity = sys.float_info.max


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


class HitRecord:
    def __init__(self, point: np.array, normal: np.array, t: float) -> None:
        self.point = point
        self.normal = normal
        self.t = t

    def set_face_normal(self, ray: Ray, outward_normal: np.array) -> None:
        is_front_face = np.dot(ray.direction, outward_normal) < 0
        self.normal = outward_normal if is_front_face else -outward_normal


class Hittable:
    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord:
        raise NotImplementedError


class Sphere(Hittable):
    def __init__(self) -> None:
        self.center = np.zeros(3)
        self.radius = 1.0

    def __init__(self, center: np.array, radius: float) -> None:
        self.center = center
        self.radius = radius

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord:
        dir_center_to_origin = ray.origin - self.center

        a = np.dot(ray.direction, ray.direction)
        half_b = np.dot(dir_center_to_origin, ray.direction)
        c = np.dot(dir_center_to_origin, dir_center_to_origin) - self.radius**2.0

        discriminant = half_b**2 - a * c
        if discriminant < 0:
            return None
        sqrt_d = np.sqrt(discriminant)

        # find the nearest root that lies in the acceptable range.
        t = (-half_b - sqrt_d) / a
        if t < t_min or t_max < t:
            t = (-half_b + sqrt_d) / a
            if t < t_min or t_max < t:
                return None

        point = ray.at(t)
        outward_normal = (point - self.center) / self.radius
        hit_record = HitRecord(point=point, normal=outward_normal, t=t)
        hit_record.set_face_normal(ray=ray, outward_normal=outward_normal)

        return hit_record


class HittableList(Hittable):
    def __init__(self) -> None:
        self.objects: List[Hittable] = []

    def clear(self) -> None:
        self.objects.clear()

    def add(self, object: Hittable) -> None:
        self.objects.append(object)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord:
        record = None
        closest_so_far = t_max

        for object in self.objects:
            tmp_record = object.hit(ray=ray, t_min=t_min, t_max=closest_so_far)
            if tmp_record is not None:
                closest_so_far = tmp_record.t
                record = tmp_record

        return record


def ray_color(ray: Ray, world: Hittable) -> np.array:
    color = np.ones(3)
    record = world.hit(ray=ray, t_min=0.0, t_max=infinity)
    if record is not None:
        return 0.5 * (record.normal + color)

    t = 0.5 * (ray.direction[1] + 1.0)
    color1 = np.array([1.0, 1.0, 1.0])
    color2 = np.array([0.5, 0.7, 1.0])
    return blend(color1=color1, color2=color2, t=t)


def compute_ray_color(
    uv, camera_origin, horizontal_vec, vertical_vec, lower_left_corner, world
):
    (u, v) = uv
    ray = Ray(
        origin=camera_origin,
        direction=(lower_left_corner + u * horizontal_vec + v * vertical_vec)
        - camera_origin,
    )

    return ray_color(ray=ray, world=world)
