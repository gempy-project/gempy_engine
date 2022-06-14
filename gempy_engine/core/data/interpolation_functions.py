import enum
import functools
from dataclasses import dataclass
from typing import Callable

import numpy as np


class InterpolationFunctions(enum.Enum):
    GAUSSIAN_PROCESS = enum.auto()
    CUSTOM = enum.auto()
    SPHERE = enum.auto()


@dataclass
class CustomInterpolationFunctions:
    scalar_field_at_surface_points: np.ndarray
    implicit_function: Callable
    gx_function: Callable
    gy_function: Callable
    gz_function: Callable

    @classmethod
    def from_builtin(cls, interpolation_function: InterpolationFunctions, scalar_field_at_surface_points: np.ndarray,
                     **kwargs):
        match interpolation_function:
            case InterpolationFunctions.SPHERE:
                # TODO: Move this block to a different module
                def implicit_sphere(xyz: np.ndarray, extent: np.ndarray):
                    x_dir = np.minimum(xyz[:, 0] - extent[0], extent[1] - xyz[:, 0])
                    y_dir = np.minimum(xyz[:, 1] - extent[2], extent[3] - xyz[:, 1])
                    z_dir = np.minimum(xyz[:, 2] - extent[4], extent[5] - xyz[:, 2])
                    return x_dir ** 2 + y_dir ** 2 + z_dir ** 2

                def gradient_sphere(xyz: np.ndarray, extent: np.ndarray, direction: str):
                    x_dir = np.minimum(xyz[:, 0] - extent[0], extent[1] - xyz[:, 0])
                    y_dir = np.minimum(xyz[:, 1] - extent[2], extent[3] - xyz[:, 1])
                    z_dir = np.minimum(xyz[:, 2] - extent[4], extent[5] - xyz[:, 2])
                    if direction == "x":
                        return 2 * x_dir
                    elif direction == "y":
                        return 2 * y_dir
                    elif direction == "z":
                        return 2 * z_dir

                implicit_sphere_function = functools.partial(implicit_sphere, extent=kwargs["extent"])
                gx_function = functools.partial(gradient_sphere, extent=kwargs["extent"], direction="x")
                gy_function = functools.partial(gradient_sphere, extent=kwargs["extent"], direction="y")
                gz_function = functools.partial(gradient_sphere, extent=kwargs["extent"], direction="z")

                return cls(scalar_field_at_surface_points, implicit_sphere_function, gx_function, gy_function, gz_function)
