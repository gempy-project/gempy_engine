from dataclasses import dataclass, field
from typing import Union

import numpy as np

from gempy_engine.modules.kernel_constructor._structs import tensor_types


@dataclass(frozen=False)
class SurfacePoints:
    sp_coords: np.ndarray
    nugget_effect_scalar: Union[np.ndarray, float]

    def __post_init__(self):
        if type(self.nugget_effect_scalar) is float or type(self.nugget_effect_scalar) is int:
            self.nugget_effect_scalar = np.ones(self.n_points) * self.nugget_effect_scalar

    def __hash__(self):
        return hash(5)

    @property
    def n_points(self):
        return self.sp_coords.shape[0]


@dataclass
class SurfacePointsInternals:
    ref_surface_points: tensor_types
    rest_surface_points: tensor_types
    nugget_effect_ref_rest: tensor_types

    @property
    def n_points(self):
        return self.ref_surface_points.shape[0]