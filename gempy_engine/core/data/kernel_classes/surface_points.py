from dataclasses import dataclass, field
from typing import Union

import numpy as np

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


