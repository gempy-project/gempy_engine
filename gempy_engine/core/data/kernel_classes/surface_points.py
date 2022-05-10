from dataclasses import dataclass, field
from typing import Union

import numpy as np

from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.input_data_descriptor import StacksStructure
from gempy_engine.modules.kernel_constructor._structs import tensor_types


@dataclass(frozen=False)
class SurfacePoints:
    sp_coords: np.ndarray
    nugget_effect_scalar: Union[np.ndarray, float] = 0.0000001

    def __post_init__(self):
        if type(self.nugget_effect_scalar) is float or type(self.nugget_effect_scalar) is int:
            self.nugget_effect_scalar = np.ones(self.n_points) * self.nugget_effect_scalar

    def __hash__(self):
        return hash(5) # TODO: These should be self.__repr__ instead of 5
    
    @classmethod
    def from_suraface_points_subset(cls, surface_points: "SurfacePoints", data_structure: StacksStructure):
        stack_n = data_structure.stack_number
        cum_sp_l0 = data_structure.nspv_stack[:stack_n + 1].sum()
        cum_sp_l1 = data_structure.nspv_stack[:stack_n + 2].sum()
        
        # TODO: Add nugget selection
        sp = SurfacePoints(surface_points.sp_coords[cum_sp_l0:cum_sp_l1])
        return sp

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