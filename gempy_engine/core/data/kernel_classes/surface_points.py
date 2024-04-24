from dataclasses import dataclass, field
from typing import Union, Optional

import numpy as np

from .server.input_parser import SurfacePointsSchema
from ..stacks_structure import StacksStructure
from ...utils import cast_type_inplace
from ...backend_tensor import BackendTensor
from gempy_engine.modules.kernel_constructor._structs import tensor_types


@dataclass(frozen=False)
class SurfacePoints:
    sp_coords: np.ndarray
    nugget_effect_scalar: Union[np.ndarray, float] = 0.000001

    # TODO (Sep 2022): Pretty sure this has to be private
    slice_feature: Optional[slice] = field(default_factory=lambda: slice(None, None))  # * Used to slice the surface points values of the interpolation (grid.values)

    def __post_init__(self):
        if type(self.nugget_effect_scalar) is float or type(self.nugget_effect_scalar) is int:
            self.nugget_effect_scalar = np.ones(self.n_points) * self.nugget_effect_scalar
        cast_type_inplace(self, requires_grad=BackendTensor.COMPUTE_GRADS)  # TODO: This has to be grabbed from options

    @classmethod
    def from_schema(cls, schema: SurfacePointsSchema):
        return cls(sp_coords=np.array(schema.sp_coords))

    @classmethod
    def from_suraface_points_subset(cls, surface_points: "SurfacePoints", data_structure: StacksStructure):
        stack_n = data_structure.stack_number
        cum_sp_l0 = data_structure.nspv_stack[stack_n]  # .sum()
        cum_sp_l1 = data_structure.nspv_stack[stack_n + 1]  # .sum()
        # TODO: Add nugget selection

        sp = SurfacePoints(
            sp_coords=surface_points.sp_coords[cum_sp_l0:cum_sp_l1],
            nugget_effect_scalar=surface_points.nugget_effect_scalar[cum_sp_l0:cum_sp_l1],
            slice_feature=slice(cum_sp_l0, cum_sp_l1)
        )

        return sp

    @property
    def n_points(self):
        return self.sp_coords.shape[0]


@dataclass(frozen=True)
class SurfacePointsInternals:
    ref_surface_points: tensor_types
    rest_surface_points: tensor_types
    nugget_effect_ref_rest: tensor_types

    def __hash__(self):
        i = hash(self.__repr__())
        return i

    @property
    def n_points(self) -> int:
        return self.ref_surface_points.shape[0]
