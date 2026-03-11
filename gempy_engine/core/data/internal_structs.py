from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import SurfacePointsInternals, OrientationsInternals, TensorsStructure
from .interpolation_input import InterpolationInput
from .kernel_classes.faults import FaultsData
from ..backend_tensor import BackendTensor
from ...modules.data_preprocess import data_preprocess_interface


@dataclass
class SolverInput(object):
    sp_internal: SurfacePointsInternals
    ori_internal: OrientationsInternals = field(init=False, hash=False)
    xyz_to_interpolate: Optional[np.ndarray] = field(init=False, hash=False)  # * it is optional if the instance is only used to create the cov
    _fault_internal: Optional[FaultsData] = field(init=False, hash=False)
    weights_x0: Optional[np.ndarray] = None

    debug = None

    def __init__(self, sp_internal: SurfacePointsInternals, ori_internal: OrientationsInternals,
                 xyz_to_interpolate: np.ndarray = None, fault_internal=None):
        self.sp_internal = sp_internal
        self.ori_internal = ori_internal
        if xyz_to_interpolate is not None and xyz_to_interpolate.dtype != BackendTensor.dtype_obj:
            self.xyz_to_interpolate = xyz_to_interpolate.astype(BackendTensor.dtype)
        else:
            self.xyz_to_interpolate = xyz_to_interpolate
        self._fault_internal = fault_internal

    def __hash__(self):
        # xyz_to_interpolate and _faults are dependent on the octree levels
        combined = hash((self.sp_internal, self.ori_internal))
        return combined

    # 
    @property
    def fault_internal(self):
        if self._fault_internal is None:
            empty_fault_values_on_sp = np.zeros((0, 0), dtype=BackendTensor.dtype)
            empty_fault_values_on_grid = np.zeros((0, 0), dtype=BackendTensor.dtype)
            return FaultsData(empty_fault_values_on_grid, empty_fault_values_on_sp)
        return self._fault_internal

    @fault_internal.setter
    def fault_internal(self, value):
        self._fault_internal = value


@dataclass
class SolverInput_v2(object):
    sp_internal: SurfacePointsInternals
    ori_internal: OrientationsInternals = field(init=False, hash=False)
    xyz_to_interpolate: Optional[np.ndarray] = field(init=False, hash=False)  # * it is optional if the instance is only used to create the cov
    _fault_internal: Optional[FaultsData] = field(init=False, hash=False)
    weights_x0: Optional[np.ndarray] = None

    debug = None

    def __init__(self, sp_internal: SurfacePointsInternals, ori_internal: OrientationsInternals,
                 # xyz_to_interpolate: np.ndarray=None, 
                 fault_internal=None
                 ):
        self.sp_internal = sp_internal
        self.ori_internal = ori_internal
        # if xyz_to_interpolate is not None and xyz_to_interpolate.dtype != BackendTensor.dtype_obj:
        #     self.xyz_to_interpolate = xyz_to_interpolate.astype(BackendTensor.dtype)
        # else:
        #     self.xyz_to_interpolate = xyz_to_interpolate
        self._fault_internal = fault_internal

    def __hash__(self):
        # xyz_to_interpolate and _faults are dependent on the octree levels
        combined = hash((self.sp_internal, self.ori_internal))
        return combined

    @property
    def fault_internal(self):
        if self._fault_internal is None:
            empty_fault_values_on_sp = np.zeros((0, 0), dtype=BackendTensor.dtype)
            empty_fault_values_on_grid = np.zeros((0, 0), dtype=BackendTensor.dtype)
            return FaultsData(empty_fault_values_on_grid, empty_fault_values_on_sp)
        return self._fault_internal

    @fault_internal.setter
    def fault_internal(self, value):
        self._fault_internal = value


@dataclass
class EvaluatorInput:
    solver_input: SolverInput_v2
    xyz_to_interpolate: Optional[np.ndarray] = field(init=False, hash=False)  # * it is optional if the instance is only used to create the cov

    _n_points_per_surface: Optional[np.ndarray] = None
    _slice_feature: Optional[slice] = field(default_factory=lambda: slice(None, None))  # Slice all the surface points
    _grid_size: Optional[int] = None

    def __init__(self,
                 solver_input: SolverInput_v2,
                 interpolation_input: InterpolationInput,
                 tensor_struct: TensorsStructure,
                 only_surface_points: bool = False
                 ):
        self.solver_input = solver_input

        if only_surface_points:
            xyz_to_interpolate = interpolation_input.all_surface_points
        else:
            xyz_to_interpolate: np.ndarray = data_preprocess_interface.prepare_grid(
                grid=interpolation_input.grid.values,
                surface_points=interpolation_input.all_surface_points
            )

        if xyz_to_interpolate is not None and xyz_to_interpolate.dtype != BackendTensor.dtype_obj:
            self.xyz_to_interpolate = xyz_to_interpolate.astype(BackendTensor.dtype)
        else:
            self.xyz_to_interpolate = xyz_to_interpolate

        self._n_points_per_surface = tensor_struct.reference_sp_position
        self._slice_feature = interpolation_input.slice_feature
        self._grid_size = interpolation_input.grid.len_all_grids

    @property
    def sp_internal(self):
        return self.solver_input.sp_internal

    @property
    def ori_internal(self):
        return self.solver_input.ori_internal

    @property
    def fault_internal(self):
        return self.solver_input.fault_internal


@dataclass
class SegmentationInput:
    unit_values: np.ndarray
    sigmoid_slope: float | np.ndarray
    # segmentation_functions_per_stack: Optional[Callable[[np.ndarray], float]]
