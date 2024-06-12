from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import SurfacePointsInternals, OrientationsInternals
from .kernel_classes.faults import FaultsData
from ..backend_tensor import BackendTensor


@dataclass
class SolverInput(object):
    sp_internal: SurfacePointsInternals
    ori_internal: OrientationsInternals = field(init=False, hash=False)
    xyz_to_interpolate: Optional[np.ndarray] = field(init=False, hash=False)  # * it is optional if the instance is only used to create the cov
    _fault_internal: Optional[FaultsData] = field(init=False, hash=False)
    weights_x0: Optional[np.ndarray] = None

    debug = None

    def __init__(self, sp_internal: SurfacePointsInternals, ori_internal: OrientationsInternals,
                 xyz_to_interpolate: np.ndarray=None, fault_internal=None):
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
