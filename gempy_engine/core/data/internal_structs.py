from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import SurfacePointsInternals, OrientationsInternals
from .kernel_classes.faults import FaultsData
from ...config import TENSOR_DTYPE


@dataclass
class SolverInput:
    sp_internal: SurfacePointsInternals
    ori_internal: OrientationsInternals
    xyz_to_interpolate: Optional[np.ndarray] = None  # * it is optional if the instance is only used to create the cov
    _fault_internal:  Optional[FaultsData] = None

    debug = None
    
    def __init__(self, sp_internal, ori_internal, xyz_to_interpolate=None, fault_internal=None):
        self.sp_internal = sp_internal
        self.ori_internal = ori_internal
        self.xyz_to_interpolate = xyz_to_interpolate.astype(TENSOR_DTYPE)
        self._fault_internal = fault_internal
    
    @property
    def fault_internal(self):
        if self._fault_internal is None:
            empty_fault_values_on_sp = np.zeros((0, 0), dtype=TENSOR_DTYPE)
            empty_fault_values_on_grid = np.zeros((0, 0), dtype=TENSOR_DTYPE)
            return FaultsData(empty_fault_values_on_grid, empty_fault_values_on_sp)
        return self._fault_internal
    
    @fault_internal.setter
    def fault_internal(self, value):
        self._fault_internal = value
    
    
