import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import SurfacePointsInternals, OrientationsInternals
from .kernel_classes.faults import FaultsData
from .options import KernelOptions


@dataclass
class SolverInput:
    sp_internal: SurfacePointsInternals
    ori_internal: OrientationsInternals
    xyz_to_interpolate: np.ndarray
    fault_internal:  Optional[FaultsData] = None
    _fault_internal:  Optional[FaultsData] = dataclasses.field(init=False, repr=False, default=None)

    debug = None
    @property
    def fault_internal(self):
        if self._fault_internal is None:
            empty_fault_values_on_sp = np.zeros((0, 0))
            empty_fault_values_on_grid = np.zeros((0, 0))
            return FaultsData(empty_fault_values_on_grid, empty_fault_values_on_sp)
        return self._fault_internal
    
    @fault_internal.setter
    def fault_internal(self, value):
        self._fault_internal = value
    
    
