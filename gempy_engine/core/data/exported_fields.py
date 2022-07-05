from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(init=True)
class ExportedFields:
    _scalar_field: Optional[np.ndarray]
    _gx_field: Optional[np.ndarray] = None
    _gy_field: Optional[np.ndarray] = None
    _gz_field: Optional[np.ndarray] = None
    
    n_points_per_surface: Optional[np.ndarray] = None
    slice_feature: Optional[slice] = slice(None, None)  # Slice all the surface points
    grid_size: Optional[int] = None
    
    _scalar_field_at_surface_points: Optional[np.ndarray] = None
    scalar_field_at_fault_shell: Optional[np.ndarray] = None
    
    debug = None
    
    @property
    def scalar_field_at_surface_points(self) -> Optional[np.ndarray]:
        if self.scalar_field_at_fault_shell is not None:  # * For now this has priority over everything else
            return self.scalar_field_at_fault_shell
        elif self._scalar_field_at_surface_points is None:
            scalar_field_at_all_sp = self._scalar_field[self.grid_size:]
            scalar_field_at_feature_sp = scalar_field_at_all_sp[self.slice_feature]
            scalar_field_at_one_point_per_surface = scalar_field_at_feature_sp[self.npf]
            return scalar_field_at_one_point_per_surface
        else:
            return self._scalar_field_at_surface_points
    
    @scalar_field_at_surface_points.setter
    def scalar_field_at_surface_points(self, value):
        self._scalar_field_at_surface_points = value
    
    @property
    def scalar_field(self):
        if self.slice_feature is None or self.slice_feature == 0:
            return self._scalar_field

        return self._scalar_field[:self.grid_size]
    
    @property
    def scalar_field_everywhere(self): return self._scalar_field
    
    @property
    def gx_field(self):
        return self._gx_field[:self.grid_size]

    @property
    def gy_field(self):
        return self._gy_field[:self.grid_size]

    @property
    def gz_field(self):
        return self._gz_field[:self.grid_size]

    @property
    def npf(self):
        return self.n_points_per_surface

