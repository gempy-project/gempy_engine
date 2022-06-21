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
    
    @property
    def scalar_field_at_surface_points(self) -> Optional[np.ndarray]:
        if self._scalar_field_at_surface_points is None:
            npf_ = self._scalar_field[self.slice_feature][self.npf]
            return npf_
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

