from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(init=True)
class ExportedFields:
    _scalar_field: Optional[np.ndarray]
    _gx_field: Optional[np.ndarray] = None
    _gy_field: Optional[np.ndarray] = None
    _gz_field: Optional[np.ndarray] = None
    n_points_per_surface: Optional[np.ndarray] = None
    n_surface_points: Optional[int] = 0
    
    _scalar_field_at_surface_points: Optional[np.ndarray] = None
    
    @property
    def scalar_field_at_surface_points(self):
        if self._scalar_field_at_surface_points is None:
            return self._scalar_field[-self.n_surface_points:][self.npf]
        else:
            return self._scalar_field_at_surface_points
    
    @scalar_field_at_surface_points.setter
    def scalar_field_at_surface_points(self, value):
        self._scalar_field_at_surface_points = value
    
    @property
    def scalar_field(self):
        if self.n_surface_points is None or self.n_surface_points == 0:
            return self._scalar_field

        return self._scalar_field[:-self.n_surface_points]
    
    @property
    def scalar_field_everywhere(self): return self._scalar_field
    
    @property
    def gx_field(self):
        if self.n_surface_points is None or self.n_surface_points == 0:
            return self._gx_field
        return self._gx_field[:-self.n_surface_points]

    @property
    def gy_field(self):
        if self.n_surface_points is None or self.n_surface_points == 0:
            return self._gy_field
        return self._gy_field[:-self.n_surface_points]

    @property
    def gz_field(self):
        if self.n_surface_points is None or self.n_surface_points == 0:
            return self._gz_field
        return self._gz_field[:-self.n_surface_points]

    @property
    def npf(self):
        return self.n_points_per_surface

    @classmethod
    def from_interpolation(cls, scalar_field, gx_field, gy_field, gz_field, grid_size: int):
        return cls(scalar_field[:grid_size], gx_field[:grid_size], gy_field[:grid_size], gz_field[:grid_size])
