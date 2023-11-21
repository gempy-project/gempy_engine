from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy import ndarray

from gempy_engine.core.backend_tensor import BackendTensor
from .centered_grid import CenteredGrid
from .generic_grid import GenericGrid
from .regular_grid import RegularGrid


# TODO: [ ] values is independent field to regular grid. Proabably we want to have an extra field for them
# TODO: (custom_grid?) and then having a values as a property that brings both together?
@dataclass
class Grid:
    regular_grid: RegularGrid = None
    custom_grid: Optional[GenericGrid] = None
    topography: Optional[GenericGrid] = None
    sections: Optional[GenericGrid] = None
    geophysics_grid: Optional[CenteredGrid] = None  # TODO: Not implemented this probably will need something different that the generic grid?

    debug_vals = None
    
    @classmethod
    def from_xyz_coords(cls, xyz_coords: ndarray) -> "Grid":
        return cls(custom_grid=GenericGrid(values=xyz_coords))

    @classmethod
    def from_regular_grid(cls, regular_grid: RegularGrid) -> "Grid":
        return cls(regular_grid=regular_grid)
    
    @property
    def values(self) -> np.ndarray:
        """Collect values from all associated grids."""
        # * The order is the same as in gempy v2 but hopefully this new way of doing things will be more flexible so order does not matter
        values = []
        if self.regular_grid is not None:
            values.append(self.regular_grid.values)
        if self.custom_grid is not None:
            values.append(self.custom_grid.values)
        if self.topography is not None:
            values.append(self.topography.values)
        if self.sections is not None:
            values.append(self.sections.values)
        if self.geophysics_grid is not None:
            values.append(self.geophysics_grid.values)
        
        values_array = BackendTensor.t.concatenate(values, dtype=BackendTensor.dtype)
        values_array = BackendTensor.t.array(values_array)
        
        return values_array


    @property
    def regular_grid_slice(self) -> slice:
        return slice(
            0,
            len(self.regular_grid) if self.regular_grid is not None else 0
        )

    @property
    def custom_grid_slice(self) -> slice:
        start = len(self.regular_grid) if self.regular_grid is not None else 0
        return slice(
            start,
            start + len(self.custom_grid) if self.custom_grid is not None else start
        )

    @property
    def topography_slice(self) -> slice:
        start = self.custom_grid_slice.stop
        return slice(
            start,
            start + len(self.topography) if self.topography is not None else start
        )

    @property
    def sections_slice(self) -> slice:
        start = self.topography_slice.stop
        return slice(
            start,
            start + len(self.sections) if self.sections is not None else start
        )

    @property
    def geophysics_grid_slice(self) -> slice:
        start = self.sections_slice.stop
        return slice(
            start,
            start + len(self.geophysics_grid) if self.geophysics_grid is not None else start
        )

    @property
    def len_all_grids(self) -> int:
        return self.values.shape[0]

    @property
    def regular_grid_values(self) -> np.ndarray:  # shape(nx, ny, nz, 3)
        return self.regular_grid.values.reshape(*self.regular_grid_shape, 3)

    @property
    def custom_grid_values(self) -> np.ndarray:
        return self.custom_grid.values

    @property
    def topography_values(self) -> np.ndarray:
        return self.topography.values

    @property
    def sections_values(self) -> np.ndarray:
        return self.sections.values

    @property
    def geophysics_grid_values(self) -> np.ndarray:
        return self.geophysics_grid.values

    @property
    def regular_grid_shape(self) -> ndarray | list:
        return self.regular_grid.regular_grid_shape

    @property
    def dxdydz(self) -> tuple[float, float, float]:
        return self.regular_grid.dxdydz
