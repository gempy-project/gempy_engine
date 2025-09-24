import warnings

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
class EngineGrid:
    octree_grid: Optional[RegularGrid] = None  #: This is None at least while we are interpolating branch
    dense_grid: Optional[RegularGrid] = None
    custom_grid: Optional[GenericGrid] = None
    topography: Optional[GenericGrid] = None
    sections: Optional[GenericGrid] = None
    geophysics_grid: Optional[CenteredGrid] = None  # TODO: Not implemented this probably will need something different that the generic grid?
    corners_grid: Optional[GenericGrid] = None  # TODO: Not implemented this probably will need something different that the generic grid?

    debug_vals = None

    # ? Should we add the number of octrees here instead of the general options

    def __init__(self, octree_grid: Optional[RegularGrid] = None, dense_grid: Optional[RegularGrid] = None,
                 custom_grid: Optional[GenericGrid] = None, topography: Optional[GenericGrid] = None,
                 sections: Optional[GenericGrid] = None, geophysics_grid: Optional[CenteredGrid] = None,
                 corners_grid: Optional[GenericGrid] = None):
        self.octree_grid = octree_grid
        self.dense_grid = dense_grid
        self.custom_grid = custom_grid
        self.topography = topography
        self.sections = sections
        self.geophysics_grid = geophysics_grid
        self.corners_grid = corners_grid

    @property
    def regular_grid(self):
        warnings.warn("This is deprecated. Use dense_grid instead", DeprecationWarning)
        if self.dense_grid is not None and self.octree_grid is not None:
            raise AttributeError('Both dense_grid and octree_grid are active. This is not possible.')
        elif self.dense_grid is not None:
            return self.dense_grid
        elif self.octree_grid is not None:
            return self.octree_grid
        else:
            return None


    @classmethod
    def from_xyz_coords(cls, xyz_coords: ndarray) -> "EngineGrid":
        return cls(custom_grid=GenericGrid(values=xyz_coords))

    @classmethod
    def from_regular_grid(cls, regular_grid: RegularGrid) -> "EngineGrid":
        return cls(
            dense_grid=regular_grid,
            octree_grid=RegularGrid(regular_grid.orthogonal_extent, np.array([2, 2, 2]))
        )

    @property
    def values(self) -> np.ndarray:
        """Collect values from all associated grids."""
        # * The order is the same as in gempy v2 but hopefully this new way of doing things will be more flexible so order does not matter
        values = []
        if self.octree_grid is not None:
            values.append(self.octree_grid.values)
        if self.dense_grid is not None:
            values.append(self.dense_grid.values)
        if self.custom_grid is not None:
            values.append(self.custom_grid.values)
        if self.topography is not None:
            values.append(self.topography.values)
        if self.sections is not None:
            values.append(self.sections.values)
        if self.geophysics_grid is not None:
            values.append(self.geophysics_grid.values)
        if self.corners_grid is not None:
            values.append(self.corners_grid.values)

        values_array = BackendTensor.t.concatenate(values, dtype=BackendTensor.dtype)
        values_array = BackendTensor.t.array(values_array, dtype=BackendTensor.dtype)

        return values_array

    @property
    def octree_grid_slice(self) -> slice:
        return slice(
            0,
            len(self.octree_grid) if self.octree_grid is not None else 0
        )

    @property
    def dense_grid_slice(self) -> slice:
        start = self.octree_grid_slice.stop
        return slice(
            start,
            start + len(self.dense_grid) if self.dense_grid is not None else start
        )

    @property
    def custom_grid_slice(self) -> slice:
        start = self.dense_grid_slice.stop
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
    def corners_grid_slice(self) -> slice:
        start = self.geophysics_grid_slice.stop
        return slice(
            start,
            start + len(self.corners_grid) if self.corners_grid is not None else start
        )

    @property
    def len_all_grids(self) -> int:
        return self.values.shape[0]

    @property
    def octree_grid_values(self) -> np.ndarray:  # shape(nx, ny, nz, 3)
        return self.octree_grid.values.reshape(*self.octree_grid_shape, 3)

    @property
    def dense_grid_values(self) -> np.ndarray:
        return self.dense_grid.values.reshape(*self.dense_grid_shape, 3)

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
    def octree_grid_shape(self) -> ndarray | list:
        return self.octree_grid.regular_grid_shape

    @property
    def dense_grid_shape(self) -> ndarray | list:
        return self.dense_grid.regular_grid_shape

    @property
    def octree_dxdydz(self) -> tuple[float, float, float]:
        return self.octree_grid.dxdydz
