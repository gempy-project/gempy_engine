from dataclasses import dataclass
from typing import Union, List, Optional, Dict

import numpy as np

from gempy_engine.config import DEFAULT_DTYPE
from gempy_engine.core.backend_tensor import BackendTensor


def _check_and_convert_list_to_array(field):
    if type(field) == list:
        field = np.array(field)
    return field


@dataclass
class RegularGrid:

    extent: Union[np.ndarray, List]
    regular_grid_shape: Union[np.ndarray, List]  # Shape(3)
    _active_cells: np.ndarray = None # Bool array

    def __post_init__(self):
        self.regular_grid_shape = _check_and_convert_list_to_array(self.regular_grid_shape)
        self.extent = _check_and_convert_list_to_array(self.extent)

        self.values = self._create_regular_grid(self.extent, self.regular_grid_shape)

    @classmethod
    def from_dxdydz(cls, values, extent, dxdydz):
        raise NotImplementedError
        regular_grid_shape = -1  # TODO: add logic here
        return cls(values, extent, regular_grid_shape)

    @property
    def active_cells(self):
        if self._active_cells is not None:
            return self._active_cells
        else:
            return np.ones(self.regular_grid_shape, dtype=bool)

    @active_cells.setter
    def active_cells(self, value):
        self._active_cells = value

    @property
    def resolution(self):
        return self.regular_grid_shape

    @property
    def dxdydz(self):
        extent = self.extent
        resolution = self.regular_grid_shape
        dx, dy, dz = self._compute_dxdydz(extent, resolution)
        return dx, dy, dz

    @property
    def d_diagonal(self):
        dx, dy,dz = self.dxdydz
        return np.sqrt(dx**2+dy**2+dz**2)

    @property
    def regular_grid_dx(self):
        return self.dxdydz[0]

    @property
    def regular_grid_dy(self):
        return self.dxdydz[1]

    @property
    def regular_grid_dz(self):
        return self.dxdydz[2]

    @property
    def values_vtk_format(self):
        extent = self.extent
        resolution = self.resolution + 1

        x = np.linspace(extent[0] , extent[1] , resolution[0], dtype="float64")
        y = np.linspace(extent[2] , extent[3] , resolution[1], dtype="float64")
        z = np.linspace(extent[4] , extent[5] , resolution[2], dtype="float64")
        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
        g = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T

        return g

    @property
    def corners_values(self):
        def _generate_corners(xyz_coord, dxdydz, level=1):
            x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
            dx, dy, dz = dxdydz

            def stack_left_right(a_edg, d_a):
                return np.stack((a_edg - d_a / level / 2, a_edg + d_a / level / 2), axis=1)

            x_ = np.repeat(stack_left_right(x_coord, dx), 4, axis=1)
            x = x_.ravel()
            y_ = np.tile(np.repeat(stack_left_right(y_coord, dy), 2, axis=1), (1, 2))
            y = y_.ravel()
            z_ = np.tile(stack_left_right(z_coord, dz), (1, 4))
            z = z_.ravel()

            new_xyz = np.stack((x, y, z)).T
            return new_xyz

        return _generate_corners(self.values, self.dxdydz)

    @property
    def faces_values(self):
        def _generate_faces(xyz_coord, dxdydz):
            x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
            dx, dy, dz = dxdydz

            x = np.array([[x_coord - dx / 2, x_coord + dx / 2],
                          [x_coord, x_coord],
                          [x_coord, x_coord]]).ravel()
            y = np.array([[y_coord, y_coord],
                          [y_coord - dy / 2, y_coord + dy / 2],
                          [y_coord, y_coord]]).ravel()
            z = np.array([[z_coord, z_coord],
                          [z_coord, z_coord],
                          [z_coord - dz / 2, z_coord + dz / 2]]).ravel()

            new_xyz = np.stack((x, y, z)).T
            return new_xyz

        return _generate_faces(self.values, self.dxdydz)

    @property
    def faces_values_3d(self):
        face_values = self.faces_values
        return face_values.reshape(*self.regular_grid_shape, 6, 3)

    @classmethod
    def _compute_dxdydz(cls, extent, resolution):
        dx = (extent[1] - extent[0]) / resolution[0]
        dy = (extent[3] - extent[2]) / resolution[1]
        dz = (extent[5] - extent[4]) / resolution[2]
        return dx, dy, dz

    @classmethod
    def _create_regular_grid(cls, extent, resolution):
        dx, dy, dz = cls._compute_dxdydz(extent, resolution)

        x = np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0], dtype=BackendTensor.default_dtype)
        y = np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1], dtype=BackendTensor.default_dtype)
        z = np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2], dtype=BackendTensor.default_dtype)
        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
        g = np.vstack((xv.ravel(), yv.ravel(), zv.ravel())).T

        return g

# TODO: [ ] values is independent field to regular grid. Proabably we want to have an extra field for them
# TODO: (custom_grid?) and then having a values as a property that brings both together?
@dataclass
class Grid:
    values: np.ndarray
    len_grids: Union[np.ndarray, List] = None # TODO: This should be a bit more automatic?
    regular_grid: RegularGrid = None
    custom_grid: Dict[str, np.ndarray] = None

    def __post_init__(self):
        if self.len_grids is None:
            self.len_grids = [self.values.shape[0]]

        self.len_grids = _check_and_convert_list_to_array(self.len_grids)

    @classmethod
    def from_regular_grid(cls, regular_grid: RegularGrid):
        return cls(regular_grid.values, regular_grid=regular_grid)

    @property
    def len_all_grids(self):
        return self.len_grids.sum(axis=0)

    @property
    def regular_grid_values(self) -> np.ndarray:  # shape(nx, ny, nz, 3)
        if self.len_grids[0] != self.regular_grid_shape.prod():
            raise ValueError("The values and the regular grid do not match.")
        return self.values[:self.len_grids[0]].reshape(*self.regular_grid_shape, 3)

    @property
    def custom_grid_values(self) -> np.ndarray:
        return self.values[self.len_grids[0]:self.len_grids[1]]

    @property
    def regular_grid_shape(self):
        return self.regular_grid.regular_grid_shape

    @property
    def dxdydz(self):
        return self.regular_grid.dxdydz

