from dataclasses import dataclass
from typing import Union, List, Optional, Dict

import numpy as np


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
        self.values = self._create_regular_grid(self.extent, self.regular_grid_shape)

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

    @classmethod
    def _compute_dxdydz(cls, extent, resolution):
        dx = (extent[1] - extent[0]) / resolution[0]
        dy = (extent[3] - extent[2]) / resolution[1]
        dz = (extent[5] - extent[4]) / resolution[2]
        return dx, dy, dz

    @classmethod
    def from_dxdydz(cls, values, extent, dxdydz):
        raise NotImplementedError
        regular_grid_shape = -1  # TODO: add logic here
        return cls(values, extent, regular_grid_shape)

    # TODO: This should be the constructor?
    # @classmethod
    # def init_regular_grid(cls, extent, regular_grid_shape):
    #     values = cls._create_regular_grid(extent, regular_grid_shape)
    #     return cls(values, extent, regular_grid_shape)

    @classmethod
    def _create_regular_grid(cls, extent, resolution):
        dx, dy, dz = cls._compute_dxdydz(extent, resolution)

        x = np.linspace(extent[0] + dx / 2, extent[1] - dx / 2, resolution[0], dtype="float64")
        y = np.linspace(extent[2] + dy / 2, extent[3] - dy / 2, resolution[1], dtype="float64")
        z = np.linspace(extent[4] + dz / 2, extent[5] - dz / 2, resolution[2], dtype="float64")
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
        def _generate_faces(xyz_coord, dxdydz, level=1):
            x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
            dx, dy, dz = dxdydz

            x = np.array([[x_coord - dx/2, x_coord + dx/2],
                          [x_coord, x_coord],
                          [x_coord, x_coord]]).ravel()
            y = np.array([[y_coord, y_coord],
                          [y_coord - dy/2, y_coord + dy/2],
                          [y_coord, y_coord]]).ravel()
            z = np.array([[z_coord, z_coord],
                          [z_coord, z_coord],
                          [z_coord - dz / 2, z_coord + dz / 2]]).ravel()

            new_xyz = np.stack((x, y, z)).T
            return new_xyz

        if self.active_cells is None:
            voxels = self.values
        else:
            voxels = self.values[self.active_cells]

        return _generate_faces(voxels, self.dxdydz)

    @property
    def faces_values_3d(self):
        face_values = self.faces_values
        return face_values.reshape(*self.regular_grid_shape, 6, 3)



# TODO: At the moment values is independent of regular grid and custom grid. What we need is:
# TODO: [ ] values is a property depending on regular grid and custom grid
# TODO: [ ] Regular grids can be set but inactive
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

    @property
    def regular_grid_dx(self):
        return self.dxdydz[0]

    @property
    def regular_grid_dy(self):
        return self.dxdydz[1]

    @property
    def regular_grid_dz(self):
        return self.dxdydz[2]
