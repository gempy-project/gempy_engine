from dataclasses import dataclass, field
from typing import Union, List

import numpy as np

from ..backend_tensor import BackendTensor
from ..utils import _check_and_convert_list_to_array, cast_type_inplace
from .kernel_classes.server.input_parser import GridSchema


@dataclass(frozen=False)  # TODO: I want to do this class immutable
class RegularGrid:
    orthogonal_extent: Union[np.ndarray, List] #: This is always orthogonal extent. We use the transform to rotate it.
    regular_grid_shape: Union[np.ndarray, List]  # Shape(3)
    
    _active_cells: np.ndarray = field(default=None, repr=False, init=False)
    left_right: np.ndarray = field(default=None, repr=False, init=False)
    
    values: np.ndarray = field(default=None, repr=False, init=False)
    original_values: np.ndarray = field(default=None, repr=False, init=False)  #: When the regular grid is representing a octree level, only active cells are stored in values. This is the original values of the regular grid.
    
    def __len__(self):
        # return self.regular_grid_shape.prod()
        return self.values.shape[0]

    def __post_init__(self):
        self.regular_grid_shape = _check_and_convert_list_to_array(self.regular_grid_shape)
        self.orthogonal_extent = _check_and_convert_list_to_array(self.orthogonal_extent) + 1e-6  # * This to avoid some errors evaluating in 0 (e.g. bias in dual contouring)

        self._create_regular_grid_3d()

        self.original_values = BackendTensor.t.copy(self.values)

    @property
    def dx(self):
        return (self.orthogonal_extent[1] - self.orthogonal_extent[0]) / self.resolution[0]

    @property
    def dy(self):
        return (self.orthogonal_extent[3] - self.orthogonal_extent[2]) / self.resolution[1]

    @property
    def dz(self):
        return (self.orthogonal_extent[5] - self.orthogonal_extent[4]) / self.resolution[2]
    
    @property
    def x_coord(self):
        return np.linspace(self.orthogonal_extent[0] + self.dx / 2, self.orthogonal_extent[1] - self.dx / 2, self.resolution[0], dtype="float64")

    @property
    def y_coord(self):
        return np.linspace(self.orthogonal_extent[2] + self.dy / 2, self.orthogonal_extent[3] - self.dy / 2, self.resolution[1], dtype="float64")

    @property
    def z_coord(self):
        return np.linspace(self.orthogonal_extent[4] + self.dz / 2, self.orthogonal_extent[5] - self.dz / 2, self.resolution[2], dtype="float64")

    @classmethod
    def from_octree_level(cls, xyz_coords_octree: np.ndarray, previous_regular_grid: "RegularGrid",
                          active_cells: np.ndarray, left_right: np.ndarray) -> "RegularGrid":
       
        regular_grid_for_octree_level = cls(
            orthogonal_extent=previous_regular_grid.orthogonal_extent,
            regular_grid_shape=previous_regular_grid.regular_grid_shape * 2,
        )

        regular_grid_for_octree_level.values = xyz_coords_octree  # ! Overwrite the common values
        regular_grid_for_octree_level._active_cells = active_cells
        regular_grid_for_octree_level.left_right = left_right

        return regular_grid_for_octree_level

    @classmethod
    def from_dxdydz(cls, values, extent, dxdydz):
        raise NotImplementedError
        regular_grid_shape = -1  # TODO: add logic here
        return cls(values, extent, regular_grid_shape)

    @classmethod
    def from_schema(cls, schema: GridSchema):
        raise NotImplementedError("This method has to be updated")
        return cls(
            extent=schema.orthogonal_extent,
            regular_grid_shape=[2, 2, 2],  # ! This needs to be generalized. For now I hardcoded the octree initial shapes
            left_right=None
        )

    @property
    def active_cells(self) -> np.ndarray:
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
        extent = self.orthogonal_extent
        resolution = self.regular_grid_shape
        dx, dy, dz = self._compute_dxdydz(extent, resolution)
        return dx, dy, dz

    @property
    def d_diagonal(self):
        dx, dy, dz = self.dxdydz
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

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
    def values_vtk_format(self) -> np.ndarray:
        extent = self.orthogonal_extent
        resolution = self.resolution + 1

        x = np.linspace(extent[0], extent[1], resolution[0], dtype="float64")
        y = np.linspace(extent[2], extent[3], resolution[1], dtype="float64")
        z = np.linspace(extent[4], extent[5], resolution[2], dtype="float64")
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


    def _create_regular_grid_3d(self):
        coords = self.x_coord, self.y_coord, self.z_coord

        g = np.meshgrid(*coords, indexing="ij")
        values = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")
        values = np.ascontiguousarray(values)
        values = BackendTensor.tfnp.array(values, dtype=BackendTensor.dtype_obj)

        self.values = values
