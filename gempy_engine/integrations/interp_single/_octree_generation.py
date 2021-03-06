from ...core import data
from ...core.data.exported_structs import OctreeLevel
from ...core.data.grid import Grid
from ...core.data.interpolation_input import InterpolationInput
import numpy as np

from ._interp_single_internals import interpolate


def interpolate_on_octree(octree: OctreeLevel, interpolation_input: InterpolationInput,
                          options: data.InterpolationOptions,
                          data_shape: data.TensorsStructure) -> OctreeLevel:

    grid_0_centers = interpolation_input.grid
    # interpolate - centers

    output_0_centers = interpolate(interpolation_input, options, data_shape,
                                                 clean_buffer=False)

    # Interpolate - corners
    grid_0_corners = Grid(_generate_corners(grid_0_centers.values, grid_0_centers.dxdydz))
    interpolation_input.grid = grid_0_corners

    output_0_corners = interpolate(interpolation_input, options, data_shape,
                                                 clean_buffer=False)  # TODO: This is unnecessary for the last level except for Dual contouring

    # Set values to octree
    octree.set_interpolation_values(grid_0_centers, grid_0_corners,
                                    output_0_centers, output_0_corners)
    return octree


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