import copy
from typing import List

from ...core.data.generic_grid import GenericGrid
from ...core.data.options import KernelOptions, InterpolationOptions
from ...core.data.octree_level import OctreeLevel
from ...core.data.interp_output import InterpOutput
from ...core.data.grid import Grid
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput

from ._multi_scalar_field_manager import interpolate_all_fields

import numpy as np


def interpolate_on_octree(interpolation_input: InterpolationInput, options: InterpolationOptions,
                          data_shape: InputDataDescriptor) -> OctreeLevel:
    interpolation_input = copy.deepcopy(interpolation_input)
    
    grid_0_centers = interpolation_input.grid
    output_0_centers: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, data_shape)  # interpolate - centers

    # Interpolate - corners
    if options.compute_corners:
        grid_0_corners = Grid(
            custom_grid=GenericGrid(
                values=(_generate_corners(grid_0_centers.values, grid_0_centers.dxdydz))
            )
        )
        interpolation_input.grid = grid_0_corners
        output_0_corners: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, data_shape)  # * This is unnecessary for the last level except for Dual contouring
    else:
        output_0_corners = []
        grid_0_corners = None
    
    # Set values to octree
    next_octree_level = OctreeLevel(grid_0_centers, grid_0_corners, output_0_centers, output_0_corners)
    return next_octree_level


def _generate_corners(xyz_coord, dxdydz, level=1) -> np.ndarray:
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
    return np.ascontiguousarray(new_xyz)
