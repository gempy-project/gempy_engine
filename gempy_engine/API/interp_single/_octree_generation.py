import copy
from typing import List

from ...core.backend_tensor import BackendTensor
from ...config import COMPUTE_GRADIENTS, AvailableBackends
from ...core.data.regular_grid import RegularGrid
from ...core.data.options import InterpolationOptions
from ...core.data.octree_level import OctreeLevel
from ...core.data.interp_output import InterpOutput
from ...core.data.engine_grid import EngineGrid
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput

from ._multi_scalar_field_manager import interpolate_all_fields

import numpy as np


def interpolate_on_octree(interpolation_input: InterpolationInput, options: InterpolationOptions,
                          data_shape: InputDataDescriptor) -> OctreeLevel:
    
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH and COMPUTE_GRADIENTS is False:
        temp_interpolation_input = copy.deepcopy(interpolation_input)
    else:
        temp_interpolation_input = interpolation_input
    
    # * Interpolate - centers
    output_0_centers: List[InterpOutput] = interpolate_all_fields(temp_interpolation_input, options, data_shape)  # interpolate - centers

    # * Interpolate - corners
    grid_0_centers: EngineGrid = temp_interpolation_input.grid  # ? This could be moved to the next section
    if options.compute_corners:
        grid_0_corners = EngineGrid.from_xyz_coords(
            xyz_coords=_generate_corners(regular_grid=grid_0_centers.octree_grid)
        )
        temp_interpolation_input.grid = grid_0_corners  # * Prepare grid for next interpolation
        output_0_corners: List[InterpOutput] = interpolate_all_fields(temp_interpolation_input, options, data_shape)  # * This is unnecessary for the last level except for Dual contouring
    else:
        output_0_corners = []
        grid_0_corners = None
    
    # * Create next octree level
    next_octree_level = OctreeLevel(
        grid_centers=grid_0_centers,
        grid_corners=grid_0_corners,
        outputs_centers=output_0_centers, 
        outputs_corners=output_0_corners
    )
    return next_octree_level


def _generate_corners(regular_grid: RegularGrid, level=1) -> np.ndarray:
    xyz_coord = regular_grid.values
    dxdydz = regular_grid.dxdydz
    
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
