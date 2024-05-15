import copy
from typing import List, Optional

from ...core.backend_tensor import BackendTensor
from ...config import NOT_MAKE_INPUT_DEEP_COPY, AvailableBackends
from ...core.data.generic_grid import GenericGrid
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
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH and NOT_MAKE_INPUT_DEEP_COPY is False:
        temp_interpolation_input = copy.deepcopy(interpolation_input)
    else:
        temp_interpolation_input = interpolation_input

    # * Interpolate - centers
    output_0_centers: List[InterpOutput] = interpolate_all_fields(temp_interpolation_input, options, data_shape)  # interpolate - centers

    # * Interpolate - corners
    grid_0_centers: EngineGrid = temp_interpolation_input.grid  # ? This could be moved to the next section
    if options.compute_corners:
        grid_0_corners: Optional[EngineGrid] = EngineGrid.from_xyz_coords(
            xyz_coords=_generate_corners(regular_grid=grid_0_centers.octree_grid)
        )

        # ! Here we need to swap the grid temporarily but it is
        # ! important to set up the og grid back for the gradients
        temp_interpolation_input.set_temp_grid(grid_0_corners)  # * Prepare grid for next interpolation
        output_0_corners: List[InterpOutput] = interpolate_all_fields(  # * This is unnecessary for the last level except for Dual contouring
            interpolation_input=temp_interpolation_input,
            options=options,
            data_descriptor=data_shape
        )

        temp_interpolation_input.set_grid_to_original()
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
    if regular_grid is None: raise ValueError("Regular grid is None")

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
