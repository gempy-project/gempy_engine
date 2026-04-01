import copy
from typing import List

import numpy as np

from ._multi_scalar_field_manager import interpolate_all_fields
from ...config import NOT_MAKE_INPUT_DEEP_COPY, AvailableBackends
from ...core.backend_tensor import BackendTensor
from ...core.data.engine_grid import EngineGrid
from ...core.data.generic_grid import GenericGrid
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interp_output import InterpOutput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.octree_level import OctreeLevel
from ...core.data.options import InterpolationOptions
from ...core.data.regular_grid import RegularGrid


def interpolate_on_octree(interpolation_input: InterpolationInput, options: InterpolationOptions,
                          data_shape: InputDataDescriptor) -> OctreeLevel:
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH and NOT_MAKE_INPUT_DEEP_COPY is False:
        temp_interpolation_input = copy.deepcopy(interpolation_input)
    else:
        temp_interpolation_input = interpolation_input

    # * Interpolate - corners
    if options.compute_corners:
        grid_0_centers: EngineGrid = temp_interpolation_input.grid  # ? This could be moved to the next section
        xyz_corners = _generate_corners(regular_grid=grid_0_centers.octree_grid)

        corner_grid = GenericGrid(values=xyz_corners)
        grid_0_centers.corners_grid = corner_grid
        output_0_centers: List[InterpOutput] = interpolate_all_fields(temp_interpolation_input, options, data_shape)  # interpolate - centers

        # * Create next octree level
        next_octree_level = OctreeLevel(
            grid=grid_0_centers,
            outputs=output_0_centers,
        )
    else:
        grid_0_centers: EngineGrid = temp_interpolation_input.grid  # ? This could be moved to the next section
        output_0_centers: List[InterpOutput] = interpolate_all_fields(temp_interpolation_input, options, data_shape)  # interpolate - centers

        # * Create next octree level
        next_octree_level = OctreeLevel(
            grid=grid_0_centers,
            outputs=output_0_centers,
        )

    return next_octree_level


def _generate_corners(regular_grid: RegularGrid, level=1):
    if regular_grid is None:
        raise ValueError("Regular grid is None")

    xyz_coord = regular_grid.values
    dxdydz = regular_grid.dxdydz

    x_coord, y_coord, z_coord = xyz_coord[:, 0], xyz_coord[:, 1], xyz_coord[:, 2]
    dx, dy, dz = dxdydz[0], dxdydz[1], dxdydz[2]

    def stack_left_right(a_edg, d_a):
        left = a_edg - d_a / level / 2
        right = a_edg + d_a / level / 2
        return BackendTensor.tfnp.stack([left, right], axis=1)

    x_ = BackendTensor.tfnp.repeat(stack_left_right(x_coord, dx), 4, axis=1)
    x = BackendTensor.tfnp.ravel(x_)

    y_temp = BackendTensor.tfnp.repeat(stack_left_right(y_coord, dy), 2, axis=1)
    y_ = BackendTensor.tfnp.tile(y_temp, (1, 2))
    y = BackendTensor.tfnp.ravel(y_)

    z_ = BackendTensor.tfnp.tile(stack_left_right(z_coord, dz), (1, 4))
    z = BackendTensor.tfnp.ravel(z_)

    new_xyz = BackendTensor.tfnp.stack([x, y, z], axis=1)

    # For PyTorch, ensure contiguous memory (equivalent to np.ascontiguousarray)
    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        if hasattr(new_xyz, 'contiguous'):
            new_xyz = new_xyz.contiguous()

    return new_xyz
