import copy
from typing import List

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
from ...modules.octrees_topology._octree_common import _cell_children


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
    # x slowest, z fastest: the order of the corner ids that refinement compares
    return _cell_children(regular_grid.values, regular_grid.dxdydz, level, 2)
