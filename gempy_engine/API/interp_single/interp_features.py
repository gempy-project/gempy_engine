from typing import List

from . import _multi_scalar_field_manager as ms
from ._octree_generation import interpolate_on_octree
from ...core import data
from ...core.data import InterpolationOptions
from ...core.data.engine_grid import EngineGrid
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interp_output import InterpOutput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.octree_level import OctreeLevel
from ...modules.octrees_topology.octrees_topology_interface import get_next_octree_grid


def interpolate_n_octree_levels(interpolation_input: InterpolationInput, options: data.InterpolationOptions,
                                data_descriptor: InputDataDescriptor) -> List[OctreeLevel]:
    n_levels = options.number_octree_levels

    octree_list = []
    for i in range(0, n_levels):
        options.temp_interpolation_values.current_octree_level = i
        
        # * Here it goes all the different grids
        next_octree: OctreeLevel = interpolate_on_octree(interpolation_input, options, data_descriptor)
        
        if options.is_last_octree_level is False:
            # * This is only a Grid with a Regular grid
            grid_1_centers: EngineGrid = get_next_octree_grid(
                prev_octree=next_octree,
                evaluation_options=options.evaluation_options,
                current_octree_level=i
            )
            interpolation_input.set_temp_grid(grid_1_centers)
        octree_list.append(next_octree)

    interpolation_input.set_grid_to_original()
    return octree_list


def interpolate_all_fields_no_octree(interpolation_input: InterpolationInput, options: InterpolationOptions,
                                     data_descriptor: InputDataDescriptor) -> List[InterpOutput]:
    # TODO (miguel March 2026): These copies I do not think are necessary since a while
    # if BackendTensor.engine_backend is not AvailableBackends.PYTORCH and NOT_MAKE_INPUT_DEEP_COPY is False:
    #     temp_interpolation_input = copy.deepcopy(interpolation_input)
    # else:
    #     temp_interpolation_input = interpolation_input

    temp_interpolation_input = interpolation_input
    return ms.interpolate_all_fields(temp_interpolation_input, options, data_descriptor)
