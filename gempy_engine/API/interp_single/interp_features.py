import copy
from typing import List

import gempy_engine.core.data.tensors_structure
from ...core import data
from ...core.data import InterpolationOptions
from ...core.data.octree_level import OctreeLevel
from ...core.data.interp_output import InterpOutput
from ...core.data.scalar_field_output import ScalarFieldOutput
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput

from ...modules.octrees_topology import octrees_topology_interface as octrees

from . import _multi_scalar_field_manager as ms
from ._interp_scalar_field import WeightsBuffer, interpolate_scalar_field
from ._interp_single_feature import interpolate_feature, input_preprocess
from ._octree_generation import interpolate_on_octree


def interpolate_n_octree_levels(interpolation_input: InterpolationInput, options: data.InterpolationOptions,
                                data_descriptor: InputDataDescriptor) -> List[OctreeLevel]:
    n_levels = options.number_octree_levels

    octree_list = []
    for i in range(0, n_levels):
        options.current_octree_level = i
        
        next_octree: OctreeLevel = interpolate_on_octree(interpolation_input, options, data_descriptor)
        
        if options.is_last_octree_level is False:
            grid_1_centers = octrees.get_next_octree_grid(
                prev_octree=next_octree,
                compute_topology=False,
                debug=False
            )
            interpolation_input.grid = grid_1_centers
        octree_list.append(next_octree)

    return octree_list


def interpolate_all_fields_no_octree(interpolation_input: InterpolationInput, options: InterpolationOptions,
                                     data_descriptor: InputDataDescriptor) -> List[InterpOutput]:
    interpolation_input = copy.deepcopy(interpolation_input)
    return ms.interpolate_all_fields(interpolation_input, options, data_descriptor)


# region testing
def interpolate_single_field(interpolation_input: InterpolationInput, options: data.InterpolationOptions,
                             data_shape: gempy_engine.core.data.tensors_structure.TensorsStructure) -> InterpOutput:  # * Only For testing

    grid = interpolation_input.grid
    solver_input = input_preprocess(data_shape, interpolation_input)
    weights, exported_fields = interpolate_scalar_field(solver_input, options)

    exported_fields.set_structure_values(
        reference_sp_position=data_shape.reference_sp_position,
        slice_feature=interpolation_input.slice_feature,
        grid_size=interpolation_input.grid.len_all_grids
    )
    
    scalar_output = ScalarFieldOutput(
        weights=weights,
        grid=grid,
        exported_fields=exported_fields,
        values_block=None,
        mask_components=None
    )

    return InterpOutput(scalar_output)


def interpolate_and_segment(interpolation_input: InterpolationInput, options: data.InterpolationOptions,  # * Just for testing
                            data_shape: gempy_engine.core.data.tensors_structure.TensorsStructure, clean_buffer=True) -> InterpOutput:
    output: ScalarFieldOutput = interpolate_feature(interpolation_input, options, data_shape)
    return InterpOutput(output)

# endregion
