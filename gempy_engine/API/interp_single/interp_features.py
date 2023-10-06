import copy
from typing import List

import gempy_engine.core.data.tensors_structure
from ...core.data.grid import Grid
from ...core import data
from ...core.data import InterpolationOptions
from ...core.data.octree_level import OctreeLevel
from ...core.data.interp_output import InterpOutput
from ...core.data.scalar_field_output import ScalarFieldOutput
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput

from ...modules.octrees_topology.octrees_topology_interface import get_next_octree_grid

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
        
        # * Here it goes all the different grids
        next_octree: OctreeLevel = interpolate_on_octree(interpolation_input, options, data_descriptor)
        
        if options.is_last_octree_level is False:
            # * This is only a Grid with a Regular grid
            grid_1_centers: Grid = get_next_octree_grid(
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
# ? DEP - It seems I just run this in the tests
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
        stack_relation=interpolation_input.stack_relation
    )

    return InterpOutput(scalar_output, None)


def interpolate_and_segment(interpolation_input: InterpolationInput, options: data.InterpolationOptions,  # * Just for testing
                            data_shape: gempy_engine.core.data.tensors_structure.TensorsStructure, clean_buffer=True) -> InterpOutput:
    output: ScalarFieldOutput = interpolate_feature(
        interpolation_input=interpolation_input,
        options=options,
        data_shape=data_shape,
        solver_input=input_preprocess(data_shape, interpolation_input)
    )
    return InterpOutput(output, None)

# endregion
