import copy
from typing import List

from ...core import data
from ...core.data import InterpolationOptions
from ...core.data.exported_structs import InterpOutput, OctreeLevel, ScalarFieldOutput
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput

from ...modules.octrees_topology import octrees_topology_interface as octrees

from . import _multi_scalar_field_manager as ms
from ._interp_scalar_field import Buffer, interpolate_scalar_field
from ._interp_single_feature import interpolate_feature
from ._octree_generation import interpolate_on_octree


def interpolate_n_octree_levels(interpolation_input: InterpolationInput, options: data.InterpolationOptions,
                                data_descriptor: InputDataDescriptor) -> List[OctreeLevel]:
    n_levels = options.number_octree_levels

    octree_list = []
    for i in range(0, n_levels):
        next_octree: OctreeLevel = interpolate_on_octree(interpolation_input, options.kernel_options, data_descriptor)
        grid_1_centers = octrees.get_next_octree_grid(next_octree, compute_topology=False, debug=False)

        interpolation_input.grid = grid_1_centers
        octree_list.append(next_octree)

    Buffer.clean()
    return octree_list


def interpolate_all_fields_no_octree(interpolation_input: InterpolationInput, options: InterpolationOptions,
                                     data_descriptor: InputDataDescriptor) -> List[InterpOutput]:
    interpolation_input = copy.deepcopy(interpolation_input)
    return ms.interpolate_all_fields(interpolation_input, options.kernel_options, data_descriptor)


# region testing
def interpolate_single_field(interpolation_input: InterpolationInput, options: data.InterpolationOptions,
                             data_shape: data.TensorsStructure) -> InterpOutput:  # * Only For testing

    grid = interpolation_input.grid
    weights, exported_fields = interpolate_scalar_field(interpolation_input, options.kernel_options, data_shape)
    scalar_output = ScalarFieldOutput(
        weights=weights,
        grid=grid,
        exported_fields=exported_fields,
        values_block=None,
        mask_components=None
    )

    return InterpOutput(scalar_output)


def interpolate_and_segment(interpolation_input: InterpolationInput, options: data.InterpolationOptions,  # * Just for testing
                            data_shape: data.TensorsStructure, clean_buffer=True) -> InterpOutput:
    output: ScalarFieldOutput = interpolate_feature(interpolation_input, options.kernel_options, data_shape, clean_buffer)
    return InterpOutput(output)

# endregion
