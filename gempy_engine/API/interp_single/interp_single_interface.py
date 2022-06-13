import copy
from typing import List

from . import _interp_single_internals
from gempy_engine.API.interp_single._interp_single_internals import interpolate_scalar_field, _evaluate_sys_eq
from ._octree_generation import interpolate_on_octree
from ...core import data
from ...core.data.exported_structs import InterpOutput, OctreeLevel, ScalarFieldOutput
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput
from ...modules.octrees_topology import octrees_topology_interface as octrees


# * Only For testing
def interpolate_single_field(interpolation_input: InterpolationInput, options: data.InterpolationOptions,
                             data_shape: data.TensorsStructure) -> InterpOutput:
    
    grid = interpolation_input.grid
    weights, exported_fields = interpolate_scalar_field(interpolation_input, options, data_shape)
    scalar_output = ScalarFieldOutput(
        weights= weights,
        grid=grid,
        exported_fields=exported_fields,
        values_block=None,
        mask_components=None
    )

    return InterpOutput(scalar_output)


def compute_n_octree_levels(n_levels: int, interpolation_input_original: InterpolationInput,
                            options: data.InterpolationOptions, data_descriptor: InputDataDescriptor
                            ) -> List[OctreeLevel]:
    interpolation_input = copy.deepcopy(interpolation_input_original)

    octree_list = []
    next_octree = OctreeLevel()
    next_octree.is_root = True

    for i in range(0, n_levels):
        next_octree: OctreeLevel = interpolate_on_octree(next_octree, interpolation_input, options, data_descriptor)
        grid_1_centers = octrees.get_next_octree_grid(next_octree, compute_topology=False, debug=False)

        interpolation_input.grid = grid_1_centers
        octree_list.append(next_octree)

        next_octree = OctreeLevel()  # TODO: This can go inside interpolate_on_octree
    _interp_single_internals.Buffer.clean()
    return octree_list


def interpolate_and_segment(interpolation_input: InterpolationInput, options: data.InterpolationOptions,  # * Just for testing
                            data_shape: data.TensorsStructure, clean_buffer=True) -> InterpOutput:
    output: ScalarFieldOutput = _interp_single_internals._interpolate_a_scalar_field(interpolation_input, options, data_shape, clean_buffer)
    return InterpOutput(output)
