from typing import List

from . import _interp_single_internals
from ._interp_single_internals import interpolate_scalar_field
from ._octree_generation import interpolate_on_octree
from ...core import data
from ...core.data.exported_structs import InterpOutput, OctreeLevel
from ...core.data.interpolation_input import InterpolationInput
from ...modules.octrees_topology import octrees_topology_interface as octrees


def interpolate_and_segment(interpolation_input: InterpolationInput,
                            options: data.InterpolationOptions,
                            data_shape: data.TensorsStructure,
                            clean_buffer=True
                            ) -> InterpOutput:
    output = _interp_single_internals.interpolate(interpolation_input, options,
                                                  data_shape, clean_buffer)
    return output


def interpolate_single_field(interpolation_input: InterpolationInput,
                             options: data.InterpolationOptions,
                             data_shape: data.TensorsStructure
                             ) -> InterpOutput:
    output = InterpOutput()
    output.grid = interpolation_input.grid
    output.weights, output.exported_fields = interpolate_scalar_field(interpolation_input, options,
                                                                      data_shape)

    return output


def compute_n_octree_levels(n_levels:int, interpolation_input: InterpolationInput,
                            options: data.InterpolationOptions, data_shape: data.TensorsStructure)\
        -> List[OctreeLevel]:

    octree_list = []
    next_octree = OctreeLevel()
    next_octree.is_root = True

    for i in range(0, n_levels):
        next_octree = interpolate_on_octree(next_octree, interpolation_input, options, data_shape)
        grid_1_centers = octrees.get_next_octree_grid(next_octree, compute_topology=False, debug=False)

        interpolation_input.grid = grid_1_centers
        octree_list.append(next_octree)

        next_octree = OctreeLevel()
    _interp_single_internals.Buffer.clean()
    return octree_list
