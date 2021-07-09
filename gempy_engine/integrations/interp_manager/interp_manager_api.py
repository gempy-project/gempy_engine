from typing import List

from ..dual_contouring.dual_contouring import compute_dual_contouring, get_intersection_on_edges
from ..interp_single.interp_single_interface import interpolate_and_segment, \
    compute_n_octree_levels, interpolate_single_field
from ...core.data import InterpolationOptions, TensorsStructure
from ...core.data.exported_structs import OctreeLevel, InterpOutput, DualContouringData, \
    DualContouringMesh, Solutions
from ...core.data.grid import Grid
from ...core.data.interpolation_input import InterpolationInput


def interpolate_model(interpolation_input: InterpolationInput, options: InterpolationOptions,
                      data_shape: TensorsStructure) -> Solutions:
    solutions = Solutions()

    # TODO: [ ] Looping scalars
    number_octree_levels = options.number_octree_levels
    output: List[OctreeLevel] = compute_n_octree_levels(number_octree_levels, interpolation_input,
                                                        options, data_shape)
    solutions.octrees_output = output

    # Dual Contouring prep:
    dc_data: DualContouringData = get_intersection_on_edges(output[-1])
    interpolation_input.grid = Grid(dc_data.xyz_on_edge)
    output_on_edges: InterpOutput = interpolate_single_field(interpolation_input, options,
                                                             data_shape)
    dc_data.gradients = output_on_edges.exported_fields
    # --------------------
    # The following operations are applied on the FINAL lith block:

    # This should happen only on the leaf of an octree
    # TODO: [ ] Dual contouring. This method only make one vertex per voxel. It is possible to make water tight surfaces?
    # compute_dual_contouring
    # TODO [ ] The api should grab an octree level
    meshes: List[DualContouringMesh] = compute_dual_contouring(dc_data)
    solutions.dc_meshes = meshes

    # ---------------------
    # TODO: [ ] Gravity

    # TODO: [ ] Magnetics
    return solutions
