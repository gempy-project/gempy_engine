import copy
from typing import List

from scoping import scoping

from ..dual_contouring.dual_contouring import compute_dual_contouring, get_intersection_on_edges
from ..interp_single.interp_single_interface import compute_n_octree_levels, interpolate_single_field
from ...core.data import InterpolationOptions, TensorsStructure
from ...core.data.exported_structs import OctreeLevel, InterpOutput, DualContouringData, \
    DualContouringMesh, Solutions
from ...core.data.grid import Grid
from ...core.data.interpolation_input import InterpolationInput


def interpolate_model(interpolation_input: InterpolationInput, options: InterpolationOptions,
                      data_shape: TensorsStructure) -> Solutions:
    interpolation_input = copy.deepcopy(interpolation_input)  # TODO: Make sure if this works with TF

    solutions = _interpolate_stack(data_shape, interpolation_input, options)

    # TODO: final dual countoring. I need to make the masking operations first
    if False:
        meshes = _dual_contouring(data_shape, interpolation_input, options, solutions)
        solutions.dc_meshes = meshes

    # ---------------------
    # TODO: [ ] Gravity

    # TODO: [ ] Magnetics
    return solutions


def _dual_contouring(data_shape, interpolation_input, options, solutions):
    # Dual Contouring prep:
    octree_leaves = solutions.octrees_output[-1]
    dc_data: DualContouringData = get_intersection_on_edges(octree_leaves)
    interpolation_input.grid = Grid(dc_data.xyz_on_edge)
    output_on_edges: InterpOutput = interpolate_single_field(interpolation_input, options, data_shape)
    dc_data.gradients = output_on_edges.exported_fields
    # --------------------
    # The following operations are applied on the FINAL lith block:
    # This should happen only on the leaf of an octree
    # TODO: [ ] Dual contouring. This method only make one vertex per voxel. It is possible to make water tight surfaces?
    # compute_dual_contouring
    # TODO [ ] The api should grab an octree level
    meshes: List[DualContouringMesh] = compute_dual_contouring(dc_data, data_shape.n_surfaces)
    return meshes


def _interpolate_stack(root_data_shape: TensorsStructure, interpolation_input: InterpolationInput,
                       options: InterpolationOptions):
    
    all_solutions: List[Solutions] = []
    if root_data_shape.stack_structure is None:
        solutions = _interpolate_scalar(options, root_data_shape, interpolation_input)
        all_solutions.append(solutions)
    else:
        for i in range(root_data_shape.n_stacks):
            stack_data_shape = TensorsStructure.from_tensor_structure_subset(root_data_shape, i)
            stack_interpolation_input = InterpolationInput.from_interpolation_input_subset(interpolation_input, stack_data_shape)
    
            solutions = _interpolate_scalar(options, stack_data_shape, stack_interpolation_input)
            all_solutions.append(solutions)

    return all_solutions[0]


def _interpolate_scalar(options, stack_data_shape, stack_interpolation_input):
    solutions: Solutions = Solutions()
    # TODO: [ ] Looping scalars
    number_octree_levels = options.number_octree_levels
    output: List[OctreeLevel] = compute_n_octree_levels(number_octree_levels, stack_interpolation_input,
                                                        options, stack_data_shape)
    solutions.octrees_output = output
    solutions.debug_input_data = stack_interpolation_input
    return solutions
