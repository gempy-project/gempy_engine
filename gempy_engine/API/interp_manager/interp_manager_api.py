from __future__ import annotations

import copy
from typing import List

from ..dual_contouring.dual_contouring import compute_dual_contouring, get_intersection_on_edges
from ..interp_single._interp_single_internals import interpolate_all_fields
from ..interp_single.interp_single_interface import compute_n_octree_levels, interpolate_single_field
from ...core.data import InterpolationOptions, TensorsStructure
from ...core.data.exported_structs import OctreeLevel, InterpOutput, DualContouringData, \
    DualContouringMesh, Solutions, ExportedFields
from ...core.data.grid import Grid
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interpolation_input import InterpolationInput
from ...modules.octrees_topology.octrees_topology_interface import get_regular_grid_for_level


def interpolate_model(interpolation_input: InterpolationInput, options: InterpolationOptions,
                      data_descriptor: InputDataDescriptor) -> Solutions:
    interpolation_input = copy.deepcopy(interpolation_input)  # TODO: Make sure if this works with TF

    solutions: Solutions = _interpolate_all(interpolation_input, options, data_descriptor)

    # TODO: Masking logic
    # squeeze_solution = _compute_mask(solutions)

    # TODO: final dual countoring. I need to make the masking operations first
    if True:
        meshes = _dual_contouring(data_descriptor, interpolation_input, options, solutions)
        solutions.dc_meshes = meshes

    # ---------------------
    # TODO: [ ] Gravity

    # TODO: [ ] Magnetics
    return solutions


def _interpolate_all(stack_interpolation_input: InterpolationInput, options: InterpolationOptions,
                     data_descriptor: InputDataDescriptor) -> Solutions:
    # ? Should be a list of solutios or Solution should contain a list of InterpOutputs?
    solutions: Solutions = Solutions()
    # TODO: [-] Looping scalars

    number_octree_levels = options.number_octree_levels
    output: List[OctreeLevel] = compute_n_octree_levels(number_octree_levels, stack_interpolation_input,
                                                        options, data_descriptor)
    solutions.octrees_output = output
    solutions.debug_input_data = stack_interpolation_input
    return solutions


# ! DEP
def _compute_mask(solutions: List[Solutions]):
    # TODO: Add mask_fault
    all_mask_components = solutions[0].octrees_output[-1].output_centers.mask_components
    squeezed_regular_grid = get_regular_grid_for_level(solutions[0].octrees_output, 2)
    squeezed_regular_grid2 = get_regular_grid_for_level(solutions[1].octrees_output, 2)
    return squeezed_regular_grid, squeezed_regular_grid2

    # previous_mask_formation = mask_onlap
    # 
    # # mask_val = T.cumprod(mask_matrix[n_series - nsle_op: n_series, shift:x_to_interpolate_shape + shift][::-1], axis=0)[::-1]
    # 
    # # TODO: For each stack since the last erode multiply the mask with the previous one
    # mask_formation_since_last_erode 
    # 
    # mask_matrix_this_stack = mask_erode


def _dual_contouring(data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
                     options: InterpolationOptions, solutions: Solutions) -> List[DualContouringMesh]:
    # Dual Contouring prep:
    octree_leaves = solutions.octrees_output[-1]
    dc_data: DualContouringData = get_intersection_on_edges(octree_leaves)
    interpolation_input.grid = Grid(dc_data.xyz_on_edge)
    output_on_edges: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, data_descriptor)
    dc_data.gradients = output_on_edges[-1].final_exported_fields
    # --------------------
    # The following operations are applied on the FINAL lith block:
    # This should happen only on the leaf of an octree
    # TODO: [ ] Dual contouring. This method only make one vertex per voxel. It is possible to make water tight surfaces?
    # compute_dual_contouring
    # TODO [ ] The api should grab an octree level
    meshes: List[DualContouringMesh] = compute_dual_contouring(dc_data, data_descriptor.tensors_structure.n_surfaces)
    return meshes

# ? DEP
# def _interpolate_stack(root_data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
#                        options: InterpolationOptions) -> Solutions | list[Solutions]:
#     all_solutions: List[Solutions] = []
# 
#     stack_structure = root_data_descriptor.stack_structure
# 
#     if stack_structure is None:
#         solutions = _interpolate_all(interpolation_input, options, root_data_descriptor)
#         return solutions
#     else:
#         for i in range(stack_structure.n_stacks):
#             stack_structure.stack_number = i
# 
#             tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(root_data_descriptor, i)
#             interpolation_input_i = InterpolationInput.from_interpolation_input_subset(interpolation_input, stack_structure)
# 
#             solutions = _interpolate_all(interpolation_input_i, options, tensor_struct_i)
#             all_solutions.append(solutions)
# 
#     return all_solutions
# TODO: This is where we would have to include any other implicit function
