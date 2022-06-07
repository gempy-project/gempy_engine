from __future__ import annotations

import copy
from typing import List

import numpy as np

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


def _dual_contouring(data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
                     options: InterpolationOptions, solutions: Solutions) -> List[DualContouringMesh]:
    # Dual Contouring prep:
    octree_leaves = solutions.octrees_output[-1]
    all_meshes: List[DualContouringMesh] = []
    for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
        
        output_corners: InterpOutput = octree_leaves.outputs_corners[n_scalar_field]
        
        dc_data: DualContouringData = get_intersection_on_edges(octree_leaves, output_corners)
        interpolation_input.grid = Grid(dc_data.xyz_on_edge)
        output_on_edges: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, data_descriptor)

        dc_data.gradients: ExportedFields = output_on_edges[n_scalar_field].exported_fields
        n_surfaces = data_descriptor.stack_structure.number_of_surfaces_per_stack[n_scalar_field]    #     
        # --------------------
        # The following operations are applied on the FINAL lith block:
        # This should happen only on the leaf of an octree
        # TODO: [ ] Dual contouring. This method only make one vertex per voxel. It is possible to make water tight surfaces?
        # compute_dual_contouring
        # TODO [ ] The api should grab an octree level
        meshes: List[DualContouringMesh] = compute_dual_contouring(dc_data, n_surfaces)
        all_meshes.append(*meshes)
    return all_meshes
