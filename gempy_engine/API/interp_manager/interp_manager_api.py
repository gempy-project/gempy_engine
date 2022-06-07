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
    dc_data: DualContouringData = get_intersection_on_edges(octree_leaves)
    interpolation_input.grid = Grid(dc_data.xyz_on_edge)
    output_on_edges: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, data_descriptor)

    # TODO: [ ] We need to do the following for each field
    one_scalar_field_at_a_time = True
    if one_scalar_field_at_a_time:
        dc_data.gradients: ExportedFields = output_on_edges[-1].final_exported_fields
        # n_surfaces = data_descriptor.stack_structure.number_of_surfaces_per_stack[-1]
        n_surfaces = data_descriptor.tensors_structure.n_surfaces
    # else:
    #     all_gx: np.ndarray = np.zeros(0)
    #     all_gy: np.ndarray = np.zeros(0)
    #     all_gz: np.ndarray = np.zeros(0)
    # 
    #     for interp_output in output_on_edges:
    #         exported_fields = interp_output.final_exported_fields
    #         
    #         all_gx = np.concatenate((all_gx, exported_fields._gx_field))
    #         all_gy = np.concatenate((all_gy, exported_fields._gy_field))
    #         all_gz = np.concatenate((all_gz, exported_fields._gz_field))
    #     
    #     all_fields = ExportedFields(None, all_gx, all_gy, all_gz)
    #     dc_data.gradients = all_fields
    #     n_surfaces = data_descriptor.tensors_structure.n_surfaces
    #     
    # --------------------
    # The following operations are applied on the FINAL lith block:
    # This should happen only on the leaf of an octree
    # TODO: [ ] Dual contouring. This method only make one vertex per voxel. It is possible to make water tight surfaces?
    # compute_dual_contouring
    # TODO [ ] The api should grab an octree level
    meshes: List[DualContouringMesh] = compute_dual_contouring(dc_data, n_surfaces)
    return meshes
