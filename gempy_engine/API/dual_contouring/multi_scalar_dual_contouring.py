import copy
from typing import List, Any

import numpy as np

from gempy_engine.modules.dual_contouring._dual_contouring import compute_dual_contouring
from ._experimental_water_tight_DC_1 import _experimental_water_tight
from ._mask_buffer import MaskBuffer
from ..interp_single.interp_features import interpolate_all_fields_no_octree
from ...core.backend_tensor import BackendTensor
from ...core.data import InterpolationOptions
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.engine_grid import EngineGrid
from ...core.data.generic_grid import GenericGrid
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interp_output import InterpOutput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.octree_level import OctreeLevel
from ...core.utils import gempy_profiler_decorator
from ...modules.dual_contouring.dual_contouring_interface import (find_intersection_on_edge, get_triangulation_codes,
                                                                  get_masked_codes, mask_generation)


@gempy_profiler_decorator
def dual_contouring_multi_scalar(
        data_descriptor: InputDataDescriptor,
        interpolation_input: InterpolationInput,
        options: InterpolationOptions,
        octree_list: List[OctreeLevel]
) -> List[DualContouringMesh]:
    """
    Perform dual contouring for multiple scalar fields.
    
    Args:
        data_descriptor: Input data descriptor containing stack structure information
        interpolation_input: Input data for interpolation
        options: Interpolation options including debug and extraction settings
        octree_list: List of octree levels with the last being the leaf level
        
    Returns:
        List of dual contouring meshes for all processed scalar fields
    """
    # Dual Contouring prep:
    MaskBuffer.clean()

    octree_leaves = octree_list[-1]
    all_meshes: List[DualContouringMesh] = []

    dual_contouring_options = copy.deepcopy(options)
    dual_contouring_options.evaluation_options.compute_scalar_gradient = True

    # * (Miguel Sep25) This will be probably deprecated
    if options.debug_water_tight:
        _experimental_water_tight(
            all_meshes, data_descriptor, interpolation_input, octree_leaves, dual_contouring_options
        )
        return all_meshes

    # * 1) Triangulation code
    left_right_codes = get_triangulation_codes(octree_list, options)

    # * 2) Dual contouring mask
    # ? I guess this mask is different that erosion mask
    all_mask_arrays: np.ndarray = mask_generation(
        octree_leaves=octree_leaves,
        masking_option=options.evaluation_options.mesh_extraction_masking_options
    )
    
    # Process each scalar field
    all_active_cells = []
    all_stack_intersection = []
    all_valid_edges = []
    all_left_right_codes = []
    # region Interp on edges
    for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
        _validate_stack_relations(data_descriptor, n_scalar_field)
        mask: np.ndarray = all_mask_arrays[n_scalar_field]
        
        # * 3) Masking  Left_right_codes
        left_right_codes_per_stack = get_masked_codes(left_right_codes, mask)
        all_left_right_codes.append(left_right_codes_per_stack)

        # * 4) Find edges 
        output: InterpOutput = octree_leaves.outputs_centers[n_scalar_field]
        intersection_xyz, valid_edges = find_intersection_on_edge(
            _xyz_corners=octree_leaves.grid_centers.corners_grid.values,
            scalar_field_on_corners=output.exported_fields.scalar_field[output.grid.corners_grid_slice],
            scalar_at_sp=output.scalar_field_at_sp,
            masking=mask
        )

        all_stack_intersection.append(intersection_xyz)
        all_valid_edges.append(valid_edges)

    # * 5) Interpolate on edges for all stacks
    output_on_edges = _interp_on_edges(
        all_stack_intersection, data_descriptor, dual_contouring_options, interpolation_input
    )
    
    # endregion

    # region Vertex gen and triangulation
    foo = []
    # Generate meshes for each scalar field
    for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
        output: InterpOutput = octree_leaves.outputs_centers[n_scalar_field]
        mask = all_mask_arrays[n_scalar_field]

        dc_data = DualContouringData(
            xyz_on_edge=all_stack_intersection[n_scalar_field],
            valid_edges=all_valid_edges[n_scalar_field],
            xyz_on_centers=(
                    octree_leaves.grid_centers.octree_grid.values if mask is None
                    else octree_leaves.grid_centers.octree_grid.values[mask]
            ),
            dxdydz=octree_leaves.grid_centers.octree_dxdydz,
            exported_fields_on_edges=output_on_edges[n_scalar_field].exported_fields,
            n_surfaces_to_export=output.scalar_field_at_sp.shape[0],
            tree_depth=options.number_octree_levels,
        )

        meshes: List[DualContouringMesh] = compute_dual_contouring(
            dc_data_per_stack=dc_data,
            left_right_codes=all_left_right_codes[n_scalar_field],
            debug=options.debug
        )
        
        for m in meshes:
            foo.append(m.left_right)

        # TODO: If the order of the meshes does not match the order of scalar_field_at_surface points, reorder them here
        if meshes is not None:
            all_meshes.extend(meshes)

    # endregion
    # Check for repeated voxels across stacks
    if (options.debug or len(all_left_right_codes) > 1) and True:
        voxel_overlaps = find_repeated_voxels_across_stacks(foo)
        if voxel_overlaps and options.debug:
            print(f"Found voxel overlaps between stacks: {voxel_overlaps}")
            _f(all_meshes, 1, 0, voxel_overlaps)
            _f(all_meshes, 2, 0, voxel_overlaps)
            _f(all_meshes, 3, 0, voxel_overlaps)
            _f(all_meshes, 4, 0, voxel_overlaps)
            _f(all_meshes, 5, 0, voxel_overlaps)

    return all_meshes


def _f(all_meshes: list[DualContouringMesh], destination: int, origin: int, voxel_overlaps: dict):
    key = f"stack_{origin}_vs_stack_{destination}"
    all_meshes[destination].vertices[voxel_overlaps[key]["indices_in_stack_j"]] = all_meshes[origin].vertices[voxel_overlaps[key]["indices_in_stack_i"]]


def find_repeated_voxels_across_stacks(all_left_right_codes: List[np.ndarray]) -> dict:
    """
    Find repeated voxels using NumPy operations - better for very large arrays.

    Args:
        all_left_right_codes: List of left_right_codes arrays, one per stack

    Returns:
        Dictionary with detailed overlap analysis
    """

    if not all_left_right_codes:
        return {}

    # Generate voxel codes for each stack

    from gempy_engine.modules.dual_contouring.fancy_triangulation import _StaticTriangulationData
    stack_codes = []
    for left_right_codes in all_left_right_codes:
        if left_right_codes.size > 0:
            voxel_codes = (left_right_codes * _StaticTriangulationData.get_pack_directions_into_bits()).sum(axis=1)
            stack_codes.append(voxel_codes)
        else:
            stack_codes.append(np.array([]))

    overlaps = {}

    # Check each pair of stacks
    for i in range(len(stack_codes)):
        for j in range(i + 1, len(stack_codes)):
            if stack_codes[i].size == 0 or stack_codes[j].size == 0:
                continue

            # Find common voxel codes using numpy
            common_codes = np.intersect1d(stack_codes[i], stack_codes[j])

            if len(common_codes) > 0:
                # Get indices of common voxels in each stack
                indices_i = np.isin(stack_codes[i], common_codes)
                indices_j = np.isin(stack_codes[j], common_codes)

                overlaps[f"stack_{i}_vs_stack_{j}"] = {
                        'common_voxel_codes'   : common_codes,
                        'count'                : len(common_codes),
                        'indices_in_stack_i'   : np.where(indices_i)[0],
                        'indices_in_stack_j'   : np.where(indices_j)[0],
                        'common_binary_codes_i': all_left_right_codes[i][indices_i],
                        'common_binary_codes_j': all_left_right_codes[j][indices_j]
                }

    return overlaps


def _validate_stack_relations(data_descriptor: InputDataDescriptor, n_scalar_field: int) -> None:
    """
    Validate stack relations for the given scalar field.

    Args:
        data_descriptor: Input data descriptor containing stack relations
        n_scalar_field: Current scalar field index

    Raises:
        NotImplementedError: If unsupported combination of Erosion and Onlap is detected
    """
    if n_scalar_field == 0:
        return

    previous_stack_is_onlap = data_descriptor.stack_relation[n_scalar_field - 1] == 'Onlap'
    was_erosion_before = data_descriptor.stack_relation[n_scalar_field - 1] == 'Erosion'

    if previous_stack_is_onlap and was_erosion_before:
        # TODO (July, 2023): Is this still valid? I thought we have all the combinations
        raise NotImplementedError("Erosion and Onlap are not supported yet")


def _interp_on_edges(
        all_stack_intersection: List[Any],
        data_descriptor: InputDataDescriptor,
        dual_contouring_options: InterpolationOptions,
        interpolation_input: InterpolationInput
) -> List[InterpOutput]:
    """
    Interpolate scalar fields on edge intersection points.

    Args:
        all_stack_intersection: List of intersection points for all stacks
        data_descriptor: Input data descriptor
        dual_contouring_options: Dual contouring specific options
        interpolation_input: Interpolation input data

    Returns:
        List of interpolation outputs for each stack
    """

    # Set temporary grid with concatenated intersection points
    interpolation_input.set_temp_grid(
        EngineGrid(
            custom_grid=GenericGrid(
                values=BackendTensor.t.concatenate(all_stack_intersection, axis=0)
            )
        )
    )

    # TODO (@miguel 21 June): By definition in `interpolate_all_fields_no_octree`
    # we just need to interpolate up to the n_scalar_field, but need to test this
    # This should be done with buffer weights to avoid waste
    output_on_edges: List[InterpOutput] = interpolate_all_fields_no_octree(
        interpolation_input=interpolation_input,
        options=dual_contouring_options,
        data_descriptor=data_descriptor
    )

    # Restore original grid
    interpolation_input.set_grid_to_original()
    return output_on_edges