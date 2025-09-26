import copy
import warnings
from typing import List, Any

import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.modules.dual_contouring._dual_contouring import compute_dual_contouring
from ._experimental_water_tight_DC_1 import _experimental_water_tight
from ._mask_buffer import MaskBuffer
from ...core.data import InterpolationOptions
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.interp_output import InterpOutput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.octree_level import OctreeLevel
from ...core.data.options import MeshExtractionMaskingOptions
from ...core.data.stack_relation_type import StackRelationType
from ...core.utils import gempy_profiler_decorator
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge
from ...modules.dual_contouring.fancy_triangulation import get_left_right_array


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

    if options.debug_water_tight:
        _experimental_water_tight(
            all_meshes, data_descriptor, interpolation_input, octree_leaves, dual_contouring_options
        )
        return all_meshes

    # Determine triangulation strategy
    left_right_codes = _get_triangulation_codes(octree_list, options)

    # Generate masks for all scalar fields
    all_mask_arrays: np.ndarray = _mask_generation(
        octree_leaves=octree_leaves,
        masking_option=options.evaluation_options.mesh_extraction_masking_options
    )

    # Process each scalar field
    all_stack_intersection = []
    all_valid_edges = []
    all_left_right_codes = []

    for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
        _validate_stack_relations(data_descriptor, n_scalar_field)

        mask: np.ndarray = all_mask_arrays[n_scalar_field]
        left_right_codes_per_stack = _get_masked_codes(left_right_codes, mask)

        output: InterpOutput = octree_leaves.outputs_centers[n_scalar_field]
        intersection_xyz, valid_edges = find_intersection_on_edge(
            _xyz_corners=octree_leaves.grid_centers.corners_grid.values,
            scalar_field_on_corners=output.exported_fields.scalar_field[output.grid.corners_grid_slice],
            scalar_at_sp=output.scalar_field_at_sp,
            masking=mask
        )

        all_stack_intersection.append(intersection_xyz)
        all_valid_edges.append(valid_edges)
        all_left_right_codes.append(left_right_codes_per_stack)

    # Interpolate on edges for all stacks
    output_on_edges = _interp_on_edges(
        all_stack_intersection, data_descriptor, dual_contouring_options, interpolation_input
    )

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

        # TODO: If the order of the meshes does not match the order of scalar_field_at_surface points, reorder them here
        if meshes is not None:
            all_meshes.extend(meshes)

    _vertex_select_last_pass(all_mask_arrays)

    return all_meshes


def _vertex_select_last_pass(mask, ):
    pass


def _get_triangulation_codes(octree_list: List[OctreeLevel], options: InterpolationOptions) -> np.ndarray | None:
    """
    Determine the appropriate triangulation codes based on options and octree structure.
    
    Args:
        octree_list: List of octree levels
        options: Interpolation options
        
    Returns:
        Left-right codes array if fancy triangulation is enabled and supported, None otherwise
    """
    is_pure_octree = bool(np.all(octree_list[0].grid_centers.octree_grid_shape == 2))

    match (options.evaluation_options.mesh_extraction_fancy, is_pure_octree):
        case (True, True):
            return get_left_right_array(octree_list)
        case (True, False):
            warnings.warn(
                "Fancy triangulation only works with regular grid of resolution [2,2,2]. "
                "Defaulting to regular triangulation"
            )
            return None
        case (False, _):
            return None
        case _:
            raise ValueError("Invalid combination of options")


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


def _get_masked_codes(left_right_codes: np.ndarray | None, mask: np.ndarray | None) -> np.ndarray | None:
    """
    Apply mask to left-right codes if both are available.
    
    Args:
        left_right_codes: Original left-right codes array
        mask: Boolean mask array
        
    Returns:
        Masked codes if both inputs are not None, otherwise original codes
    """
    if mask is not None and left_right_codes is not None:
        return left_right_codes[mask]
    return left_right_codes


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
    from ...core.data.engine_grid import EngineGrid
    from ...core.data.generic_grid import GenericGrid
    from ..interp_single.interp_features import interpolate_all_fields_no_octree

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


def _mask_generation(
        octree_leaves: OctreeLevel,
        masking_option: MeshExtractionMaskingOptions
) -> np.ndarray | None:
    """
    Generate masks for mesh extraction based on masking options and stack relations.
    
    Args:
        octree_leaves: Octree leaf level containing scalar field outputs
        masking_option: Mesh extraction masking configuration
        
    Returns:
        Matrix of boolean masks for each scalar field
        
    Raises:
        NotImplementedError: For unsupported masking options
        ValueError: For invalid option combinations
    """
    all_scalar_fields_outputs: List[InterpOutput] = octree_leaves.outputs_centers
    n_scalar_fields = len(all_scalar_fields_outputs)
    outputs_ = all_scalar_fields_outputs[0]
    slice_corners = outputs_.grid.corners_grid_slice
    grid_size = outputs_.cornersGrid_values.shape[0]

    mask_matrix = BackendTensor.t.zeros((n_scalar_fields, grid_size // 8), dtype=bool)
    onlap_chain_counter = 0

    for i in range(n_scalar_fields):
        stack_relation = all_scalar_fields_outputs[i].scalar_fields.stack_relation

        match (masking_option, stack_relation):
            case MeshExtractionMaskingOptions.RAW, _:
                mask_matrix[i] = BackendTensor.t.ones(grid_size // 8, dtype=bool)

            case MeshExtractionMaskingOptions.DISJOINT, _:
                raise NotImplementedError(
                    "Disjoint is not supported yet. Not even sure if there is anything to support"
                )

            case MeshExtractionMaskingOptions.INTERSECT, StackRelationType.ERODE:
                mask_array = all_scalar_fields_outputs[i + onlap_chain_counter].squeezed_mask_array
                x = mask_array[slice_corners].reshape((1, -1, 8))
                mask_matrix[i] = BackendTensor.t.sum(x, -1, bool)[0]
                onlap_chain_counter = 0

            case MeshExtractionMaskingOptions.INTERSECT, StackRelationType.BASEMENT:
                mask_array = all_scalar_fields_outputs[i].squeezed_mask_array
                x = mask_array[slice_corners].reshape((1, -1, 8))
                mask_matrix[i] = BackendTensor.t.sum(x, -1, bool)[0]
                onlap_chain_counter = 0

            case MeshExtractionMaskingOptions.INTERSECT, StackRelationType.ONLAP:
                mask_array = all_scalar_fields_outputs[i].squeezed_mask_array
                x = mask_array[slice_corners].reshape((1, -1, 8))
                mask_matrix[i] = BackendTensor.t.sum(x, -1, bool)[0]
                onlap_chain_counter += 1

            case _, StackRelationType.FAULT:
                mask_matrix[i] = BackendTensor.t.ones(grid_size // 8, dtype=bool)

            case _:
                raise ValueError("Invalid combination of options")

    return mask_matrix
