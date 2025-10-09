import copy
from typing import List, Any

import numpy as np

from gempy_engine.modules.dual_contouring._dual_contouring import compute_dual_contouring

from ...modules.dual_contouring._dual_contouring_v2 import compute_dual_contouring_v2
from ._experimental_water_tight_DC_1 import _experimental_water_tight
from ._mask_buffer import MaskBuffer
from ..interp_single.interp_features import interpolate_all_fields_no_octree
from ...config import DUAL_CONTOURING_VERTEX_OVERLAP
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
from ...modules.dual_contouring._aux import _surface_slicer
from ...modules.dual_contouring.dual_contouring_interface import (find_intersection_on_edge, get_triangulation_codes,
                                                                  get_masked_codes, mask_generation,apply_faults_vertex_overlap)


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
    all_surfaces_intersection = []
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

        all_surfaces_intersection.append(intersection_xyz)
        all_valid_edges.append(valid_edges)

    # * 5) Interpolate on edges for all stacks
    output_on_edges = _interp_on_edges(
        all_surfaces_intersection, data_descriptor, dual_contouring_options, interpolation_input
    )
    
    # endregion

    compute_overlap = (len(all_left_right_codes) > 1) and DUAL_CONTOURING_VERTEX_OVERLAP
    
    # region Vertex gen and triangulation
    left_right_per_mesh = []
    # Generate meshes for each scalar field
    if LEGACY:=False:
        for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
            _compute_meshes_legacy(all_left_right_codes, all_mask_arrays, all_meshes, all_surfaces_intersection, all_valid_edges, n_scalar_field, octree_leaves, options, output_on_edges)
    else:
        dc_data_per_surface_all = []
        for n_scalar_field in range(data_descriptor.stack_structure.n_stacks):
            output: InterpOutput = octree_leaves.outputs_centers[n_scalar_field]
            mask = all_mask_arrays[n_scalar_field]
            n_surfaces_to_export = output.scalar_field_at_sp.shape[0]
            for surface_i in range(n_surfaces_to_export):
                valid_edges = all_valid_edges[n_scalar_field]
                valid_edges_per_surface =valid_edges.reshape((n_surfaces_to_export, -1, 12))
                slice_object = _surface_slicer(surface_i, valid_edges_per_surface)

                dc_data_per_surface = DualContouringData(
                    xyz_on_edge=all_surfaces_intersection[n_scalar_field][slice_object],
                    valid_edges=valid_edges_per_surface[surface_i],
                    xyz_on_centers=(
                            octree_leaves.grid_centers.octree_grid.values if mask is None
                            else octree_leaves.grid_centers.octree_grid.values[mask]
                    ),
                    dxdydz=octree_leaves.grid_centers.octree_dxdydz,
                    left_right_codes=all_left_right_codes[n_scalar_field],
                    gradients=output_on_edges[n_scalar_field][slice_object],
                    n_surfaces_to_export=n_scalar_field,
                    tree_depth=options.number_octree_levels
                )
                
                dc_data_per_surface_all.append(dc_data_per_surface)
                if (compute_overlap):
                    left_right_per_mesh.append(all_left_right_codes[n_scalar_field][dc_data_per_surface.valid_voxels])

        all_meshes = compute_dual_contouring_v2(
            dc_data_list=dc_data_per_surface_all,
        )
    # endregion
        if compute_overlap:
            apply_faults_vertex_overlap(all_meshes, data_descriptor.stack_structure, left_right_per_mesh)

    return all_meshes


def _compute_meshes_legacy(all_left_right_codes: list[Any], all_mask_arrays: np.ndarray,
                           all_meshes: list[DualContouringMesh], all_stack_intersection: list[Any],
                           all_valid_edges: list[Any], n_scalar_field: int,
                           octree_leaves: OctreeLevel, options: InterpolationOptions, output_on_edges: list[np.ndarray]):
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
        gradients=output_on_edges[n_scalar_field],
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
) -> List[np.ndarray]:
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
    
    gradients = []
    slicer_idx_start = 0
    for e,o in enumerate(output_on_edges):
        slicer_idx_end = slicer_idx_start + all_stack_intersection[e].shape[0]
        slicer = slice(slicer_idx_start, slicer_idx_end)
        ef = o.exported_fields
        gradients.append(BackendTensor.t.stack((
                ef.gx_field[slicer], 
                ef.gy_field[slicer],
                ef.gz_field[slicer] 
        ), axis=0).T)  # ! When we are computing the edges for dual contouring there is no surface points
        slicer_idx_start = slicer_idx_end
    
    return gradients