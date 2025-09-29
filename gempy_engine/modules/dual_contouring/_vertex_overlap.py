from typing import List

import numpy as np

from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.stacks_structure import StacksStructure


def _apply_fault_relations_to_overlaps(
        all_meshes: List[DualContouringMesh],
        voxel_overlaps: dict,
        stacks_structure: StacksStructure
) -> None:
    """
    Apply fault relations to voxel overlaps by updating mesh vertices.

    Args:
        all_meshes: List of dual contouring meshes
        faults_relations: Boolean matrix indicating fault relationships between stacks
        voxel_overlaps: Dictionary containing overlap information between stacks
        n_stacks: Total number of stacks
    """
    faults_relations = stacks_structure.faults_relations
    n_stacks = stacks_structure.n_stacks
    number_surfaces_per_stack = stacks_structure.number_of_surfaces_per_stack_vector

    if faults_relations is None:
        return

    # Iterate through fault relations matrix
    for origin_stack in range(n_stacks):
        for destination_stack in range(n_stacks):
            if faults_relations[origin_stack, destination_stack]:
                for surface_n in range(number_surfaces_per_stack[destination_stack], number_surfaces_per_stack[destination_stack + 1]):
                    overlap_key = f"stack_{origin_stack}_vs_stack_{surface_n}"

                    # Check if there are actual overlaps between these stacks
                    if overlap_key in voxel_overlaps:
                        _apply_vertex_sharing(all_meshes, origin_stack, surface_n, voxel_overlaps[overlap_key])


def _apply_vertex_sharing(
        all_meshes: List[DualContouringMesh],
        origin_mesh: int,
        destination_mesh: int,
        overlap_data: dict
) -> None:
    """
    Apply vertex sharing between origin and destination meshes based on overlap data.

    Args:
        all_meshes: List of dual contouring meshes
        origin_mesh: Stack index that serves as the source of vertices
        destination_mesh: Stack index that receives vertices from origin
        overlap_data: Dictionary containing indices and overlap information
        mesh_indices_offset: Starting mesh index for each stack
    """
    # origin_mesh_idx = mesh_indices_offset[origin_stack]
    # destination_mesh_idx = mesh_indices_offset[destination_stack]
    origin_mesh_idx = origin_mesh
    destination_mesh_idx = destination_mesh

    # Ensure mesh indices are valid
    if origin_mesh_idx >= len(all_meshes) or destination_mesh_idx >= len(all_meshes):
        return

    # Apply the vertex sharing (same logic as original _f function)
    origin_mesh = all_meshes[origin_mesh_idx]
    destination_mesh = all_meshes[destination_mesh_idx]

    indices_in_origin = overlap_data["indices_in_stack_i"]
    indices_in_destination = overlap_data["indices_in_stack_j"]

    destination_mesh.vertices[indices_in_destination] = origin_mesh.vertices[indices_in_origin]


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
