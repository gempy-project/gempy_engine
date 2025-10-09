from typing import List, Dict, Optional

import numpy as np

from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.stacks_structure import StacksStructure


def _apply_fault_relations_to_overlaps(
        all_meshes: List[DualContouringMesh],
        voxel_overlaps: Dict[str, dict],
        stacks_structure: StacksStructure
) -> None:
    """
    Apply fault relations to voxel overlaps by updating mesh vertices.

    Args:
        all_meshes: List of dual contouring meshes
        voxel_overlaps: Dictionary containing overlap information between stacks
        stacks_structure: Structure containing fault relations and stack information
    """
    if stacks_structure.faults_relations is None:
        return

    faults_relations = stacks_structure.faults_relations
    n_stacks = stacks_structure.n_stacks
    surfaces_per_stack = stacks_structure.number_of_surfaces_per_stack_vector

    # Process fault relations
    for origin_stack, destination_stack in _get_fault_pairs(faults_relations, n_stacks):
        surface_range = _get_surface_range(surfaces_per_stack, destination_stack)
        
        for surface_n in surface_range:
            overlap_key = f"stack_{origin_stack}_vs_stack_{surface_n}"
            
            if overlap_key in voxel_overlaps:
                _apply_vertex_sharing(
                    all_meshes=all_meshes, 
                    origin_mesh_idx=origin_stack, 
                    destination_mesh_idx=surface_n, 
                    overlap_data=voxel_overlaps[overlap_key]
                )


def _get_fault_pairs(faults_relations: np.ndarray, n_stacks: int):
    """Generate pairs of stacks that have fault relations."""
    for origin_stack in range(n_stacks):
        for destination_stack in range(n_stacks):
            if faults_relations[origin_stack, destination_stack]:
                yield origin_stack, destination_stack


def _get_surface_range(surfaces_per_stack: np.ndarray, stack_index: int) -> range:
    """Get the range of surfaces for a given stack."""
    return range(
        surfaces_per_stack[stack_index], 
        surfaces_per_stack[stack_index + 1]
    )


def _apply_vertex_sharing(
        all_meshes: List[DualContouringMesh],
        origin_mesh_idx: int,
        destination_mesh_idx: int,
        overlap_data: dict
) -> None:
    """
    Apply vertex sharing between origin and destination meshes based on overlap data.

    Args:
        all_meshes: List of dual contouring meshes
        origin_mesh_idx: Index of mesh that serves as the source of vertices
        destination_mesh_idx: Index of mesh that receives vertices from origin
        overlap_data: Dictionary containing indices and overlap information
    """
    if not _are_valid_mesh_indices(all_meshes, origin_mesh_idx, destination_mesh_idx):
        return

    origin_mesh = all_meshes[origin_mesh_idx]
    destination_mesh = all_meshes[destination_mesh_idx]

    # Share vertices from origin to destination
    origin_indices = overlap_data["indices_in_stack_i"]
    destination_indices = overlap_data["indices_in_stack_j"]
    
    destination_mesh.vertices[destination_indices] = origin_mesh.vertices[origin_indices]


def _are_valid_mesh_indices(all_meshes: List[DualContouringMesh], *indices: int) -> bool:
    """Check if all provided mesh indices are valid."""
    return all(0 <= idx < len(all_meshes) for idx in indices)


def find_repeated_voxels_across_stacks(all_left_right_codes: List[np.ndarray]) -> Dict[str, dict]:
    """
    Find repeated voxels using NumPy operations for efficient processing of large arrays.

    Args:
        all_left_right_codes: List of left_right_codes arrays, one per stack

    Returns:
        Dictionary with detailed overlap analysis between stack pairs
    """
    if not all_left_right_codes:
        return {}

    stack_codes = _generate_voxel_codes(all_left_right_codes)
    return _find_overlaps_between_stacks(stack_codes, all_left_right_codes)


def _generate_voxel_codes(all_left_right_codes: List[np.ndarray]) -> List[np.ndarray]:
    """Generate voxel codes for each stack using packed bit directions."""
    from gempy_engine.modules.dual_contouring.fancy_triangulation import _StaticTriangulationData
    
    pack_directions = _StaticTriangulationData.get_pack_directions_into_bits()
    stack_codes = []
    
    for left_right_codes in all_left_right_codes:
        if len(left_right_codes) > 0:
            voxel_codes = (left_right_codes * pack_directions).sum(axis=1)
            stack_codes.append(voxel_codes)
        else:
            stack_codes.append(np.array([]))
    
    return stack_codes


def _find_overlaps_between_stacks(
        stack_codes: List[np.ndarray], 
        all_left_right_codes: List[np.ndarray]
) -> Dict[str, dict]:
    """Find overlaps between all pairs of stacks."""
    overlaps = {}
    
    for i in range(len(stack_codes)):
        for j in range(i + 1, len(stack_codes)):
            overlap_data = _process_stack_pair(
                codes_i=stack_codes[i], 
                codes_j=stack_codes[j], 
                left_right_i=all_left_right_codes[i], 
                left_right_j=all_left_right_codes[j], 
            )
            
            if overlap_data:
                overlaps[f"stack_{i}_vs_stack_{j}"] = overlap_data
    
    return overlaps


def _process_stack_pair(
        codes_i: np.ndarray, 
        codes_j: np.ndarray,
        left_right_i: np.ndarray, 
        left_right_j: np.ndarray,
) -> Optional[dict]:
    """Process a pair of stacks to find overlapping voxels."""
    if codes_i.size == 0 or codes_j.size == 0:
        return None
    
    common_codes = np.intersect1d(codes_i, codes_j)
    
    if len(common_codes) == 0:
        return None
    
    # Find indices of common voxels in each stack
    indices_i = np.isin(codes_i, common_codes)
    indices_j = np.isin(codes_j, common_codes)
    
    return {
        'common_voxel_codes': common_codes,
        'count': len(common_codes),
        'indices_in_stack_i': np.where(indices_i)[0],
        'indices_in_stack_j': np.where(indices_j)[0],
        'common_binary_codes_i': left_right_i[indices_i],
        'common_binary_codes_j': left_right_j[indices_j]
    }