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
    Apply fault relations to voxel overlaps by updating meshes based on fault relations.

    Instead of sharing vertices across meshes, remove triangles on the non-fault
    surfaces where voxels overlap with the fault surfaces.

    Args:
        all_meshes: List of dual contouring meshes (ordered per-surface)
        voxel_overlaps: Dictionary containing overlap information between per-surface meshes
        stacks_structure: Structure containing fault relations and stack->surface index mapping
    """
    if stacks_structure.faults_relations is None:
        return

    faults_relations = stacks_structure.faults_relations
    n_stacks = stacks_structure.n_stacks
    surfaces_per_stack = stacks_structure.number_of_surfaces_per_stack_vector

    # Map each stack to its contiguous surface index range in all_meshes
    # surfaces_per_stack is assumed to be an exclusive prefix sum array of length n_stacks+1
    def surface_range_for_stack(stack_idx: int) -> range:
        return range(surfaces_per_stack[stack_idx], surfaces_per_stack[stack_idx + 1])

    # For each fault relation, process all pairs of origin (fault) surfaces and destination (affected) surfaces
    for origin_stack, destination_stack in _get_fault_pairs(faults_relations, n_stacks):
        origin_surface_range = surface_range_for_stack(origin_stack)
        destination_surface_range = surface_range_for_stack(destination_stack)

        for origin_surface_idx in origin_surface_range:
            for destination_surface_idx in destination_surface_range:
                overlap_key = f"stack_{origin_surface_idx}_vs_stack_{destination_surface_idx}"
                if overlap_key in voxel_overlaps:
                    # Remove triangles from the destination (non-fault) surface where voxels overlap
                    if True:
                        _remove_triangles_in_voxels(
                            mesh=all_meshes[destination_surface_idx],
                            voxel_indices=voxel_overlaps[overlap_key]["indices_in_stack_j"]
                        )
                    else:
                        _apply_vertex_sharing(
                            all_meshes=all_meshes,
                            origin_mesh_idx=origin_stack,
                            destination_mesh_idx=destination_surface_idx,
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
    Legacy behavior: apply vertex sharing between origin and destination meshes based on overlap data.
    Kept for backward compatibility but not used by default anymore.
    """
    if not _are_valid_mesh_indices(all_meshes, origin_mesh_idx, destination_mesh_idx):
        return

    origin_mesh = all_meshes[origin_mesh_idx]
    destination_mesh = all_meshes[destination_mesh_idx]

    origin_indices = overlap_data["indices_in_stack_i"]
    destination_indices = overlap_data["indices_in_stack_j"]

    # Share vertices from origin to destination
    destination_mesh.vertices[destination_indices] = origin_mesh.vertices[origin_indices]


def _are_valid_mesh_indices(all_meshes: List[DualContouringMesh], *indices: int) -> bool:
    """Check if all provided mesh indices are valid."""
    return all(0 <= idx < len(all_meshes) for idx in indices)


def find_repeated_voxels_across_stacks(all_left_right_codes: List[np.ndarray],
                                       base_numbers: tuple[int, int, int]) -> Dict[str, dict]:
    """
    Find repeated voxels using NumPy operations for efficient processing of large arrays.

    Args:
        all_left_right_codes: List of left_right_codes arrays, one per stack

    Returns:
        Dictionary with detailed overlap analysis between stack pairs
    """
    if not all_left_right_codes:
        return {}

    stack_codes = _generate_voxel_codes(all_left_right_codes, base_numbers)
    return _find_overlaps_between_stacks(stack_codes, all_left_right_codes)


def _generate_voxel_codes(all_left_right_codes: List[np.ndarray], base_numbers) -> List[np.ndarray]:
    """Generate stable voxel codes for each per-surface mesh using packed bit directions.

    Works for arbitrary base numbers (nx, ny, nz may differ).
    """
    from gempy_engine.modules.dual_contouring.fancy_triangulation import _get_pack_factors

    # Ensure we always compute pack factors once and reuse them for all surfaces
    pack_directions = _get_pack_factors(*base_numbers)

    stack_codes = []
    for left_right_array in all_left_right_codes:
        voxel_codes = (left_right_array * pack_directions).sum(axis=1)
        stack_codes.append(voxel_codes)

    return stack_codes


def _find_overlaps_between_stacks(
        stack_codes: List[np.ndarray], 
        all_left_right_codes: List[np.ndarray]
) -> Dict[str, dict]:
    """Find overlaps between all pairs of per-surface meshes (order-sensitive)."""
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
    """Process a pair of per-surface meshes to find overlapping voxels."""
    if codes_i.size == 0 or codes_j.size == 0:
        return None

    # Use intersect1d with assume_unique=False for robustness; ensure same dtype
    # codes_i = codes_i.astype('int64', copy=False)
    # codes_j = codes_j.astype('int64', copy=False)
    common_codes = np.intersect1d(codes_i, codes_j, assume_unique=False)

    if common_codes.size == 0:
        return None

    # Find indices of common voxels in each surface mesh
    indices_i = np.isin(codes_i, common_codes)
    indices_j = np.isin(codes_j, common_codes)

    return {
        'common_voxel_codes': common_codes,
        'count': int(common_codes.size),
        'indices_in_stack_i': np.where(indices_i)[0],
        'indices_in_stack_j': np.where(indices_j)[0],
        'common_binary_codes_i': left_right_i[indices_i],
        'common_binary_codes_j': left_right_j[indices_j]
    }
def _remove_triangles_in_voxels(mesh: DualContouringMesh, voxel_indices: np.ndarray) -> None:
    """Remove triangles from a mesh whose vertices belong to the given voxel indices.

    Each vertex corresponds to one valid voxel. Triangles referencing any of the
    specified voxel indices are removed from the mesh to avoid double surfaces on
    non-fault planes.
    """
    if mesh is None or mesh.edges is None or mesh.edges.size == 0:
        return

    if voxel_indices is None or voxel_indices.size == 0:
        return

    # Build a boolean mask for vertex indices that should be removed
    to_remove = np.zeros(mesh.vertices.shape[0], dtype=bool)
    to_remove[voxel_indices] = True

    faces = mesh.edges
    # Keep faces where none of the vertices are marked for removal
    keep = ~to_remove[faces].any(axis=1)

    mesh.edges = faces[keep]
