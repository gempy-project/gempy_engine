from typing import List, Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ...core.data.dual_contouring_mesh import DualContouringMesh


def apply_overlap_to_surface_pair(
        all_meshes: List['DualContouringMesh'],
        origin_surface_idx: int,
        destination_surface_idx: int,
        voxel_overlaps: Dict[str, dict]
) -> None:
    """Apply vertex sharing and triangle removal for a pair of surfaces."""
    # Try both orders in voxel_overlaps since it's only populated for i < j
    idx_i, idx_j = min(origin_surface_idx, destination_surface_idx), max(origin_surface_idx, destination_surface_idx)
    overlap_key = f"stack_{idx_i}_vs_stack_{idx_j}"

    if overlap_key in voxel_overlaps:
        overlap_data = voxel_overlaps[overlap_key]

        # Determine which indices in overlap_data correspond to origin and destination
        if origin_surface_idx == idx_i:
            origin_indices_key = "indices_in_stack_i"
            dest_indices_key = "indices_in_stack_j"
        else:
            origin_indices_key = "indices_in_stack_j"
            dest_indices_key = "indices_in_stack_i"

        # STEP 1: Vertex Sharing
        _apply_vertex_sharing_ordered(
            all_meshes=all_meshes,
            origin_mesh_idx=origin_surface_idx,
            destination_mesh_idx=destination_surface_idx,
            origin_indices=overlap_data[origin_indices_key],
            destination_indices=overlap_data[dest_indices_key]
        )

        # STEP 2: Conservative Triangle Removal
        _remove_triangles_in_voxels(
            mesh=all_meshes[destination_surface_idx],
            voxel_indices=overlap_data[dest_indices_key],
            mode='all'
        )


def _apply_vertex_sharing_ordered(
        all_meshes: List['DualContouringMesh'],
        origin_mesh_idx: int,
        destination_mesh_idx: int,
        origin_indices: np.ndarray,
        destination_indices: np.ndarray
) -> None:
    if not _are_valid_mesh_indices(all_meshes, origin_mesh_idx, destination_mesh_idx):
        return

    origin_mesh = all_meshes[origin_mesh_idx]
    destination_mesh = all_meshes[destination_mesh_idx]

    # Share vertices from origin to destination
    destination_mesh.vertices[destination_indices] = origin_mesh.vertices[origin_indices]


def _are_valid_mesh_indices(all_meshes: List['DualContouringMesh'], *indices: int) -> bool:
    """Check if all provided mesh indices are valid."""
    return all(0 <= idx < len(all_meshes) for idx in indices)


def _remove_triangles_in_voxels(
        mesh: 'DualContouringMesh',
        voxel_indices: np.ndarray,
        mode: str = 'any'
) -> None:
    """
    Remove triangles from a mesh based on vertex overlap.

    Args:
        mesh: The mesh to modify.
        voxel_indices: Indices of vertices that are in the overlap/fault zone.
        mode: 
            'any': Remove triangle if ANY vertex is in the zone (Aggressive, creates gaps).
            'all': Remove triangle if ALL vertices are in the zone (Conservative, cleans internal).
    """
    if mesh is None or mesh.edges is None or mesh.edges.size == 0:
        return

    if voxel_indices is None or voxel_indices.size == 0:
        return

    # Build a boolean mask for vertex indices that are in the overlap
    is_overlap_vertex = np.zeros(mesh.vertices.shape[0], dtype=bool)
    is_overlap_vertex[voxel_indices] = True

    faces = mesh.edges

    # Vectorized check for faces
    if mode == 'all':
        # Remove only if ALL vertices of the triangle are in the overlap
        # Keep if ANY vertex is outside (preserve bridges)
        to_remove = is_overlap_vertex[faces].all(axis=1)
    else:
        # Legacy/Default behavior: Remove if ANY vertex is in the overlap
        to_remove = is_overlap_vertex[faces].any(axis=1)

    # Keep faces that are NOT marked for removal
    mesh.edges = faces[~to_remove]
