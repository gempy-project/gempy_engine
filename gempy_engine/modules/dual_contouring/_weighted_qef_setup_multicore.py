import numpy as np
import concurrent.futures
from typing import List, Tuple, Optional

from ._find_vertex_overlap import _generate_voxel_codes
from ...core.data.dual_contouring_data import DualContouringData

# Assuming _generate_voxel_codes is available

DEFAULT_CROSS_SURFACE_WEIGHT = 10.0


def _process_single_surface(
        i: int,
        n_surfaces: int,
        surface_cache: list,
        cross_weight: float
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Worker function to process one target surface against all others."""
    if surface_cache[i] is None:
        return i, None, None, None

    codes_i = surface_cache[i]['codes']
    n_valid_i = surface_cache[i]['n_valid']

    extra_xyz_rows, extra_norm_rows, extra_w_rows = [], [], []

    for j in range(n_surfaces):
        if i == j or surface_cache[j] is None:
            continue

        codes_j = surface_cache[j]['codes']

        # Fast C-level intersection. NumPy releases the GIL here!
        common_codes, idx_i_common, idx_j_common = np.intersect1d(
            codes_i, codes_j, assume_unique=True, return_indices=True
        )

        if common_codes.size == 0:
            continue

        # Pre-allocate
        block_xyz = np.zeros((n_valid_i, 12, 3), dtype=np.float64)
        block_norm = np.zeros((n_valid_i, 12, 3), dtype=np.float64)
        block_w = np.zeros((n_valid_i, 12), dtype=np.float64)

        # Direct assignment (GIL released again)
        block_xyz[idx_i_common] = surface_cache[j]['xyz'][idx_j_common]
        block_norm[idx_i_common] = surface_cache[j]['norm'][idx_j_common]

        has_data = np.any(block_norm[idx_i_common] != 0, axis=-1)
        block_w[idx_i_common] = has_data * cross_weight

        extra_xyz_rows.append(block_xyz)
        extra_norm_rows.append(block_norm)
        extra_w_rows.append(block_w)

    # Concatenate local results if we found overlaps
    if extra_xyz_rows:
        return (
                i,
                np.concatenate(extra_xyz_rows, axis=1),
                np.concatenate(extra_norm_rows, axis=1),
                np.concatenate(extra_w_rows, axis=1)
        )

    return i, None, None, None


def find_and_inject_multi_surface_constraints_multicore(
        dc_data_list: List[DualContouringData],
        left_right_per_mesh: List[np.ndarray],
        base_number: tuple[int, int, int],
        cross_weight: float = DEFAULT_CROSS_SURFACE_WEIGHT,
        max_workers: int = None  # None defaults to min(32, os.cpu_count() + 4)
) -> None:
    n_surfaces = len(dc_data_list)
    if n_surfaces < 2:
        return

    # --- 1. Generate codes ---
    voxel_codes_per_surface = _generate_voxel_codes(left_right_per_mesh, base_number)

    # --- 2. PRE-COMPUTE cache sequentially (fast O(N) operation) ---
    surface_cache = []
    for i, dc in enumerate(dc_data_list):
        valid = dc.valid_voxels
        n_valid = int(valid.sum())
        if n_valid == 0:
            surface_cache.append(None)
            continue

        valid_edges_bool = dc.valid_edges[valid] > 0
        tmp_xyz = np.zeros((n_valid, 12, 3), dtype=np.float64)
        tmp_norm = np.zeros((n_valid, 12, 3), dtype=np.float64)

        tmp_xyz[valid_edges_bool] = np.asarray(dc.xyz_on_edge)
        tmp_norm[valid_edges_bool] = np.asarray(dc.gradients)

        surface_cache.append({
                'n_valid': n_valid,
                'xyz'    : tmp_xyz,
                'norm'   : tmp_norm,
                'codes'  : voxel_codes_per_surface[i]
        })

    # --- 3. Parallelize the N^2 intersection loop ---
    # We use ThreadPoolExecutor to share the surface_cache in memory without cloning it.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
                executor.submit(_process_single_surface, i, n_surfaces, surface_cache, cross_weight)
                for i in range(n_surfaces)
        ]

        # Collect and apply results as they finish
        for future in concurrent.futures.as_completed(futures):
            i, extra_xyz, extra_normals, extra_weights = future.result()

            if extra_xyz is not None:
                dc_data_list[i].extra_edge_xyz = extra_xyz
                dc_data_list[i].extra_edge_normals = extra_normals
                dc_data_list[i].extra_weights = extra_weights