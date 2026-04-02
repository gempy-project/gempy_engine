import numpy as np
import concurrent.futures
from typing import List, Tuple, Optional

from ._find_vertex_overlap import _generate_voxel_codes
from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData

# Assuming _generate_voxel_codes is available

DEFAULT_CROSS_SURFACE_WEIGHT = 10.0



def _intersect1d_assume_unique(codes_i, codes_j):
    import torch
    # 1. Concatenate the tensors
    cat = torch.cat([codes_i, codes_j])

    # 2. Sort stably. Stable sort is CRITICAL here: it guarantees that 
    # if an element exists in both, the one from codes_i stays before the one from codes_j.
    sorted_cat, perm = torch.sort(cat, stable=True)

    # 3. Because the arrays are unique internally, any adjacent duplicates 
    # in the sorted array MUST be a match between codes_i and codes_j.
    mask = sorted_cat[1:] == sorted_cat[:-1]

    # 4. Extract the values and indices
    common_codes = sorted_cat[:-1][mask]

    # The first element of the matched pair came from codes_i
    idx_i_common = perm[:-1][mask]

    # The second element came from codes_j. We subtract the length of codes_i 
    # to get its original index in codes_j.
    idx_j_common = perm[1:][mask] - codes_i.numel()

    return common_codes, idx_i_common, idx_j_common


def _process_single_surface(
        i: int,
        n_surfaces: int,
        surface_cache: list,
        cross_weight: float,
        allowed_partners: Optional[set] = None
) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Worker function to process one target surface against its allowed partners.
    
    Args:
        allowed_partners: If provided, only inject constraints from surface indices
            in this set.  When ``None`` (legacy behaviour), all other surfaces are
            considered.
    """
    if surface_cache[i] is None:
        return i, None, None, None

    codes_i = surface_cache[i]['codes']
    n_valid_i = surface_cache[i]['n_valid']

    extra_xyz_rows, extra_norm_rows, extra_w_rows = [], [], []

    for j in range(n_surfaces):
        if i == j or surface_cache[j] is None:
            continue
        if allowed_partners is not None and j not in allowed_partners:
            continue

        codes_j = surface_cache[j]['codes']

        # Fast C-level intersection. NumPy releases the GIL here!
        if BackendTensor.engine_backend is AvailableBackends.PYTORCH:
            common_codes, idx_i_common, idx_j_common = _intersect1d_assume_unique(codes_i, codes_j)
        else:
            common_codes, idx_i_common, idx_j_common = np.intersect1d(
                codes_i, codes_j, assume_unique=True, return_indices=True
            )

        if common_codes.size == 0:
            continue

        # Pre-allocate
        block_xyz = BackendTensor.t.zeros((n_valid_i, 12, 3))
        block_norm = BackendTensor.t.zeros((n_valid_i, 12, 3))
        block_w = BackendTensor.t.zeros((n_valid_i, 12))

        # Direct assignment (GIL released again)
        block_xyz[idx_i_common] = surface_cache[j]['xyz'][idx_j_common]
        block_norm[idx_i_common] = surface_cache[j]['norm'][idx_j_common]

        has_data = BackendTensor.t.any(block_norm[idx_i_common] != 0, axis=-1)
        block_w[idx_i_common] = has_data * cross_weight

        extra_xyz_rows.append(block_xyz)
        extra_norm_rows.append(block_norm)
        extra_w_rows.append(block_w)

    # Concatenate local results if we found overlaps
    if extra_xyz_rows:
        return (
                i,
                BackendTensor.t.concatenate(extra_xyz_rows, axis=1),
                BackendTensor.t.concatenate(extra_norm_rows, axis=1),
                BackendTensor.t.concatenate(extra_w_rows, axis=1)
        )

    return i, None, None, None


def _build_allowed_partners(
        surface_to_stack: Optional[List[int]],
        faults_relations: Optional[np.ndarray],
        n_surfaces: int
) -> Optional[List[Optional[set]]]:
    """Build per-surface sets of partner surface indices based on geological relations.
    
    A surface *i* should only exchange QEF constraints with surface *j* when their
    respective stacks overlap.  Surfaces within the same stack are skipped.
    Fault–fault pairs (both stacks are fault stacks) are also excluded so that
    independent faults do not distort each other's QEF.
    
    Returns ``None`` when no filtering should be applied (legacy behaviour).
    """
    if surface_to_stack is None or faults_relations is None:
        return None

    # Determine which stacks are fault stacks (i.e. act as a fault on any other stack)
    n_stacks = faults_relations.shape[0]
    is_fault_stack = set()
    for fs in range(n_stacks):
        for ds in range(n_stacks):
            if faults_relations[fs, ds]:
                is_fault_stack.add(fs)
                break

    partners: List[Optional[set]] = [None] * n_surfaces
    for i in range(n_surfaces):
        si = surface_to_stack[i]
        allowed = set()
        for j in range(n_surfaces):
            if i == j:
                continue
            sj = surface_to_stack[j]
            if si == sj:
                continue
            # Skip fault–fault pairs: faults should not affect each other
            if si in is_fault_stack and sj in is_fault_stack:
                continue
            allowed.add(j)
        partners[i] = allowed if allowed else None
    return partners


def find_and_inject_multi_surface_constraints_multicore(
        dc_data_list: List[DualContouringData],
        left_right_per_mesh: List[np.ndarray],
        base_number: tuple[int, int, int],
        cross_weight: float = DEFAULT_CROSS_SURFACE_WEIGHT,
        max_workers: int = None,  # None defaults to min(32, os.cpu_count() + 4)
        surface_to_stack: Optional[List[int]] = None,
        faults_relations: Optional[np.ndarray] = None
) -> None:
    n_surfaces = len(dc_data_list)
    if n_surfaces < 2:
        return

    # --- 0. Determine allowed pairwise partners (improvement 4.1 + 4.2) ---
    allowed_partners_per_surface = _build_allowed_partners(
        surface_to_stack, faults_relations, n_surfaces
    )

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
        tmp_xyz = BackendTensor.t.zeros((n_valid, 12, 3))
        tmp_norm = BackendTensor.t.zeros((n_valid, 12, 3))

        tmp_xyz[valid_edges_bool] = dc.xyz_on_edge
        tmp_norm[valid_edges_bool] = dc.gradients

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
                executor.submit(
                    _process_single_surface, i, n_surfaces, surface_cache, cross_weight,
                    allowed_partners_per_surface[i] if allowed_partners_per_surface is not None else None
                )
                for i in range(n_surfaces)
        ]

        # Collect and apply results as they finish
        for future in concurrent.futures.as_completed(futures):
            i, extra_xyz, extra_normals, extra_weights = future.result()

            if extra_xyz is not None:
                dc_data_list[i].extra_edge_xyz = extra_xyz
                dc_data_list[i].extra_edge_normals = extra_normals
                dc_data_list[i].extra_weights = extra_weights
