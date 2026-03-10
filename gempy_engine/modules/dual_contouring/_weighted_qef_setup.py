"""Detect multi-surface voxels and populate extra weighted QEF constraints on DualContouringData."""
from typing import List, Optional

import numpy as np

from ...core.data.dual_contouring_data import DualContouringData
from ._find_vertex_overlap import _generate_voxel_codes


DEFAULT_CROSS_SURFACE_WEIGHT = 10.0


def find_and_inject_multi_surface_constraints(
        dc_data_list: List[DualContouringData],
        left_right_per_mesh: List[np.ndarray],
        base_number: tuple[int, int, int],
        cross_weight: float = DEFAULT_CROSS_SURFACE_WEIGHT,
) -> None:
    """Detect voxels shared by multiple surfaces and inject weighted constraints.

    For every pair of surfaces that share voxels, the edge intersection data
    (xyz and normals) from the *other* surface is written into the
    ``extra_edge_xyz / extra_edge_normals / extra_weights`` fields of each
    ``DualContouringData`` so that the QEF solver can produce vertices that
    lie close to both surfaces.

    The function mutates *dc_data_list* in-place (populates the extra_* fields).
    """
    n_surfaces = len(dc_data_list)
    if n_surfaces < 2:
        return

    # --- 1. Build voxel codes for every surface (only valid voxels) ----------
    voxel_codes_per_surface: List[np.ndarray] = _generate_voxel_codes(
        left_right_per_mesh, base_number
    )

    # --- 2. For each surface, collect extra constraints from all others ------
    for i in range(n_surfaces):
        dc_i = dc_data_list[i]
        valid_i = dc_i.valid_voxels
        n_valid_i = int(valid_i.sum())
        if n_valid_i == 0:
            continue

        codes_i = voxel_codes_per_surface[i]

        # Accumulate extra rows from all other surfaces
        extra_xyz_rows: List[np.ndarray] = []   # each (n_valid_i, K_j, 3)
        extra_norm_rows: List[np.ndarray] = []
        extra_w_rows: List[np.ndarray] = []

        for j in range(n_surfaces):
            if j == i:
                continue
            dc_j = dc_data_list[j]
            valid_j = dc_j.valid_voxels
            n_valid_j = int(valid_j.sum())
            if n_valid_j == 0:
                continue

            codes_j = voxel_codes_per_surface[j]

            # Find common voxel codes
            common_codes = np.intersect1d(codes_i, codes_j, assume_unique=False)
            if common_codes.size == 0:
                continue

            # Build full edge data for surface j (same logic as _gen_vertices)
            _inject_constraints_from_j(
                dc_i=dc_i,
                dc_j=dc_j,
                codes_i=codes_i,
                codes_j=codes_j,
                common_codes=common_codes,
                n_valid_i=n_valid_i,
                extra_xyz_rows=extra_xyz_rows,
                extra_norm_rows=extra_norm_rows,
                extra_w_rows=extra_w_rows,
                weight=cross_weight,
            )

        # Concatenate all extra constraints along the K dimension
        if extra_xyz_rows:
            dc_i.extra_edge_xyz = np.concatenate(extra_xyz_rows, axis=1)       # (n_valid_i, K_total, 3)
            dc_i.extra_edge_normals = np.concatenate(extra_norm_rows, axis=1)
            dc_i.extra_weights = np.concatenate(extra_w_rows, axis=1)          # (n_valid_i, K_total)


def _inject_constraints_from_j(
        dc_i: DualContouringData,
        dc_j: DualContouringData,
        codes_i: np.ndarray,
        codes_j: np.ndarray,
        common_codes: np.ndarray,
        n_valid_i: int,
        extra_xyz_rows: list,
        extra_norm_rows: list,
        extra_w_rows: list,
        weight: float,
) -> None:
    """Build the 12-edge constraint block from surface *j* for surface *i*."""
    from ...core.backend_tensor import BackendTensor

    valid_j = dc_j.valid_voxels

    # Reconstruct full (n_all_voxels, 12, 3) arrays for surface j
    # edges_xyz_j_full = BackendTensor.t.zeros((dc_j.valid_edges.shape[0], 12, 3), dtype='float64')
    # edges_norm_j_full = BackendTensor.t.zeros((dc_j.valid_edges.shape[0], 12, 3), dtype='float64')

    valid_edges_bool_j = dc_j.valid_edges[valid_j] > 0
    # Temporary dense arrays for valid voxels only
    n_valid_j = int(valid_j.sum())
    tmp_xyz = np.zeros((n_valid_j, 12, 3), dtype='float64')
    tmp_norm = np.zeros((n_valid_j, 12, 3), dtype='float64')
    tmp_xyz[valid_edges_bool_j] = np.asarray(dc_j.xyz_on_edge)
    tmp_norm[valid_edges_bool_j] = np.asarray(dc_j.gradients)

    # Map common codes → local indices in i and j (among valid voxels)
    mask_i = np.isin(codes_i, common_codes)
    mask_j = np.isin(codes_j, common_codes)

    # Sort both by their voxel code so rows align
    order_i = np.argsort(codes_i[mask_i])
    order_j = np.argsort(codes_j[mask_j])

    idx_i_common = np.where(mask_i)[0][order_i]  # indices into valid-voxel arrays of surface i
    idx_j_common = np.where(mask_j)[0][order_j]

    # Build (n_valid_i, 12, 3) blocks – zeros for non-overlapping voxels
    block_xyz = np.zeros((n_valid_i, 12, 3), dtype='float64')
    block_norm = np.zeros((n_valid_i, 12, 3), dtype='float64')
    block_w = np.zeros((n_valid_i, 12), dtype='float64')

    block_xyz[idx_i_common] = tmp_xyz[idx_j_common]
    block_norm[idx_i_common] = tmp_norm[idx_j_common]

    # Weight is applied only where the other surface actually has edge data
    has_data = np.any(block_norm[idx_i_common] != 0, axis=-1)  # (n_common, 12)
    w_block = np.zeros((n_valid_i, 12), dtype='float64')
    w_block[idx_i_common] = has_data * weight

    extra_xyz_rows.append(block_xyz)
    extra_norm_rows.append(block_norm)
    extra_w_rows.append(w_block)
