import torch
import numpy as np
from typing import List

from ._find_vertex_overlap import _generate_voxel_codes
from ...core.data.dual_contouring_data import DualContouringData

# Assuming _generate_voxel_codes is updated or returns something we can cast to tensor

DEFAULT_CROSS_SURFACE_WEIGHT = 10.0


def find_and_inject_multi_surface_constraints_pt(
        dc_data_list: List[DualContouringData],
        left_right_per_mesh: List[np.ndarray],
        base_number: tuple[int, int, int],
        cross_weight: float = DEFAULT_CROSS_SURFACE_WEIGHT,
        device: str = 'cuda'  # Push everything here
) -> None:
    n_surfaces = len(dc_data_list)
    if n_surfaces < 2:
        return

    # --- 1. Generate codes and move to GPU ---
    cpu_codes = _generate_voxel_codes(left_right_per_mesh, base_number)
    codes_gpu = [torch.as_tensor(c, device=device) for c in cpu_codes]

    # --- 2. PRE-COMPUTE dense tensors for all surfaces (The big speedup) ---
    surface_cache = []
    for i, dc in enumerate(dc_data_list):
        valid = torch.as_tensor(dc.valid_voxels, device=device)
        n_valid = int(valid.sum().item())

        if n_valid == 0:
            surface_cache.append(None)
            continue

        # Build dense xyz and norm blocks ONCE per surface
        valid_edges_bool = torch.as_tensor(dc.valid_edges, device=device)[valid] > 0

        tmp_xyz = torch.zeros((n_valid, 12, 3), dtype=torch.float64, device=device)
        tmp_norm = torch.zeros((n_valid, 12, 3), dtype=torch.float64, device=device)

        tmp_xyz[valid_edges_bool] = torch.as_tensor(dc.xyz_on_edge, dtype=torch.float64, device=device)
        tmp_norm[valid_edges_bool] = torch.as_tensor(dc.gradients, dtype=torch.float64, device=device)

        surface_cache.append({
                'n_valid': n_valid,
                'xyz'    : tmp_xyz,
                'norm'   : tmp_norm
        })

    # --- 3. Vectorized N^2 Loop on GPU ---
    for i in range(n_surfaces):
        if surface_cache[i] is None:
            continue

        codes_i = codes_gpu[i]
        n_valid_i = surface_cache[i]['n_valid']

        extra_xyz_rows, extra_norm_rows, extra_w_rows = [], [], []

        for j in range(n_surfaces):
            if i == j or surface_cache[j] is None:
                continue

            codes_j = codes_gpu[j]

            # Fast GPU intersection
            mask_i = torch.isin(codes_i, codes_j)
            if not mask_i.any():
                continue

            mask_j = torch.isin(codes_j, codes_i)

            # Align common codes. Since codes are unique per surface, 
            # sorting the masked codes guarantees 1:1 alignment.
            order_i = torch.argsort(codes_i[mask_i])
            order_j = torch.argsort(codes_j[mask_j])

            # Map back to indices in the valid-voxel arrays
            idx_i_common = torch.nonzero(mask_i, as_tuple=True)[0][order_i]
            idx_j_common = torch.nonzero(mask_j, as_tuple=True)[0][order_j]

            # Allocate destination blocks for this i, j pair
            block_xyz = torch.zeros((n_valid_i, 12, 3), dtype=torch.float64, device=device)
            block_norm = torch.zeros((n_valid_i, 12, 3), dtype=torch.float64, device=device)
            block_w = torch.zeros((n_valid_i, 12), dtype=torch.float64, device=device)

            # Inject data
            block_xyz[idx_i_common] = surface_cache[j]['xyz'][idx_j_common]
            block_norm[idx_i_common] = surface_cache[j]['norm'][idx_j_common]

            # Compute weights based on where normals are non-zero
            has_data = (block_norm[idx_i_common] != 0).any(dim=-1)
            block_w[idx_i_common] = has_data.to(torch.float64) * cross_weight

            extra_xyz_rows.append(block_xyz)
            extra_norm_rows.append(block_norm)
            extra_w_rows.append(block_w)

        # Concatenate and push back to CPU/NumPy if the rest of your pipeline expects it
        if extra_xyz_rows:
            dc_data_list[i].extra_edge_xyz = torch.cat(extra_xyz_rows, dim=1).cpu().numpy()
            dc_data_list[i].extra_edge_normals = torch.cat(extra_norm_rows, dim=1).cpu().numpy()
            dc_data_list[i].extra_weights = torch.cat(extra_w_rows, dim=1).cpu().numpy()
