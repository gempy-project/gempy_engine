from typing import Optional

from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData


def generate_dual_contouring_vertices(dc_data_per_stack: DualContouringData, slice_surface: Optional[slice] = None, debug: bool = False):
    # @off
    n_edges = dc_data_per_stack.n_valid_edges
    valid_edges = dc_data_per_stack.valid_edges
    valid_voxels = dc_data_per_stack.valid_voxels
    if slice_surface is not None:
        xyz_on_edge = dc_data_per_stack.xyz_on_edge[slice_surface]
        gradients = dc_data_per_stack.gradients[slice_surface]
    else:
        xyz_on_edge = dc_data_per_stack.xyz_on_edge
        gradients = dc_data_per_stack.gradients
    # @on

    n_valid_voxels = BackendTensor.tfnp.sum(valid_voxels)
    edges_xyz = BackendTensor.tfnp.zeros((n_valid_voxels, 15, 3), dtype=BackendTensor.dtype_obj)
    edges_normals = BackendTensor.tfnp.zeros((n_valid_voxels, 15, 3), dtype=BackendTensor.dtype_obj)
    
    # Filter valid_edges to only valid voxels
    valid_edges_bool = valid_edges[valid_voxels] > 0
    
    # Assign edge data (now only to valid voxels)
    edges_xyz[:, :12][valid_edges_bool] = xyz_on_edge
    edges_normals[:, :12][valid_edges_bool] = gradients

    # Use nanmean directly without intermediate copy
    bias_xyz_slice = edges_xyz[:, :12]
    
    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        mask = bias_xyz_slice == 0
        bias_xyz_masked = BackendTensor.tfnp.where(mask, float('nan'), bias_xyz_slice)
        mass_points = BackendTensor.tfnp.nanmean(bias_xyz_masked, axis=1)
    else:
        # NumPy: more efficient approach using sum and count
        mask = bias_xyz_slice != 0
        sum_valid = (bias_xyz_slice * mask).sum(axis=1)
        count_valid = mask.sum(axis=1)
        # Avoid division by zero
        count_valid = BackendTensor.tfnp.maximum(count_valid, 1)
        mass_points = sum_valid / count_valid

    # Assign mass points to bias positions
    edges_xyz[:, 12:15] = mass_points[:, None, :]

    BIAS_STRENGTH = 1
    bias_normals = BackendTensor.tfnp.array([
        [BIAS_STRENGTH, 0, 0],
        [0, BIAS_STRENGTH, 0],
        [0, 0, BIAS_STRENGTH]
    ], dtype=BackendTensor.dtype_obj)
    
    edges_normals[:, 12:15] = bias_normals[None, :, :]

    # --- Append extra weighted constraints from other surfaces (if any) ---
    if dc_data_per_stack.extra_edge_xyz is not None:
        K = dc_data_per_stack.extra_edge_xyz.shape[1]
        extra_xyz = BackendTensor.tfnp.array(dc_data_per_stack.extra_edge_xyz, dtype=BackendTensor.dtype_obj)
        extra_norm = BackendTensor.tfnp.array(dc_data_per_stack.extra_edge_normals, dtype=BackendTensor.dtype_obj)
        extra_w = BackendTensor.tfnp.array(dc_data_per_stack.extra_weights, dtype=BackendTensor.dtype_obj)

        edges_xyz = BackendTensor.tfnp.concatenate([edges_xyz, extra_xyz], axis=1)
        edges_normals = BackendTensor.tfnp.concatenate([edges_normals, extra_norm], axis=1)

        n_rows = 15 + K
        w = BackendTensor.tfnp.ones((n_valid_voxels, n_rows), dtype=BackendTensor.dtype_obj)
        w[:, 12:15] = BIAS_STRENGTH
        w[:, 15:] = extra_w
    else:
        n_rows = 15
        w = BackendTensor.tfnp.ones((n_valid_voxels, n_rows), dtype=BackendTensor.dtype_obj)
        w[:, 12:15] = BIAS_STRENGTH

    # Apply sqrt(w) scaling for weighted QEF
    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        W_sqrt = BackendTensor.tfnp.sqrt(w).unsqueeze(-1)
    else:
        W_sqrt = BackendTensor.tfnp.sqrt(w)[..., None]

    A = edges_normals * W_sqrt

    # Compute A^T @ A more efficiently
    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        # For PyTorch: use bmm (batch matrix multiply) which is optimized
        A_T = A.transpose(1, 2)
        ATA = BackendTensor.tfnp.matmul(A_T, A)  # (n_voxels, 3, 3)
        
        # Compute A^T @ (A * edges_xyz).sum(axis=2)
        b = (A * edges_xyz).sum(axis=2)  # (n_voxels, 15)
        ATb = BackendTensor.tfnp.matmul(A_T, b.unsqueeze(-1)).squeeze(-1)  # (n_voxels, 3)
        
        # Solve ATA @ x = ATb  (use solve instead of inv for numerical stability)
        import torch
        reg = 1e-4 * torch.eye(3, device=ATA.device, dtype=ATA.dtype).unsqueeze(0)
        vertices = torch.linalg.solve(ATA + reg, ATb)
    else:
        # NumPy: use efficient einsum
        b = (A * edges_xyz).sum(axis=2)
        
        # A^T @ A
        ATA = BackendTensor.tfnp.einsum("ijk,ijl->ikl", A, A)
        # A^T @ b
        ATb = BackendTensor.tfnp.einsum("ijk,ij->ik", A, b)
        
        # Solve
        ATA_inv = BackendTensor.tfnp.linalg.inv(ATA)
        vertices = BackendTensor.tfnp.einsum("ijk,ij->ik", ATA_inv, ATb)

    if debug:
        dc_data_per_stack.bias_center_mass = edges_xyz[:, 12:].reshape(-1, 3)
        dc_data_per_stack.bias_normals = edges_normals[:, 12:].reshape(-1, 3)

    return vertices
