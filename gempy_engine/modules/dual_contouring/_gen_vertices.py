from typing import Optional
import numpy as np

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

    # Zero coordinates are valid samples, not missing edge constraints.
    bias_xyz_slice = edges_xyz[:, :12]
    mask = valid_edges_bool[:, :, None]
    sum_valid = (bias_xyz_slice * mask).sum(axis=1)
    count_valid = mask.sum(axis=1)
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
        # The mass-point constraints already make ATA positive definite. Extra
        # origin-centered regularization biases the capped solid's volume.
        strength = 0 if dc_data_per_stack.strict_crossings else 1e-4
        reg = strength * torch.eye(3, device=ATA.device, dtype=ATA.dtype).unsqueeze(0)
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

    if dc_data_per_stack.extent is not None:
        tolerance = snap_tolerance(dc_data_per_stack.dxdydz)
        vertices = _snap_to_extent(vertices, edges_xyz[:, :12], edges_normals[:, :12], valid_edges_bool,
                                   dc_data_per_stack.extent, tolerance, BIAS_STRENGTH)

    if debug:
        dc_data_per_stack.bias_center_mass = edges_xyz[:, 12:].reshape(-1, 3)
        dc_data_per_stack.bias_normals = edges_normals[:, 12:].reshape(-1, 3)

    return vertices


def _snap_to_extent(vertices, edges_xyz, edges_normals, valid_edges, extent, tolerance, bias_strength):
    """Move the vertices of cells whose surface crosses an extent face onto that face.

    A free QEF vertex lies inside its cell, so the open border of the mesh zigzags
    up to a cell away from the extent. For those cells, solve the QEF of the
    crossings lying on the face, restricted to the face plane: the border then
    follows the contour of the surface on the face. Cells on a box edge with
    crossings on both faces are pinned to both planes.
    """
    t = BackendTensor.tfnp
    bounds = BackendTensor.t.array(extent, dtype=BackendTensor.dtype).reshape(3, 2)
    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        bounds = bounds.to(vertices.device)

    # (n_voxels, 12 edges, 3 axes, 2 sides): crossing lies on that extent face
    on_face = (t.abs(edges_xyz[:, :, :, None] - bounds[None, None]) <= tolerance) & valid_edges[:, :, None, None]
    face_hit = on_face.any(axis=1)  # (n_voxels, 3, 2)
    fixed = face_hit.any(axis=2)  # (n_voxels, 3)
    snapped = fixed.any(axis=1)
    if not bool(snapped.any()):
        return vertices

    fixed = fixed[snapped]
    on_face = on_face[snapped]
    face_hit = face_hit[snapped]
    xyz = edges_xyz[snapped]
    free = (~fixed).to(xyz.dtype) if BackendTensor.engine_backend == AvailableBackends.PYTORCH else (~fixed).astype(xyz.dtype)
    fixed_f = 1 - free
    target = t.where(face_hit[:, :, 0], bounds[None, :, 0], bounds[None, :, 1])

    # Only crossings on the snapped faces, with normals projected onto the free axes
    w = (on_face.any(axis=3).any(axis=2)).to(xyz.dtype) if BackendTensor.engine_backend == AvailableBackends.PYTORCH \
        else on_face.any(axis=3).any(axis=2).astype(xyz.dtype)
    normals = edges_normals[snapped] * free[:, None, :]
    mass_point = (xyz * w[:, :, None]).sum(axis=1) / w.sum(axis=1)[:, None]

    ATA = t.einsum("ni,nik,nil->nkl", w, normals, normals)
    ATb = t.einsum("ni,nik,ni->nk", w, normals, (normals * xyz).sum(axis=2))
    diagonal = bias_strength * free + fixed_f
    ATA = ATA + diagonal[:, :, None] * t.eye(3, dtype=xyz.dtype)[None] if BackendTensor.engine_backend != AvailableBackends.PYTORCH \
        else ATA + diagonal[:, :, None] * t.eye(3, dtype=xyz.dtype, device=xyz.device)[None]
    ATb = ATb + bias_strength * free * mass_point + fixed_f * target
    snapped_vertices = t.linalg.solve(ATA, ATb[:, :, None])[:, :, 0]

    vertices = vertices.clone() if hasattr(vertices, "clone") else vertices.copy()
    vertices[snapped] = snapped_vertices
    return vertices


def snap_tolerance(dxdydz):
    """Distance under which a point counts as lying on an extent face."""
    return 1e-3 * float(np.min(BackendTensor.t.to_numpy(dxdydz)))


def triangles_in_extent_face(corners, low, high, tolerance):
    """(T,) True where all three corners (T, 3, 3) lie on one extent face.

    With snapping, the quad around a primal edge leaving an extent face joins
    four snapped vertices, so it lies flat in the face: a zero-area or sliver
    fold along the border, not part of the surface.
    """
    on_low = (abs(corners - low) <= tolerance).all(axis=1)
    on_high = (abs(corners - high) <= tolerance).all(axis=1)
    return (on_low | on_high).any(axis=1)
