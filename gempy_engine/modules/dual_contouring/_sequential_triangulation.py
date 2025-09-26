from typing import Any

import numpy as np
import warnings

from ...config import AvailableBackends
from ._gen_vertices import _compute_vertices
from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData
from ._triangulate import triangulate_dual_contouring
from ...modules.dual_contouring.fancy_triangulation import triangulate


def _sequential_triangulation(dc_data_per_stack: DualContouringData,
                              debug: bool,
                              i: int,
                              left_right_codes,
                              valid_edges_per_surface,
                              compute_indices=True
                              ) -> tuple[Any, Any]:
    """Orchestrator function that combines vertex computation and triangulation."""
    dc_data_per_surface, vertices_numpy = _compute_vertices(
        dc_data_per_stack, debug, i, valid_edges_per_surface
    )

    if not compute_indices:
        return None, vertices_numpy

    indices_numpy = _compute_triangulation(
        dc_data_per_surface, dc_data_per_stack, i, left_right_codes, valid_edges_per_surface
    )

    return indices_numpy, vertices_numpy


def _compute_triangulation(dc_data_per_surface: DualContouringData,
                           dc_data_per_stack: DualContouringData,
                           i: int,
                           left_right_codes,
                           valid_edges_per_surface) -> Any:
    """Compute triangulation indices for a specific surface."""
    valid_edges: np.ndarray = valid_edges_per_surface[i]
    next_surface_edge_idx: int = valid_edges_per_surface[:i + 1].sum()
    if i == 0:
        last_surface_edge_idx = 0
    else:
        last_surface_edge_idx: int = valid_edges_per_surface[:i].sum()
    slice_object: slice = slice(last_surface_edge_idx, next_surface_edge_idx)

    if left_right_codes is None:
        # * Legacy triangulation
        indices = triangulate_dual_contouring(dc_data_per_surface)
    else:
        # * Fancy triangulation 👗

        # * Average gradient for the edges
        edges_normals = BackendTensor.t.zeros((valid_edges.shape[0], 12, 3), dtype=BackendTensor.dtype_obj)
        edges_normals[:] = np.nan
        edges_normals[valid_edges] = dc_data_per_stack.gradients[slice_object]

        # if LEGACY:=True:
        if BackendTensor.engine_backend != AvailableBackends.PYTORCH:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                voxel_normal = np.nanmean(edges_normals, axis=1)
                voxel_normal = voxel_normal[(~np.isnan(voxel_normal).any(axis=1))]  # drop nans
                pass
        else:
            # Assuming edges_normals is a PyTorch tensor
            nan_mask = BackendTensor.t.isnan(edges_normals)
            valid_count = (~nan_mask).sum(dim=1)

            # Replace NaNs with 0 for sum calculation
            safe_normals = edges_normals.clone()
            safe_normals[nan_mask] = 0

            # Compute the sum of non-NaN elements
            sum_normals = BackendTensor.t.sum(safe_normals, 1)

            # Calculate the mean, avoiding division by zero
            voxel_normal = sum_normals / valid_count.clamp(min=1)

            # Remove rows where all elements were NaN (and hence valid_count is 0)
            voxel_normal = voxel_normal[valid_count > 0].reshape(-1, 3)

        valid_voxels = dc_data_per_surface.valid_voxels

        left_right_per_surface = left_right_codes[valid_voxels]
        valid_voxels_per_surface = dc_data_per_surface.valid_edges[valid_voxels]
        voxel_normal_per_surface = voxel_normal
        tree_depth_per_surface = dc_data_per_surface.tree_depth

        indices = triangulate(
            left_right_array=left_right_per_surface,
            valid_edges=valid_voxels_per_surface,
            tree_depth=tree_depth_per_surface,
            voxel_normals=voxel_normal_per_surface
        )
        indices = BackendTensor.t.concatenate(indices, axis=0)

    # @on
    indices_numpy = BackendTensor.t.to_numpy(indices)
    return indices_numpy

