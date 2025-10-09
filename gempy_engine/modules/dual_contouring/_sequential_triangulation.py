from typing import Any

import numpy as np
import warnings

from ._aux import _surface_slicer
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
                              ) -> tuple[DualContouringData, Any, Any]:
    """Orchestrator function that combines vertex computation and triangulation."""
    dc_data_per_surface, vertices = _compute_vertices(
        dc_data_per_stack=dc_data_per_stack, 
        debug=debug,
        surface_i=i,
        valid_edges_per_surface=valid_edges_per_surface
    )

    slice_object = _surface_slicer(i, valid_edges_per_surface)

    # * Average gradient for the edges
    valid_edges = valid_edges_per_surface[i]
    edges_normals = BackendTensor.t.zeros((valid_edges.shape[0], 12, 3), dtype=BackendTensor.dtype_obj)
    edges_normals[:] = 0
    edges_normals[valid_edges] = dc_data_per_stack.gradients[slice_object]

    indices_numpy = _compute_triangulation(
        dc_data_per_surface=dc_data_per_surface, 
        left_right_codes=left_right_codes,
        edges_normals=edges_normals,
        vertex= vertices
    )


    vertices_numpy = BackendTensor.t.to_numpy(vertices)
    return dc_data_per_surface, indices_numpy, vertices_numpy


def _compute_triangulation(dc_data_per_surface: DualContouringData, 
                           left_right_codes, edges_normals, vertex) -> Any:
    """Compute triangulation indices for a specific surface."""

    if left_right_codes is None:
        # * Legacy triangulation
        indices = triangulate_dual_contouring(dc_data_per_surface)
    else:
        # * Fancy triangulation 👗
        if BackendTensor.engine_backend != AvailableBackends.PYTORCH:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                voxel_normal = np.nanmean(edges_normals, axis=1)
                voxel_normal = voxel_normal[(~np.isnan(voxel_normal).any(axis=1))]  # drop nans
        else:
            # Assuming edges_normals is a PyTorch tensor
            nan_mask = BackendTensor.t.isnan(edges_normals)
            valid_count = (~nan_mask).sum(dim=1)

            # Replace NaNs with 0 for sum calculation
            safe_normals = edges_normals.clone()
            safe_normals[nan_mask] = 0

            # Compute the sum of non-NaN elements
            safe_normals[:,:,0]
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

        voxels_normals = edges_normals[valid_voxels]
        # nan_mask = BackendTensor.t.isnan(voxels_normals)
        # voxels_normals[nan_mask] = 0

        indices = triangulate(
            left_right_array=left_right_per_surface,
            valid_edges=valid_voxels_per_surface,
            tree_depth=tree_depth_per_surface,
            voxel_normals=voxels_normals,
            vertex=vertex
        )
        # indices = BackendTensor.t.concatenate(indices, axis=0)

    # @on
    indices_numpy = BackendTensor.t.to_numpy(indices)
    return indices_numpy


