from typing import Any, Optional

import numpy as np

from ._aux import _surface_slicer
from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData


def _compute_vertices(dc_data_per_stack: DualContouringData,
                      debug: bool,
                      surface_i: int,
                      valid_edges_per_surface) -> tuple[DualContouringData, Any]:
    """Compute vertices for a specific surface."""
    valid_edges: np.ndarray = valid_edges_per_surface[surface_i]

    slice_object = _surface_slicer(surface_i, valid_edges_per_surface)

    dc_data_per_surface = DualContouringData(
        xyz_on_edge=dc_data_per_stack.xyz_on_edge,
        valid_edges=valid_edges,
        xyz_on_centers=dc_data_per_stack.xyz_on_centers,
        dxdydz=dc_data_per_stack.dxdydz,
        gradients=dc_data_per_stack.gradients,
        n_surfaces_to_export=dc_data_per_stack.n_surfaces_to_export,
        tree_depth=dc_data_per_stack.tree_depth
    )

    vertices_numpy = generate_dual_contouring_vertices(dc_data_per_surface, slice_object, debug)
    return dc_data_per_surface, vertices_numpy


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

    # * Coordinates for all posible edges (12) and 3 dummy edges_normals in the center
    edges_xyz = BackendTensor.tfnp.zeros((n_edges, 15, 3), dtype=BackendTensor.dtype_obj)
    valid_edges = valid_edges > 0
    edges_xyz[:, :12][valid_edges] = xyz_on_edge

    # Normals
    edges_normals = BackendTensor.tfnp.zeros((n_edges, 15, 3), dtype=BackendTensor.dtype_obj)
    edges_normals[:, :12][valid_edges] = gradients

    if OLD_METHOD := False:
        # ! Moureze model does not seems to work with the new method
        # ! This branch is all nans at least with ch1_1 model
        bias_xyz = BackendTensor.tfnp.copy(edges_xyz[:, :12])
        isclose = BackendTensor.tfnp.isclose(bias_xyz, 0)
        bias_xyz[isclose] = BackendTensor.tfnp.nan  # zero values to nans
        mass_points = BackendTensor.tfnp.nanmean(bias_xyz, axis=1)  # Mean ignoring nans
    else:  # ? This is actually doing something
        bias_xyz = BackendTensor.tfnp.copy(edges_xyz[:, :12])
        if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
            # PyTorch doesn't have masked arrays, so we'll use a different approach
            mask = bias_xyz == 0
            # Replace zeros with NaN for mean calculation
            bias_xyz_masked = BackendTensor.tfnp.where(mask, float('nan'), bias_xyz)
            mass_points = BackendTensor.tfnp.nanmean(bias_xyz_masked, axis=1)
        else:
            # NumPy approach with masked arrays
            bias_xyz = BackendTensor.tfnp.to_numpy(bias_xyz)
            import numpy as np
            mask = bias_xyz == 0
            masked_arr = np.ma.masked_array(bias_xyz, mask)
            mass_points = masked_arr.mean(axis=1)
            mass_points = BackendTensor.tfnp.array(mass_points)

    edges_xyz[:, 12] = mass_points
    edges_xyz[:, 13] = mass_points
    edges_xyz[:, 14] = mass_points

    BIAS_STRENGTH = 1

    bias_x = BackendTensor.tfnp.array([BIAS_STRENGTH, 0, 0], dtype=BackendTensor.dtype_obj)
    bias_y = BackendTensor.tfnp.array([0, BIAS_STRENGTH, 0], dtype=BackendTensor.dtype_obj)
    bias_z = BackendTensor.tfnp.array([0, 0, BIAS_STRENGTH], dtype=BackendTensor.dtype_obj)

    edges_normals[:, 12] = bias_x
    edges_normals[:, 13] = bias_y
    edges_normals[:, 14] = bias_z

    # Remove unused voxels
    edges_xyz = edges_xyz[valid_voxels]
    edges_normals = edges_normals[valid_voxels]

    # Compute LSTSQS in all voxels at the same time
    A = edges_normals
    b = (A * edges_xyz).sum(axis=2)

    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        transpose_shape = (2, 1, 0)  # For PyTorch: (batch, dim2, dim1)
    else:
        transpose_shape = (0, 2, 1)  # For NumPy: (batch, dim2, dim1)

    term1 = BackendTensor.tfnp.einsum("ijk, ilj->ikl", A, BackendTensor.tfnp.transpose(A, transpose_shape))
    term2 = BackendTensor.tfnp.linalg.inv(term1)
    term3 = BackendTensor.tfnp.einsum("ijk,ik->ij", BackendTensor.tfnp.transpose(A, transpose_shape), b)
    vertices = BackendTensor.tfnp.einsum("ijk, ij->ik", term2, term3)

    if debug:
        dc_data_per_stack.bias_center_mass = edges_xyz[:, 12:].reshape(-1, 3)
        dc_data_per_stack.bias_normals = edges_normals[:, 12:].reshape(-1, 3)

    return vertices
