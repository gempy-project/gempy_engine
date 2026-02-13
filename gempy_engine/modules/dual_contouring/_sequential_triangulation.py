from typing import Any

from ._aux import _surface_slicer
from ._gen_vertices import _compute_vertices
from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData
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
        vertex=vertices
    )

    vertices_numpy = BackendTensor.t.to_numpy(vertices)
    return dc_data_per_surface, indices_numpy, vertices_numpy


def _compute_triangulation(dc_data_per_surface: DualContouringData,
                           left_right_codes, edges_normals, vertex) -> Any:
    """Compute triangulation indices for a specific surface."""

    # * Fancy triangulation 👗
    valid_voxels = dc_data_per_surface.valid_voxels

    left_right_per_surface = left_right_codes[valid_voxels]
    valid_voxels_per_surface = dc_data_per_surface.valid_edges[valid_voxels]
    tree_depth_per_surface = dc_data_per_surface.tree_depth

    voxels_normals = edges_normals[valid_voxels]

    indices = triangulate(
        left_right_array=left_right_per_surface,
        valid_edges=valid_voxels_per_surface,
        tree_depth=tree_depth_per_surface,
        voxel_normals=voxels_normals,
        vertex=vertex,
        base_number=dc_data_per_surface.base_number
    )

    # @on
    indices_numpy = BackendTensor.t.to_numpy(indices)
    return indices_numpy
