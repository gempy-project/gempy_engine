from typing import Any

import numpy as np

from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData
from .dual_contouring_interface import generate_dual_contouring_vertices


def _compute_vertices(dc_data_per_stack: DualContouringData,
                      debug: bool,
                      i: int,
                      valid_edges_per_surface) -> tuple[DualContouringData, Any]:
    """Compute vertices for a specific surface."""
    valid_edges: np.ndarray = valid_edges_per_surface[i]
    next_surface_edge_idx: int = valid_edges_per_surface[:i + 1].sum()
    if i == 0:
        last_surface_edge_idx = 0
    else:
        last_surface_edge_idx: int = valid_edges_per_surface[:i].sum()
    slice_object: slice = slice(last_surface_edge_idx, next_surface_edge_idx)

    dc_data_per_surface = DualContouringData(
        xyz_on_edge=dc_data_per_stack.xyz_on_edge,
        valid_edges=valid_edges,
        xyz_on_centers=dc_data_per_stack.xyz_on_centers,
        dxdydz=dc_data_per_stack.dxdydz,
        exported_fields_on_edges=dc_data_per_stack.exported_fields_on_edges,
        n_surfaces_to_export=dc_data_per_stack.n_surfaces_to_export,
        tree_depth=dc_data_per_stack.tree_depth
    )

    vertices_numpy = _generate_vertices(dc_data_per_surface, debug, slice_object)
    return dc_data_per_surface, vertices_numpy


def _generate_vertices(dc_data_per_surface: DualContouringData, debug: bool, slice_object: slice) -> Any:
    vertices: np.ndarray = generate_dual_contouring_vertices(
        dc_data_per_stack=dc_data_per_surface,
        slice_surface=slice_object,
        debug=debug
    )
    vertices_numpy = BackendTensor.t.to_numpy(vertices)
    return vertices_numpy
