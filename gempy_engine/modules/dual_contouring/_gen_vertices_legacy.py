from typing import Any

import numpy as np

from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.modules.dual_contouring._aux import _surface_slicer
from gempy_engine.modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices


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
        tree_depth=dc_data_per_stack.tree_depth,
        base_number=dc_data_per_stack.base_number,
        left_right_codes=dc_data_per_stack.left_right_codes
    )

    vertices_numpy = generate_dual_contouring_vertices(dc_data_per_surface, slice_object, debug)
    return dc_data_per_surface, vertices_numpy
