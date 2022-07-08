from typing import Tuple, List, Optional

from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.octree_level import OctreeLevel
from ...core.data.interp_output import InterpOutput
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge, \
    triangulate_dual_contouring, generate_dual_contouring_vertices

import numpy as np


def get_intersection_on_edges(octree_level: OctreeLevel, output_corners: InterpOutput, mask: Optional[np.ndarray] = None,
                              multiple_scalars: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # First find xyz on edges:
    xyz_corners = octree_level.grid_corners.values
    
    scalar_field_corners = output_corners.exported_fields.scalar_field
    scalar_field_at_all_sp = output_corners.scalar_field_at_sp
    
    intersection_xyz, valid_edges = find_intersection_on_edge(xyz_corners, scalar_field_corners, scalar_field_at_all_sp, mask)
    return intersection_xyz, valid_edges


def compute_dual_contouring(dc_data: DualContouringData, debug: bool = False) -> List[DualContouringMesh]:

    vertices = generate_dual_contouring_vertices(dc_data, debug)
    
    indices = triangulate_dual_contouring(dc_data)

    return [DualContouringMesh(vertices, indices, dc_data)]

