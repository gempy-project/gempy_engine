from typing import Tuple, List, Optional

from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.octree_level import OctreeLevel
from ...core.data.interp_output import InterpOutput
from ...core.utils import gempy_profiler_decorator
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge, \
    triangulate_dual_contouring, generate_dual_contouring_vertices

import numpy as np


def get_intersection_on_edges(octree_level: OctreeLevel, output_corners: InterpOutput,
                              mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    # First find xyz on edges:
    intersection_xyz, valid_edges = find_intersection_on_edge(
        _xyz_corners=octree_level.grid_corners.values,
        scalar_field=output_corners.exported_fields.scalar_field,
        scalar_at_sp=output_corners.scalar_field_at_sp,
        masking=mask
    )
    return intersection_xyz, valid_edges


@gempy_profiler_decorator
def compute_dual_contouring(dc_data: DualContouringData, left_right_codes=None, debug: bool = False) -> List[DualContouringMesh]:

    vertices = generate_dual_contouring_vertices(dc_data, debug)
    
    if left_right_codes is None:
        # * Legacy triangulation
        indices = triangulate_dual_contouring(dc_data)
    else:
        # * Fancy triangulation
        from gempy_engine.modules.dual_contouring.fancy_triangulation import triangulate
        validated_stacked = left_right_codes[dc_data.valid_voxels]
        validated_edges = dc_data.valid_edges[dc_data.valid_voxels]
        indices = triangulate(validated_stacked, validated_edges, dc_data.tree_depth)
        indices = np.vstack(indices)

    return [DualContouringMesh(vertices, indices, dc_data)]

