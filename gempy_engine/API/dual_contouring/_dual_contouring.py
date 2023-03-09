from typing import Tuple, List, Optional

from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.octree_level import OctreeLevel
from ...core.data.interp_output import InterpOutput
from ...core.utils import gempy_profiler_decorator
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge, \
    triangulate_dual_contouring, generate_dual_contouring_vertices

import numpy as np

from ...modules.dual_contouring.fancy_triangulation import triangulate



@gempy_profiler_decorator
def compute_dual_contouring(dc_data_per_stack: DualContouringData, left_right_codes=None, debug: bool = False) -> List[DualContouringMesh]:
    valid_edges_per_surface = dc_data_per_stack.valid_edges.reshape((dc_data_per_stack.n_surfaces_to_export, -1, 12))

    vertices: np.ndarray = generate_dual_contouring_vertices(dc_data_per_stack, debug)  # * In multilayers, the vertex array contains all the vertex of the stack. This is a waste of memory but for example in unity we could use submeshes.

    stack_meshes: List[DualContouringMesh] = []
    last_index = 0
    for i in range(dc_data_per_stack.n_surfaces_to_export):
        dc_data_per_surface = DualContouringData(
            # @off
            xyz_on_edge              = dc_data_per_stack.xyz_on_edge,
            valid_edges              = valid_edges_per_surface[i],
            xyz_on_centers           = dc_data_per_stack.xyz_on_centers,
            dxdydz                   = dc_data_per_stack.dxdydz,
            exported_fields_on_edges = dc_data_per_stack.exported_fields_on_edges,
            n_surfaces_to_export     = dc_data_per_stack.n_surfaces_to_export,
            tree_depth               = dc_data_per_stack.tree_depth
            # @on
        )

        if left_right_codes is None:
            # * Legacy triangulation
            indices = triangulate_dual_contouring(dc_data_per_surface, last_index)
            last_index = indices.max() + 1
        else:
            # * Fancy triangulation 👗
            valid_voxels = dc_data_per_surface.valid_voxels
            validated_stacked = left_right_codes[valid_voxels]
            edges = dc_data_per_surface.valid_edges
            validated_edges = edges[valid_voxels]
            indices = triangulate(validated_stacked, validated_edges, dc_data_per_surface.tree_depth)

            indices = np.vstack(indices)
            indices += last_index
            last_index = indices.max() + 1

        stack_meshes.append(DualContouringMesh(vertices, indices, dc_data_per_stack))
    return stack_meshes
