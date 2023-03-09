from typing import Tuple, List, Optional

from ...core.data.dual_contouring_mesh import DualContouringMesh
from ...core.data.dual_contouring_data import DualContouringData
from ...core.data.octree_level import OctreeLevel
from ...core.data.interp_output import InterpOutput
from ...core.utils import gempy_profiler_decorator
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


@gempy_profiler_decorator
def compute_dual_contouring(dc_data_per_stack: DualContouringData, left_right_codes=None, debug: bool = False) -> List[DualContouringMesh]:
    valid_voxels_per_surface = dc_data_per_stack.valid_voxels.reshape((dc_data_per_stack.n_surfaces_to_export, -1))
    valid_edges_per_surface = dc_data_per_stack.valid_edges.reshape((dc_data_per_stack.n_surfaces_to_export, -1, 12))

    vertices = generate_dual_contouring_vertices(dc_data_per_stack, debug)

    stack_meshes: List[DualContouringMesh] = []
    last_index = 0
    for i in range(dc_data_per_stack.n_surfaces_to_export):
        dc_data_per_surface = DualContouringData(
            # @ off
            xyz_on_edge              = dc_data_per_stack.xyz_on_edge,
            valid_edges              = valid_edges_per_surface[i],
            xyz_on_centers           = dc_data_per_stack.xyz_on_centers,
            dxdydz                   = dc_data_per_stack.dxdydz,
            exported_fields_on_edges = dc_data_per_stack.exported_fields_on_edges,
            n_surfaces_to_export     = dc_data_per_stack.n_surfaces_to_export,
            tree_depth               = dc_data_per_stack.tree_depth
            # @ on
        )

        if left_right_codes is None:
            # * Legacy triangulation
            indices = triangulate_dual_contouring(dc_data_per_surface, last_index)
            last_index = indices.max() + 1
        else:
            # * Fancy triangulation 👗
            from gempy_engine.modules.dual_contouring.fancy_triangulation import triangulate

            # validated_stacked = left_right_codes[valid_voxels_per_surface[i]]
            # validated_edges = valid_edges_per_surface[i][valid_voxels_per_surface[i]]
            # indices = triangulate(validated_stacked, validated_edges, dc_data_per_stack.tree_depth)

            valid_voxels = dc_data_per_surface.valid_voxels
            validated_stacked = left_right_codes[valid_voxels]
            edges = dc_data_per_surface.valid_edges
            validated_edges   = edges[valid_voxels]
            indices = triangulate(validated_stacked, validated_edges, dc_data_per_surface.tree_depth)

            indices = np.vstack(indices)
            indices += last_index
            last_index = indices.max() + 1

        stack_meshes.append(DualContouringMesh(vertices, indices, dc_data_per_stack))
    return stack_meshes
