from typing import Tuple, List

from ...core.data.exported_structs import OctreeLevel, DualContouringData, DualContouringMesh, InterpOutput
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge, \
    triangulate_dual_contouring, generate_dual_contouring_vertices

import numpy as np


def get_intersection_on_edges(octree_level: OctreeLevel, output_corners: InterpOutput, 
                              multiple_scalars: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # First find xyz on edges:
    xyz_corners = octree_level.grid_corners.values
    
    scalar_field_at_all_sp = np.zeros(0)
    if multiple_scalars:
        scalar_field_corners = output_corners.final_exported_fields.scalar_field
        for output_corners in octree_level.outputs_corners:
            scalar_field_at_all_sp = np.concatenate((scalar_field_at_all_sp, output_corners.scalar_field_at_sp))
    else:
        scalar_field_corners = output_corners.exported_fields.scalar_field
        scalar_field_at_all_sp = output_corners.scalar_field_at_sp

    intersection_xyz, valid_edges = find_intersection_on_edge(xyz_corners, scalar_field_corners, scalar_field_at_all_sp)
    return intersection_xyz, valid_edges


def compute_dual_contouring(dc_data: DualContouringData, n_surfaces: int, debug: bool = False) -> List[DualContouringMesh]:

    vertices = generate_dual_contouring_vertices(dc_data, debug)
    indices = triangulate_dual_contouring(dc_data, n_surfaces)

    return [DualContouringMesh(vertices, indices, dc_data)]


"""
    last_edge = 0
    s0 = 0
    for i in range(1):
        n_voxels_per_mesh = n_edges//3

        valid_voxels_mesh = valid_voxels[n_voxels_per_mesh * i: n_voxels_per_mesh * (i + 1)]
        valid_edges_mesh = valid_edges[n_voxels_per_mesh*i: n_voxels_per_mesh*(i+1)]
        n_edges_mesh = valid_edges_mesh.shape[0]

        idx = np.nonzero(valid_edges_mesh.sum(axis=1))[0] + last_edge
        #n_edges_mesh = idx.shape[0]

        s1 = valid_edges_mesh.sum()

        grad_mesh = gradients[s0:s1]
        xyz_on_edge_mesh = xyz_on_edge[s0:s1]
        last_edge = idx[-1]

        s0 = s1
        # valid_voxels, vertices = generate_dual_contouring_vertices(gradients, n_edges, valid_edges,
        #                                                            xyz_on_edge, valid_voxels)
        valid_voxels, vertices = generate_dual_contouring_vertices(grad_mesh,
                                                                   n_edges_mesh,
                                                                   valid_edges_mesh,
                                                                   xyz_on_edge_mesh,
                                                                   valid_voxels_mesh)

"""
