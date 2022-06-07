from ...core.data.exported_structs import OctreeLevel, DualContouringData, DualContouringMesh
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge, \
    triangulate_dual_contouring, generate_dual_contouring_vertices

import numpy as np


def get_intersection_on_edges(octree_level: OctreeLevel) -> DualContouringData:
    # First find xyz on edges:
    xyz_corners = octree_level.grid_corners.values
    scalar_field_corners = octree_level.output_corners.exported_fields.scalar_field
    scalar_field_at_sp = octree_level.output_corners.scalar_field_at_sp

    dc_data = find_intersection_on_edge(xyz_corners, scalar_field_corners, scalar_field_at_sp)
    dc_data.grid_centers = octree_level.grid_centers

    return dc_data


def compute_dual_contouring(dc_data: DualContouringData, n_surfaces: int):
    # QEF:
    valid_edges = dc_data.valid_edges
    valid_voxels = valid_edges.sum(axis=1, dtype=bool)
    xyz_on_edge = dc_data.xyz_on_edge
    gradients = dc_data.gradients
    n_edges = valid_edges.shape[0]

    valid_voxels, vertices = generate_dual_contouring_vertices(gradients, n_edges, valid_edges, xyz_on_edge, valid_voxels)

    # Triangulate
    # ===========

    # * For each edge that exhibits a sign change, generate a quad
    # * connecting the minimizing vertices of the four cubes containing the edge.

    def triangulate_params(dc_data, n_surfaces):
        dxdydz = dc_data.grid_centers.dxdydz
        centers_xyz = dc_data.grid_centers.values  # ? Can I extract here too. (UPDATE: Not sure what I meant)
        centers_xyz = np.tile(centers_xyz, (n_surfaces, 1))
        return centers_xyz, dxdydz

    centers_xyz, dxdydz = triangulate_params(dc_data, n_surfaces)

    indices = triangulate_dual_contouring(centers_xyz, dxdydz, valid_edges, valid_voxels)

    return [DualContouringMesh(vertices, indices)]


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
