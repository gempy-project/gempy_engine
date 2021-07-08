from ...core.data.exported_structs import OctreeLevel, DualContouringData, DualContouringMesh
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge, \
    triangulate_dual_contouring, generate_dual_contouring_vertices


def get_intersection_on_edges(octree_level: OctreeLevel) -> DualContouringData:
    # First find xyz on edges:
    xyz_corners = octree_level.grid_corners.values
    scalar_field_corners = octree_level.output_corners.exported_fields.scalar_field
    scalar_field_at_sp = octree_level.output_corners.scalar_field_at_sp

    dc_data = find_intersection_on_edge(xyz_corners, scalar_field_corners, scalar_field_at_sp)
    dc_data.grid_centers = octree_level.grid_centers

    return dc_data


def compute_dual_contouring(dc_data: DualContouringData):
    # QEF:
    valid_edges = dc_data.valid_edges
    valid_voxels = valid_edges.sum(axis=1, dtype=bool)
    xyz_on_edge = dc_data.xyz_on_edge
    gradients = dc_data.gradients
    n_edges = valid_edges.shape[0]

    valid_voxels, vertices = generate_dual_contouring_vertices(gradients, n_edges, valid_edges,
                                                               xyz_on_edge, valid_voxels)

    # Triangulate
    # ===========

    # For each edge that exhibits a sign change, generate a quad
    # connecting the minimizing vertices of the four cubes containing the edge.

    dxdydz = dc_data.grid_centers.dxdydz
    centers_xyz = dc_data.grid_centers.values

    indices = triangulate_dual_contouring(centers_xyz, dxdydz, valid_edges, valid_voxels)

    return [DualContouringMesh(vertices, indices)]
