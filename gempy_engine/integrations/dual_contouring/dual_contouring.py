import numpy as np

from ...core.data.exported_structs import OctreeLevel, DualContouringData, DualContouringMesh
from ...modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge


def get_intersection_on_edges(octree_level: OctreeLevel) -> DualContouringData:
    # First find xyz on edges:
    xyz_corners = octree_level.grid_corners.values
    scalar_field_corners = octree_level.output_corners.exported_fields.scalar_field
    scalar_field_at_sp = octree_level.output_corners.scalar_field_at_sp

    dc_data = find_intersection_on_edge(xyz_corners, scalar_field_corners, scalar_field_at_sp)

    return dc_data


def compute_dual_contouring(dc_data: DualContouringData):
    # QEF:
    valid_edges = dc_data.valid_edges
    xyz_on_edge = dc_data.xyz_on_edge
    gradients = dc_data.gradients

    n_edges = valid_edges.shape[0]

    # Coordinates for all posible edges (12) and 3 dummy normals in the center
    xyz = np.zeros((n_edges, 15, 3))
    normals = np.zeros((n_edges, 15, 3))

    xyz[:, :12][valid_edges] = xyz_on_edge
    normals[:, :12][valid_edges] = gradients

    BIAS_STRENGTH = 0.1

    xyz_aux = np.copy(xyz[:, :12])

    # Numpy zero values to nans
    xyz_aux[np.isclose(xyz_aux, 0)] = np.nan
    # Mean ignoring nans
    mass_points = np.nanmean(xyz_aux, axis=1)

    xyz[:, 12] = mass_points
    xyz[:, 13] = mass_points
    xyz[:, 14] = mass_points

    normals[:, 12] = np.array([BIAS_STRENGTH, 0, 0])
    normals[:, 13] = np.array([0, BIAS_STRENGTH, 0])
    normals[:, 14] = np.array([0, 0, BIAS_STRENGTH])

    # Remove unused voxels
    bo = valid_edges.sum(axis=1, dtype=bool)
    xyz = xyz[bo]
    normals = normals[bo]

    # Compute LSTSQS in all voxels at the same time
    A1 = normals
    b1 = xyz
    bb1 = (A1 * b1).sum(axis=2)
    s1 = np.einsum("ijk, ilj->ikl", A1, np.transpose(A1, (0, 2, 1)))
    s2 = np.linalg.inv(s1)
    s3 = np.einsum("ijk,ik->ij", np.transpose(A1, (0, 2, 1)), bb1)
    v_pro = np.einsum("ijk, ij->ik", s2, s3)

    # Convex Hull
    # ===========

    # For each edge that exhibits a sign change, generate a quad
    # connecting the minimizing vertices of the four cubes containing the edge.
    from scipy.spatial import ConvexHull

    hull = ConvexHull(v_pro)
    indices = hull.simplices

    return [DualContouringMesh(v_pro, indices)]

