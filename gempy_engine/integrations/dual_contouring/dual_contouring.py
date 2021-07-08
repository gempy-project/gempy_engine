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


def compute_dual_contouring(dc_data: DualContouringData, dxdydz, centers_xyz):
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
    if True:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(v_pro)
        indices = hull.simplices
    else:
        pass

    # x_1 = xyz.reshape(-1, 3)
    # x_2 = xyz.reshape(-1, 3)
    #
    # x_1 = x_1[:, None, :]
    # x_2 = x_2[None, :, :]
    #
    # sqd = np.sqrt(np.reduce_sum(((x_1 - x_2) ** 2), -1))

    # Set magic space for triangulation
    x_11 = xyz[:, :12]#.reshape((-1, 1, 3))
    x_21 = v_pro.reshape((1, -1, 3))
   # manhattan = np.abs(x_1 - x_2)

    #n_edges_per_direction = x_1.shape[0]//3
    dx, dy, dz = dxdydz
    #
    # x_direction_0 = np.abs(x_1[:, :4 ].reshape((-1, 1, 3)) - x_2)[:, :, 1] < dy
    # x_direction_1 = np.abs(x_1[:,:4 ].reshape((-1, 1, 3)) - x_2)[:, :, 2] < dz
    # y_direction_0 = np.abs(x_1[:,4:8].reshape((-1, 1, 3)) - x_2)[:, :, 0] < dx
    # y_direction_1 = np.abs(x_1[:,4:8].reshape((-1, 1, 3)) - x_2)[:, :, 2] < dz
    # z_direction_0 = np.abs(x_1[:,8: ].reshape((-1, 1, 3)) - x_2)[:, :, 0] < dx
    # z_direction_1 = np.abs(x_1[:,8: ].reshape((-1, 1, 3)) - x_2)[:, :, 1] < dy
    #
    # # Intersect all the previous directions
    # foo2 = x_direction_0 * x_direction_1 * y_direction_0 * y_direction_1 * z_direction_0 \
    #        *z_direction_1
    #
    #
    #
    #
    # foo = (np.abs((x_1 - x_2))[:, :, 0] < dxdydz[0]) * \
    #       (np.abs((x_1 - x_2))[:, :, 1] < dxdydz[1]) *\
    #       (np.abs((x_1 - x_2))[:, :, 2] < dxdydz[2])
    #
    #
    #
    # bar = foo[foo.sum(axis=1) == 4]
    # indices = np.where(bar)[1]

    x_1 = centers_xyz[bo][:, None, :]
    x_2 = centers_xyz[bo][None, :, :]
    manhattan = x_1 - x_2
    close_x = np.isclose(manhattan[:,:, 0], 0, .00001) * \
              np.isclose(manhattan[:,:, 1], dy, .00001) + \
              np.isclose(manhattan[:, :, 0], 0, .00001) * \
              np.isclose(manhattan[:, :,2], dz, .00001)

    zeros = np.isclose(manhattan[:,:, :], 0, .00001)
    a =  np.isclose(manhattan[:, :, 0], dx, .00001)
    a2 = np.isclose(manhattan[:, :, 0], -dx, .00001)
    b =  np.isclose(manhattan[:, :, 1], dy, .00001)
    b2 = np.isclose(manhattan[:, :, 1], -dy, .00001)
    c =  np.isclose(manhattan[:, :, 2], dz, .00001)
    c2 = np.isclose(manhattan[:, :, 2], -dz, .00001)


    valid_edg = valid_edges[bo][:, :]

    x_direction  = a * zeros[:, :, 1] * zeros[:, :, 2]
    nx_direction = a2 * zeros[:, :, 1] * zeros[:, :, 2]
    y_direction  = b * zeros[:, :, 0] * zeros[:, :, 2]
    ny_direction = b2 * zeros[:, :, 0] * zeros[:, :, 2]
    z_direction  = c * zeros[:, :, 0] * zeros[:, :, 1]
    nz_direction = c2 * zeros[:, :, 0] * zeros[:, :, 1]
    #
    np.fill_diagonal(x_direction ,True)
    np.fill_diagonal(nx_direction,True)
    np.fill_diagonal(y_direction,True)
    np.fill_diagonal(nx_direction,True)
    np.fill_diagonal(z_direction,True)
    np.fill_diagonal(nz_direction,True)

    # X edges
    nynz_direction = ny_direction + nz_direction
    nyz_direction = ny_direction + z_direction
    ynz_direction = y_direction + nz_direction
    yz_direction = y_direction + z_direction

    # Y edges
    nxnz_direction = nx_direction + nz_direction
    xnz_direction = x_direction + nz_direction
    nxz_direction = nx_direction + z_direction
    xz_direction = x_direction + z_direction

    # Z edges
    nxny_direction = nx_direction + ny_direction
    nxy_direction  = nx_direction + y_direction
    xny_direction  = x_direction  + ny_direction
    xy_direction   = x_direction  + y_direction

    # Stack all 12 directions
    directions = np.dstack([nynz_direction, nyz_direction, ynz_direction, yz_direction,
                            nxnz_direction, xnz_direction, nxz_direction, xz_direction,
                            nxny_direction, nxy_direction, xny_direction, xy_direction])


    fuck = directions[:, valid_edg]
    valid_edg2 = np.tile(valid_edg, (1, 22, 1))
    bar = (directions * valid_edg)


    # Pick only edges with more than 2 voxels nearby
    sel = (directions * valid_edg).sum(axis=0) == 3

    indices = np.where(np.transpose((bar*sel), (1,2,0)))[2].reshape(-1, 3)
    seems_something = (directions * valid_edg)[:, :, -1]


    #bar = x_direction + y_direction + z_direction

    # foo = (bar).sum(axis=0)
    # foo_choose = np.where(foo == 4)
    # indices = np.where(bar[foo_choose])[1]
    if False:
        try2 = np.triu(bar)
        np.fill_diagonal(try2, True)
        foo2 = try2.sum(axis=1)
        foo_choose2 = np.where(foo2 == 3)
        indices = np.where(try2[foo_choose2])[1]

        try2 = np.tril(bar)
        np.fill_diagonal(try2, True)
        foo2 = try2.sum(axis=1)
        foo_choose2 = np.where(foo2 == 3)
        foo_choose3 = np.where(foo2 == 4)

        quads = np.where(try2[foo_choose3])[1].reshape(-1, 4)
        tri = transform1(quads)

        indices = np.append(indices,np.where(try2[foo_choose2])[1])
        indices = np.append(indices, tri)

    return [DualContouringMesh(v_pro, indices)]


def find_arg(value: np.ndarray):
    """Find the index for values with four occurrences.
    Examples:
        >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 5, 6], [4, 5, 6], [4, 5, 6]])
        >>> find_arg(a)
        array([4, 5, 6])
    """
    return np.where(np.bincount(value) == 4)[0]


def transform1(a):
    idx = np.flatnonzero(a[:,-1] == 0)
    out0 = np.empty((a.shape[0],2,3),dtype=a.dtype)

    out0[:,0,1:] = a[:,1:-1]
    out0[:,1,1:] = a[:,2:]

    out0[...,0] = a[:,0,None]

    out0.shape = (-1,3)

    mask = np.ones(out0.shape[0],dtype=bool)
    mask[idx*2+1] = 0
    return out0[mask]