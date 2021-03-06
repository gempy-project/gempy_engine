from ...core.data.exported_structs import DualContouringData
import numpy as np


def find_intersection_on_edge(_xyz_8: np.ndarray, scalar_field: np.ndarray,
                              scalar_at_sp: np.ndarray) -> DualContouringData:
    # I have to do the topology analysis anyway because is the last octree
    scalar_8_ = scalar_field
    scalar_8 = scalar_8_.reshape((1, -1, 8))
    scalar_at_sp = scalar_at_sp.reshape((-1, 1, 1))

    xyz_8 = _xyz_8.reshape((-1, 8, 3))
    n_isosurface = scalar_at_sp.shape[0]
    xyz_8 = np.tile(xyz_8, (n_isosurface, 1, 1)) # TODO: Generalize

    # Compute distance of scalar field on the corners
    scalar_dx = scalar_8[:, :, :4] - scalar_8[:, :, 4:]
    scalar_d_y = scalar_8[:, :, [0, 1, 4, 5]] - scalar_8[:, :, [2, 3, 6, 7]]
    scalar_d_z = scalar_8[:,:, ::2] - scalar_8[:, :, 1::2]

    """
    -4.31216,-1.87652,-4.19625,-2.57581
    -1.87652,-13.91019,-2.57581,-30.15220
    """

    # Compute the weights
    weight_x = ((scalar_at_sp - scalar_8[:, :, 4:]) / scalar_dx).reshape(-1, 4, 1)
    weight_y = ((scalar_at_sp -  scalar_8[:, :, [2, 3, 6, 7]]) / scalar_d_y).reshape(-1, 4, 1)
    weight_z = ((scalar_at_sp - scalar_8[:, :, 1::2]) / scalar_d_z).reshape(-1, 4, 1)
    
    # Calculate eucledian distance between the corners
    d_x = xyz_8[:, :4]           - xyz_8[:, 4:]
    d_y = xyz_8[:, [0, 1, 4, 5]] - xyz_8[:, [2, 3, 6, 7]]
    d_z = xyz_8[:, ::2]          - xyz_8[:, 1::2]

    # Compute the weighted distance
    intersect_dx = d_x[:, :, :] * weight_x[:,:, :]
    intersect_dy = d_y[:, :, :] * weight_y[:,:, :]
    intersect_dz = d_z[:, :, :] * weight_z[:,:, :]

    # Mask invalid edges
    # TODO: This still only works for the first layer of a sequence
    valid_edge_x = np.logical_and(weight_x > 0, weight_x < 1)
    valid_edge_y = np.logical_and(weight_y > 0, weight_y < 1)
    valid_edge_z = np.logical_and(weight_z > 0, weight_z < 1)

    #Note(miguel) From this point on the arrays become sparse
    xyz_8_edges = np.hstack([xyz_8[:, 4:], xyz_8[:, [2, 3, 6, 7]], xyz_8[:, 1::2]])
    intersect_segment = np.hstack([intersect_dx, intersect_dy, intersect_dz])
    valid_edges = np.hstack([valid_edge_x, valid_edge_y, valid_edge_z])[:, :, 0]

    intersection_xyz = xyz_8_edges[valid_edges] + intersect_segment[valid_edges]

    return DualContouringData(intersection_xyz, valid_edges) # TODO: Add attributes?


def triangulate_dual_contouring(centers_xyz, dxdydz, valid_edges, valid_voxels):
    dx, dy, dz = dxdydz
    x_1 = centers_xyz[valid_voxels][:, None, :]
    x_2 = centers_xyz[valid_voxels][None, :, :]
    manhattan = x_1 - x_2
    zeros = np.isclose(manhattan[:, :, :], 0, .00001)
    x_direction_neighbour = np.isclose(manhattan[:, :, 0], dx, .00001)
    nx_direction_neighbour = np.isclose(manhattan[:, :, 0], -dx, .00001)
    y_direction_neighbour = np.isclose(manhattan[:, :, 1], dy, .00001)
    ny_direction_neighbour = np.isclose(manhattan[:, :, 1], -dy, .00001)
    z_direction_neighbour = np.isclose(manhattan[:, :, 2], dz, .00001)
    nz_direction_neighbour = np.isclose(manhattan[:, :, 2], -dz, .00001)
    valid_edg = valid_edges[valid_voxels][:, :]
    x_direction = x_direction_neighbour * zeros[:, :, 1] * zeros[:, :, 2]
    nx_direction = nx_direction_neighbour * zeros[:, :, 1] * zeros[:, :, 2]
    y_direction = y_direction_neighbour * zeros[:, :, 0] * zeros[:, :, 2]
    ny_direction = ny_direction_neighbour * zeros[:, :, 0] * zeros[:, :, 2]
    z_direction = z_direction_neighbour * zeros[:, :, 0] * zeros[:, :, 1]
    nz_direction = nz_direction_neighbour * zeros[:, :, 0] * zeros[:, :, 1]
    #
    np.fill_diagonal(x_direction, True)
    np.fill_diagonal(nx_direction, True)
    np.fill_diagonal(y_direction, True)
    np.fill_diagonal(nx_direction, True)
    np.fill_diagonal(z_direction, True)
    np.fill_diagonal(nz_direction, True)
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
    nxy_direction = nx_direction + y_direction
    xny_direction = x_direction + ny_direction
    xy_direction = x_direction + y_direction
    # Stack all 12 directions
    directions = np.dstack([nynz_direction, nyz_direction, ynz_direction, yz_direction,
                            nxnz_direction, xnz_direction, nxz_direction, xz_direction,
                            nxny_direction, nxy_direction, xny_direction, xy_direction])
    direction_each_edge = (directions * valid_edg)
    # Pick only edges with more than 2 voxels nearby
    three_neighbours = (directions * valid_edg).sum(axis=0) == 3
    matrix_to_right_C_order = np.transpose((direction_each_edge * three_neighbours), (1, 2, 0))
    indices = np.where(matrix_to_right_C_order)[2].reshape(-1, 3)
    return indices


def generate_dual_contouring_vertices(gradients, n_edges, valid_edges, xyz_on_edge, valid_voxels):
    # Coordinates for all posible edges (12) and 3 dummy edges_normals in the center
    edges_xyz = np.zeros((n_edges, 15, 3))
    edges_normals = np.zeros((n_edges, 15, 3))
    edges_xyz[:, :12][valid_edges] = xyz_on_edge
    edges_normals[:, :12][valid_edges] = gradients
    BIAS_STRENGTH = 10
    bias_xyz = np.copy(edges_xyz[:, :12])
    # np zero values to nans
    bias_xyz[np.isclose(bias_xyz, 0)] = np.nan
    # Mean ignoring nans
    mass_points = np.nanmean(bias_xyz, axis=1)
    edges_xyz[:, 12] = mass_points
    edges_xyz[:, 13] = mass_points
    edges_xyz[:, 14] = mass_points
    edges_normals[:, 12] = np.array([BIAS_STRENGTH, 0, 0])
    edges_normals[:, 13] = np.array([0, BIAS_STRENGTH, 0])
    edges_normals[:, 14] = np.array([0, 0, BIAS_STRENGTH])

    # Remove unused voxels
    edges_xyz = edges_xyz[valid_voxels]
    edges_normals = edges_normals[valid_voxels]
    # Compute LSTSQS in all voxels at the same time
    A = edges_normals
    b = (A * edges_xyz).sum(axis=2)
    term1 = np.einsum("ijk, ilj->ikl", A, np.transpose(A, (0, 2, 1)))
    term2 = np.linalg.inv(term1)
    term3 = np.einsum("ijk,ik->ij", np.transpose(A, (0, 2, 1)), b)
    vertices = np.einsum("ijk, ij->ik", term2, term3)
    return valid_voxels, vertices


# NOTE(miguel, July 2021): This class is only used for sanity check
class QEF:
    """Represents and solves the quadratic error function"""
    def __init__(self, A, b, fixed_values):
        self.A = A
        self.b = b
        self.fixed_values = fixed_values

    def evaluate(self, x):
        """Evaluates the function at a given point.
        This is what the solve method is trying to minimize.
        NB: Doesn't work with fixed axes."""
        x = np.array(x)
        return np.linalg.norm(np.matmul(self.A, x) - self.b)

    def eval_with_pos(self, x):
        """Evaluates the QEF at a position, returning the same format solve does."""
        return self.evaluate(x), x


    @staticmethod
    def make_3d(positions, normals):
        """Returns a QEF that measures the the error from a bunch of normals, each emanating
         from given positions"""
        A = np.array(normals)
        b = [v[0] * n[0] + v[1] * n[1] + v[2] * n[2] for v, n in zip(positions, normals)]
        fixed_values = [None] * A.shape[1]
        return QEF(A, b, fixed_values)

    def solve(self):
        """Finds the point that minimizes the error of this QEF,
        and returns a tuple of the error squared and the point itself"""
        result, residual, rank, s = np.linalg.lstsq(self.A, self.b)
        if len(residual) == 0:
            residual = self.evaluate(result)
        else:
            residual = residual[0]
        # Result only contains the solution for the unfixed axis,
        # we need to add back all the ones we previously fixed.
        position = []
        i = 0
        for value in self.fixed_values:
            if value is None:
                position.append(result[i])
                i += 1
            else:
                position.append(value)
        return residual, position