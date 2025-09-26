from ...core.backend_tensor import BackendTensor
from ...core.data.dual_contouring_data import DualContouringData


def triangulate_dual_contouring(dc_data_per_surface: DualContouringData):
    """
    For each edge that exhibits a sign change, generate a quad
    connecting the minimizing vertices of the four cubes containing the edge.\
    """
    dxdydz = dc_data_per_surface.dxdydz
    centers_xyz = dc_data_per_surface.xyz_on_centers
    indices_arrays = []

    valid_voxels = dc_data_per_surface.valid_voxels
    valid_edges = dc_data_per_surface.valid_edges

    # ! This assumes a vertex per voxel
    dx, dy, dz = dxdydz
    x_1 = centers_xyz[valid_voxels][:, None, :]
    x_2 = centers_xyz[valid_voxels][None, :, :]

    manhattan = x_1 - x_2
    zeros = BackendTensor.tfnp.isclose(manhattan[:, :, :], 0, .00001)
    x_direction_neighbour = BackendTensor.tfnp.isclose(manhattan[:, :, 0], dx, .00001)
    nx_direction_neighbour = BackendTensor.tfnp.isclose(manhattan[:, :, 0], -dx, .00001)
    y_direction_neighbour = BackendTensor.tfnp.isclose(manhattan[:, :, 1], dy, .00001)
    ny_direction_neighbour = BackendTensor.tfnp.isclose(manhattan[:, :, 1], -dy, .00001)
    z_direction_neighbour = BackendTensor.tfnp.isclose(manhattan[:, :, 2], dz, .00001)
    nz_direction_neighbour = BackendTensor.tfnp.isclose(manhattan[:, :, 2], -dz, .00001)

    x_direction = x_direction_neighbour * zeros[:, :, 1] * zeros[:, :, 2]
    nx_direction = nx_direction_neighbour * zeros[:, :, 1] * zeros[:, :, 2]
    y_direction = y_direction_neighbour * zeros[:, :, 0] * zeros[:, :, 2]
    ny_direction = ny_direction_neighbour * zeros[:, :, 0] * zeros[:, :, 2]
    z_direction = z_direction_neighbour * zeros[:, :, 0] * zeros[:, :, 1]
    nz_direction = nz_direction_neighbour * zeros[:, :, 0] * zeros[:, :, 1]

    BackendTensor.tfnp.fill_diagonal(x_direction, True)
    BackendTensor.tfnp.fill_diagonal(nx_direction, True)
    BackendTensor.tfnp.fill_diagonal(y_direction, True)
    BackendTensor.tfnp.fill_diagonal(ny_direction, True)
    BackendTensor.tfnp.fill_diagonal(z_direction, True)
    BackendTensor.tfnp.fill_diagonal(nz_direction, True)

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
    directions = BackendTensor.tfnp.dstack([nynz_direction, nyz_direction, ynz_direction, yz_direction,
                                            nxnz_direction, xnz_direction, nxz_direction, xz_direction,
                                            nxny_direction, nxy_direction, xny_direction, xy_direction])

    # endregion

    valid_edg = valid_edges[valid_voxels][:, :]
    valid_edg = BackendTensor.tfnp.to_numpy(valid_edg)

    direction_each_edge = (directions * valid_edg)

    # Pick only edges with more than 2 voxels nearby
    three_neighbours = (directions * valid_edg).sum(axis=0) == 3
    matrix_to_right_C_order = BackendTensor.tfnp.transpose((direction_each_edge * three_neighbours), (1, 2, 0))
    indices = BackendTensor.tfnp.where(matrix_to_right_C_order)[2].reshape(-1, 3)

    indices_shift = indices
    indices_arrays.append(indices_shift)
    indices_arrays_f = BackendTensor.tfnp.vstack(indices_arrays)

    return indices_arrays_f
