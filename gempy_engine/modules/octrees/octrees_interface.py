import numpy as np
from ...core.backend_tensor import BackendTensor, BackendTensor as b, AvailableBackends

# TODO: [ ] Check if fortran order speeds up this function
# TODO: Substitute numpy for b.tfnp
from ...core.data.grid import Grid


def create_oct_level_dense(values_block: np.ndarray, grid: Grid):
    unique_ids = values_block[:, :grid.len_grids[0]].reshape(grid.regular_grid_shape)  # TODO: [ ] For faults it has to be lith_block + self.max_lith * fault_block[2]
    ids = np.rint(unique_ids) # shape (nx, ny, nz)
    regular_grid_xyz = grid.regular_grid # shape (nx, ny, nz, 3)

    shift_x = ids[1:, :, :] - ids[:-1, :, :]
    shift_y = ids[:, 1:, :] - ids[:, :-1, :]
    shift_z = ids[:, :, 1:] - ids[:, :, :-1]

    shift_x_select = np.not_equal(shift_x, 0)
    shift_y_select = np.not_equal(shift_y, 0)
    shift_z_select = np.not_equal(shift_z, 0)

    x_edg = (regular_grid_xyz[:-1, :, :][shift_x_select] + regular_grid_xyz[1:, :, :][shift_x_select]) / 2
    y_edg = (regular_grid_xyz[:, :-1, :][shift_y_select] + regular_grid_xyz[:, 1:, :][shift_y_select]) / 2
    z_edg = (regular_grid_xyz[:, :, :-1][shift_z_select] + regular_grid_xyz[:, :, 1:][shift_z_select]) / 2
    #new_shape = np.concatenate([regular_grid.shape, np.array(3)], dtype='int32')

    # shift = grid.shape[0]



    # uv_3d = T.cast(
    #     T.round(unique_val[0, :T.prod(self.regular_grid_res)]
    #         .reshape(self.regular_grid_res, ndim=3)),
    #     'int32')

    # new_shape = T.concatenate([self.regular_grid_res, T.stack([3])])
    # xyz = grid[:T.prod(self.regular_grid_res)].reshape(new_shape, ndim=4)
    #
    # shift_x = uv_3d[1:, :, :] - uv_3d[:-1, :, :]
    # shift_x_select = T.neq(shift_x, 0)
    # x_edg = (xyz[:-1, :, :][shift_x_select] + xyz[1:, :, :][shift_x_select]) / 2
    #
    # shift_y = uv_3d[:, 1:, :] - uv_3d[:, :-1, :]
    # shift_y_select = T.neq(shift_y, 0)
    # y_edg = (xyz[:, :-1, :][shift_y_select] + xyz[:, 1:, :][shift_y_select]) / 2
    #
    # shift_z = uv_3d[:, :, 1:] - uv_3d[:, :, :-1]
    # shift_z_select = T.neq(shift_z, 0)
    # z_edg = (xyz[:, :, :-1][shift_z_select] + xyz[:, :, 1:][shift_z_select]) / 2



    new_xyz_edg = np.vstack((x_edg, y_edg, z_edg))

    return _create_oct_voxels(new_xyz_edg[:, 0], new_xyz_edg[:, 1], new_xyz_edg[:, 2], *grid.dxdydz, level=1)


def _create_oct_voxels(x_edg, y_edg, z_edg, dx, dy, dz, level=1):

    def stack_left_right(a_edg, d_a):
        return np.stack((a_edg - d_a / level / 4, a_edg + d_a / level / 4), axis = 1)

    x_ = np.repeat(stack_left_right(x_edg, dx), 4, axis= 1)
    x = x_.ravel()
    y_ = np.tile(np.repeat(stack_left_right(y_edg, dy), 2, axis = 1), (1, 2))
    y = y_.ravel()
    z_ = np.tile(stack_left_right(z_edg, dz), (1, 4))
    z = z_.ravel()

    new_xyz = np.stack((x, y, z)).T
    return new_xyz
    #
    # x_ = T.repeat(T.stack((xyz[:, 0] - self.dxdydz[0] / level / 4,
    #                        xyz[:, 0] + self.dxdydz[0] / level / 4), axis=1), 4,
    #               axis=1)


    #
    # y_ = T.tile(T.repeat(T.stack((xyz[:, 1] - self.dxdydz[1] / level / 4,
    #                               xyz[:, 1] + self.dxdydz[1] / level / 4),
    #                              axis=1),
    #                      2, axis=1), (1, 2))
    #
    # z_ = T.tile(T.stack((xyz[:, 2] - self.dxdydz[2] / level / 4,
    #                      xyz[:, 2] + self.dxdydz[2] / level / 4), axis=1),
    #             (1, 4))

    # return T.stack((x_.ravel(), y_.ravel(), z_.ravel())).T
