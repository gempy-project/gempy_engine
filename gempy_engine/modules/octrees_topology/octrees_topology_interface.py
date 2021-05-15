from typing import List

import numpy as np
from ...core.backend_tensor import BackendTensor, BackendTensor as b, AvailableBackends

from ...core.data.grid import Grid

# TODO: [ ] Check if fortran order speeds up this function
# TODO: Substitute numpy for b.tfnp
# TODO: Remove all stack to be able to compile TF



def compute_octtree_level_0(values_block: np.ndarray, grid: Grid, compute_topology=False):
    ids = _extract_regular_grid_ids(values_block, grid.regular_grid_shape)
    shift_select_xyz: list = _mark_edge_voxels(ids)

    xyz_level1 = _create_oct_level_dense(shift_select_xyz, grid)
    if compute_topology:
        edges_id, count_edges = compute_topology(shift_select_xyz, ids)

    # ---


    # ---

    return  # TODO: Create a class for each level with the new xyz and selected voxels, edge_id, count_edges


def create_oct_level_sparse(ids, xyz: np.ndarray, dxdydz, level):
    xyz_8 = xyz.reshape((-1, 8, 3))
    # uv_8 = T.round(unique_val[0, :-2 * self.len_points].reshape((-1, 8)))

    uv_8 = ids[0, :].reshape((-1, 8))

    shift_x = uv_8[:, :4] - uv_8[:, 4:]
    shift_x_select = np.not_equal(shift_x, 0)
    x_edg = (xyz_8[:, :4, :][shift_x_select] + xyz_8[:, 4:, :][shift_x_select]) / 2

    shift_y = uv_8[:, [0, 1, 4, 5]] - uv_8[:, [2, 3, 6, 7]]
    shift_y_select = np.not_equal(shift_y, 0)
    y_edg = (xyz_8[:, [0, 1, 4, 5], :][shift_y_select] + xyz_8[:, [2, 3, 6, 7], :][shift_y_select]) / 2

    shift_z = uv_8[:, ::2] - uv_8[:, 1::2]
    shift_z_select = np.not_equal(shift_z, 0)
    z_edg = (xyz_8[:, ::2, :][shift_z_select] + xyz_8[:, 1::2, :][shift_z_select]) / 2

    new_xyz_edg = np.vstack((x_edg, y_edg, z_edg))
    return _create_oct_voxels(new_xyz_edg[:, 0], new_xyz_edg[:, 1], new_xyz_edg[:, 2], *dxdydz, level=level)


def _extract_regular_grid_ids(values_block, regular_grid_shape):
    # TODO: [ ] For faults it has to be lith_block + self.max_lith * fault_block[2]
    unique_ids = values_block[:, :regular_grid_shape.sum(axis=0)].reshape(regular_grid_shape)
    ids = np.rint(unique_ids)  # shape (nx, ny, nz)
    return ids


def _create_oct_level_dense(shift_select_xyz: List[np.ndarray], grid: Grid):
    regular_grid_xyz = grid.regular_grid

    x_edg = (regular_grid_xyz[:-1, :, :][shift_select_xyz[0]] + regular_grid_xyz[1:, :, :][shift_select_xyz[0]]) / 2
    y_edg = (regular_grid_xyz[:, :-1, :][shift_select_xyz[1]] + regular_grid_xyz[:, 1:, :][shift_select_xyz[1]]) / 2
    z_edg = (regular_grid_xyz[:, :, :-1][shift_select_xyz[2]] + regular_grid_xyz[:, :, 1:][shift_select_xyz[2]]) / 2

    new_xyz_edg = np.vstack((x_edg, y_edg, z_edg))

    return _create_oct_voxels(new_xyz_edg[:, 0], new_xyz_edg[:, 1], new_xyz_edg[:, 2], *grid.dxdydz, level=1)


def compute_topology(shift_select_xyz: List[np.ndarray], ids: np.ndarray):
    """This is for the typology of level 0. Probably for the rest of octtrees
    levels it will be a bit different
    """

    shift_x_select, shift_y_select, shift_z_select = shift_select_xyz

    x_l = ids[1:, :, :][shift_x_select]
    x_r = ids[:-1, :, :][shift_x_select]

    y_l = ids[:, 1:, :][shift_y_select]
    y_r = ids[:, :-1, :][shift_y_select]

    z_l = ids[:, :, 1:][shift_z_select]
    z_r = ids[:, :, :-1][shift_z_select]

    contiguous_voxels = np.vstack([np.hstack((x_l, y_l, z_l)), np.hstack((x_r, y_r, z_r))])
    edges_id, count_edges = np.unique(contiguous_voxels, return_counts=True, axis=1)

    return edges_id, count_edges


def _mark_edge_voxels(ids):
    shift_x = ids[1:, :, :] - ids[:-1, :, :]
    shift_y = ids[:, 1:, :] - ids[:, :-1, :]
    shift_z = ids[:, :, 1:] - ids[:, :, :-1]

    shift_x_select = np.not_equal(shift_x, 0)
    shift_y_select = np.not_equal(shift_y, 0)
    shift_z_select = np.not_equal(shift_z, 0)

    shift_select_xyz = [shift_x_select, shift_y_select, shift_z_select]

    return shift_select_xyz


def _create_oct_voxels(x_edg, y_edg, z_edg, dx, dy, dz, level=1):
    def stack_left_right(a_edg, d_a):
        return np.stack((a_edg - d_a / level / 4, a_edg + d_a / level / 4), axis=1)

    x_ = np.repeat(stack_left_right(x_edg, dx), 4, axis=1)
    x = x_.ravel()
    y_ = np.tile(np.repeat(stack_left_right(y_edg, dy), 2, axis=1), (1, 2))
    y = y_.ravel()
    z_ = np.tile(stack_left_right(z_edg, dz), (1, 4))
    z = z_.ravel()

    new_xyz = np.stack((x, y, z)).T
    return new_xyz
