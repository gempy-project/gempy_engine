from typing import List

import numpy as np
from ...core.backend_tensor import BackendTensor, BackendTensor as b, AvailableBackends
from ...core.data.exported_structs import InterpOutput, OctreeLevel

from ...core.data.grid import Grid

# TODO: [ ] Check if fortran order speeds up this function
# TODO: Substitute numpy for b.tfnp
# TODO: Remove all stack to be able to compile TF


def compute_octree_root(prev_octree: OctreeLevel, regular_grid_xyz, dxdydz, compute_topology=False) -> OctreeLevel:

    # Old octree
    shift_select_xyz: list = _mark_edge_voxels_dense(prev_octree.id_block)
    prev_octree.marked_edges = shift_select_xyz

    if compute_topology:
        prev_octree.edges_id, prev_octree.count_edges = calculate_topology(shift_select_xyz, prev_octree.id_block)

    # New Octree
    xyz_coords = _create_oct_level_dense(shift_select_xyz, regular_grid_xyz, dxdydz)
    new_octree_level = OctreeLevel(xyz_coords)

    return new_octree_level


def compute_octree_leaf(prev_octree: OctreeLevel, dxdydz, level):
    xyz = prev_octree.xyz_coords
    ids = prev_octree.id_block

    xyz_8 = xyz.reshape((-1, 8, 3))
    uv_8 = ids.reshape((-1, 8))

    shift_select_xyz = _mark_edges_sparse(uv_8)
    prev_octree.marked_edges = shift_select_xyz

    return _create_oct_level_sparse(dxdydz, level, shift_select_xyz, xyz_8)


def calculate_topology(shift_select_xyz: List[np.ndarray], ids: np.ndarray):
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


def _create_oct_level_dense(shift_select_xyz: List[np.ndarray], regular_grid_xyz, dxdydz):

    x_edg = (regular_grid_xyz[:-1, :, :][shift_select_xyz[0]] + regular_grid_xyz[1:, :, :][shift_select_xyz[0]]) / 2
    y_edg = (regular_grid_xyz[:, :-1, :][shift_select_xyz[1]] + regular_grid_xyz[:, 1:, :][shift_select_xyz[1]]) / 2
    z_edg = (regular_grid_xyz[:, :, :-1][shift_select_xyz[2]] + regular_grid_xyz[:, :, 1:][shift_select_xyz[2]]) / 2

    new_xyz_edg = np.vstack((x_edg, y_edg, z_edg))

    return _create_oct_voxels(new_xyz_edg[:, 0], new_xyz_edg[:, 1], new_xyz_edg[:, 2], *dxdydz, level=1)


def _create_oct_level_sparse(dxdydz, level, shift_select_xyz, xyz_8):
    x_edg = (xyz_8[:, :4, :][shift_select_xyz[0]] + xyz_8[:, 4:, :][shift_select_xyz[0]]) / 2
    y_edg = (xyz_8[:, [0, 1, 4, 5], :][shift_select_xyz[1]] + xyz_8[:, [2, 3, 6, 7], :][shift_select_xyz[1]]) / 2
    z_edg = (xyz_8[:, ::2, :][shift_select_xyz[2]] + xyz_8[:, 1::2, :][shift_select_xyz[2]]) / 2
    new_xyz_edg = np.vstack((x_edg, y_edg, z_edg))
    return _create_oct_voxels(new_xyz_edg[:, 0], new_xyz_edg[:, 1], new_xyz_edg[:, 2], *dxdydz, level=level)


def _mark_edges_sparse(uv_8):
    shift_x = uv_8[:, :4] - uv_8[:, 4:]
    shift_y = uv_8[:, [0, 1, 4, 5]] - uv_8[:, [2, 3, 6, 7]]
    shift_z = uv_8[:, ::2] - uv_8[:, 1::2]
    shift_x_select = np.not_equal(shift_x, 0)
    shift_y_select = np.not_equal(shift_y, 0)
    shift_z_select = np.not_equal(shift_z, 0)
    shift_select_xyz = [shift_x_select, shift_y_select, shift_z_select]
    return shift_select_xyz


def _mark_edge_voxels_dense(ids):
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
