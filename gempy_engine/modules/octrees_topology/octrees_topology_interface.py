from typing import List

import numpy as np

from ...core.data.exported_structs import OctreeLevel
from ...core.data.grid import Grid, RegularGrid
from . import _octree_root

# TODO: [ ] Check if fortran order speeds up this function
# TODO: Substitute numpy for b.tfnp
# TODO: Remove all stack to be able to compile TF


def compute_octree_root(prev_octree: OctreeLevel, compute_topology=False) -> OctreeLevel:
    return _octree_root.compute_octree_root(prev_octree, compute_topology)


def set_bool_array_sparse(shift_select_xyz, regular_grid):
    shift_x_select, shift_y_select, shift_z_select = shift_select_xyz
    oct_chunk = (shift_x_select + shift_y_select + shift_z_select).sum(axis=1).astype(bool)

    base_array = np.zeros(regular_grid.resolution  * 2
                          , dtype=bool)
    active_cells = regular_grid.active_cells # Are the number of active cells equals to shift_select?
    active_cells = np.repeat(active_cells, 2, axis=0)
    active_cells = np.repeat(active_cells, 2, axis=1)
    active_cells = np.repeat(active_cells, 2, axis=2)


    w_ = np.where(active_cells)
    # slicex = w_[0][oct_chunk]#[shift_x_select.sum(axis=1).astype(bool)]
    # slicey = w_[1][oct_chunk]#[shift_y_select.sum(axis=1).astype(bool)]
    # slicez = w_[2][oct_chunk]#[shift_z_select.sum(axis=1).astype(bool)]

    slicex0 = w_[0].reshape(-1, 8)[:, :4][shift_x_select] # TODO: This has to be 38?
    slicex1 = w_[0].reshape(-1, 8)[:, :4][shift_y_select]
    slicex2 = w_[0].reshape(-1, 8)[:, :4][shift_z_select]
    slicey0 = w_[1].reshape(-1, 8)[:, [0, 1, 4, 5]][shift_x_select]
    slicey1 = w_[1].reshape(-1, 8)[:, [0, 1, 4, 5]][shift_y_select]
    slicey2 = w_[1].reshape(-1, 8)[:, [0, 1, 4, 5]][shift_z_select]
    slicez0 = w_[2].reshape(-1, 8)[:, ::2][shift_x_select]  # TODO: This has to be 38?
    slicez1 = w_[2].reshape(-1, 8)[:, ::2][shift_y_select]
    slicez2 = w_[2].reshape(-1, 8)[:, ::2][shift_z_select]


    slicex = np.concatenate((slicex0, slicex1, slicex2))
    slicey = np.concatenate((slicey0, slicey1, slicey2))
    slicez = np.concatenate((slicez0, slicez1, slicez2))
    #slicey = w_[1][::2][shift_y_select.ravel()]
    #slicez = w_[2][::2][shift_z_select.ravel()]

    # TODO: I need 304
    active_cells[slicex, slicey, slicez] = True # TODO: we need 152

    slicex0 = w_[0].reshape(-1, 8)[:, 4:][shift_x_select]  # TODO: This has to be 38?
    slicex1 = w_[0].reshape(-1, 8)[:, 4:][shift_y_select]
    slicex2 = w_[0].reshape(-1, 8)[:, 4:][shift_z_select]
    slicey0 = w_[1].reshape(-1, 8)[:, [2, 3, 6, 7]][shift_x_select]
    slicey1 = w_[1].reshape(-1, 8)[:, [2, 3, 6, 7]][shift_y_select]
    slicey2 = w_[1].reshape(-1, 8)[:, [2, 3, 6, 7]][shift_z_select]
    slicez0 = w_[2].reshape(-1, 8)[:, 1::2][shift_x_select]  # TODO: This has to be 38?
    slicez1 = w_[2].reshape(-1, 8)[:, 1::2][shift_y_select]
    slicez2 = w_[2].reshape(-1, 8)[:, 1::2][shift_z_select]

    slicex = np.concatenate((slicex0, slicex1, slicex2))
    slicey = np.concatenate((slicey0, slicey1, slicey2))
    slicez = np.concatenate((slicez0, slicez1, slicez2))
    #slicey = w_[1][::2][shift_y_select.ravel()]
    #slicez = w_[2][::2][shift_z_select.ravel()]

    # TODO: I need 304
    active_cells[slicex, slicey, slicez] = True
    return base_array

def set_bool_array_sparse_wrong(shift_select_xyz, regular_grid):
    shift_x_select, shift_y_select, shift_z_select = shift_select_xyz
    oct_chunk = (shift_x_select + shift_y_select + shift_z_select).sum(axis=1).astype(bool)

    base_array = np.zeros(regular_grid.resolution # * 2
                          , dtype=bool)
    active_cells = regular_grid.active_cells # Are the number of active cells equals to shift_select?
    active_cells = np.repeat(active_cells, 2, axis=0)
    active_cells = np.repeat(active_cells, 2, axis=1)
    active_cells = np.repeat(active_cells, 2, axis=2)


    w_ = np.where(active_cells)
    # slicex = w_[0][oct_chunk]#[shift_x_select.sum(axis=1).astype(bool)]
    # slicey = w_[1][oct_chunk]#[shift_y_select.sum(axis=1).astype(bool)]
    # slicez = w_[2][oct_chunk]#[shift_z_select.sum(axis=1).astype(bool)]

    slicex0 = w_[0][::2][shift_x_select.ravel()] # TODO: This has to be 38?
    slicex1 = w_[0][::2][shift_y_select.ravel()]
    slicex2 = w_[0][::2][shift_z_select.ravel()]
    slicey0 = w_[1][::2][shift_x_select.ravel()]  # TODO: This has to be 38?
    slicey1 = w_[1][::2][shift_y_select.ravel()]
    slicey2 = w_[1][::2][shift_z_select.ravel()]
    slicez0 = w_[2][::2][shift_x_select.ravel()]  # TODO: This has to be 38?
    slicez1 = w_[2][::2][shift_y_select.ravel()]
    slicez2 = w_[2][::2][shift_z_select.ravel()]


    slicex = np.concatenate((slicex0, slicex1, slicex2))
    slicey = np.concatenate((slicey0, slicey1, slicey2))
    slicez = np.concatenate((slicez0, slicez1, slicez2))
    #slicey = w_[1][::2][shift_y_select.ravel()]
    #slicez = w_[2][::2][shift_z_select.ravel()]

    # TODO: I need 304
    base_array[slicex, slicey, slicez] = True


    slicex0 = w_[0][1::2][shift_x_select.ravel()] # TODO: This has to be 38?
    slicex1 = w_[0][1::2][shift_y_select.ravel()]
    slicex2 = w_[0][1::2][shift_z_select.ravel()]
    slicey0 = w_[1][1::2][shift_x_select.ravel()]  # TODO: This has to be 38?
    slicey1 = w_[1][1::2][shift_y_select.ravel()]
    slicey2 = w_[1][1::2][shift_z_select.ravel()]
    slicez0 = w_[2][1::2][shift_x_select.ravel()]  # TODO: This has to be 38?
    slicez1 = w_[2][1::2][shift_y_select.ravel()]
    slicez2 = w_[2][1::2][shift_z_select.ravel()]


    slicex = np.concatenate((slicex0, slicex1, slicex2))
    slicey = np.concatenate((slicey0, slicey1, slicey2))
    slicez = np.concatenate((slicez0, slicez1, slicez2))
    #slicey = w_[1][::2][shift_y_select.ravel()]
    #slicez = w_[2][::2][shift_z_select.ravel()]

    # TODO: I need 304
    base_array[slicex, slicey, slicez] = True
    return base_array




def compute_octree_branch(prev_octree: OctreeLevel, level) -> OctreeLevel:
    xyz    = prev_octree.xyz_coords
    dxdydz = prev_octree.grid.dxdydz
    ids    = prev_octree.id_block
    scalar = prev_octree.exported_fields.scalar_field

    xyz_8 = xyz.reshape((-1, 8, 3))
    uv_8 = ids.reshape((-1, 8))
    scalar_8 = scalar.reshape((-1, 8))

    shift_select_xyz = _mark_edges_sparse(uv_8)
    prev_octree.marked_edges = shift_select_xyz

    prev_xyz_cleaned = _mark_voxel_oct_level_sparse(shift_select_xyz, xyz_8)
    prev_octree.grid.xyz_coords = prev_xyz_cleaned
    # TODO: instead getting the edge find which node is closer to the layer in scalar units and pick that point.
    # TODO: This point is the voxel that contains the layer and where we have to apply the dual contouring

    # xyz_anchor = select_next_anchor(scalar_8, scalar_field_at_sp, xyz_8)


    _ = np.unique(_create_oct_level_sparse(shift_select_xyz, xyz_8), axis=0)
    xyz_anchor = _
    new_xyz = _create_oct_voxels(xyz_anchor[:, 0], xyz_anchor[:, 1], xyz_anchor[:, 2], *dxdydz, level=level)
    new_xyz2 = _create_oct_voxels(new_xyz[:, 0], new_xyz[:, 1], new_xyz[:, 2], *dxdydz, level=level)

    new_regular_grid = RegularGrid.init_regular_grid(prev_octree.grid.regular_grid.extent,
                                                     prev_octree.grid.regular_grid.resolution * 2)

    try:
        bool_regular_grid = set_bool_array_sparse(shift_select_xyz, prev_octree.grid.regular_grid)
        bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=0)
        bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=1)
        bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=2)
        new_regular_grid.active_cells = bool_regular_grid
    except:
        pass

    # TODO: Check is active_cells Trues equal to xyzcoord? -> No (it has to be yes)

    new_octree_grid = Grid(new_xyz2,
                           len_grids=[0, new_xyz2.shape[0]],
                           regular_grid=new_regular_grid,
                           custom_grid={
                               "centers": new_xyz,
                               "corners": new_xyz2
                           }
                           )

    new_octree_level = OctreeLevel(new_octree_grid)
    return new_octree_level


def select_next_anchor(scalar_8, scalar_field_at_sp, xyz_8):
    close = np.square(scalar_8 - scalar_field_at_sp[0])

    x_left_is_closer = close[:, :4] < close[:, 4:]
    y_left_is_closer = close[:, [0, 1, 4, 5]] < close[:, [2, 3, 6, 7]]
    z_left_is_closer = close[:, ::2] < close[:, 1::2]

    x_left = xyz_8[:, :4, :][x_left_is_closer]
    x_right = xyz_8[:, 4:, :][~x_left_is_closer]
    y_left = xyz_8[:, [0, 1, 4, 5], :][y_left_is_closer]
    y_right = xyz_8[:, [2, 3, 6, 7], :][~y_left_is_closer]
    z_left = xyz_8[:, ::2, :][z_left_is_closer]
    z_right = xyz_8[:, 1::2, :][~z_left_is_closer]

    xyz_anchor = np.vstack((
        # x_left, x_right,
        y_left, y_right,
        z_left, z_right))

    return xyz_anchor


def compute_octree_leaf(prev_octree: OctreeLevel):
    xyz = prev_octree.xyz_coords
    ids = prev_octree.id_block

    xyz_8 = xyz.reshape((-1, 8, 3))
    uv_8 = ids.reshape((-1, 8))

    shift_select_xyz = _mark_edges_sparse(uv_8)
    prev_octree.marked_edges = shift_select_xyz

    return _create_oct_last_level_sparse(shift_select_xyz, xyz_8)




def select_next_anchor_dense(shift_select_xyz: List[np.ndarray], regular_grid_xyz, scalar_field, scalar_field_at_sp):
    close = np.square(scalar_field - scalar_field_at_sp[0])
    x_left_is_closer = close[:-1, :, :][shift_select_xyz[0]] < close[1:, :, :][shift_select_xyz[0]]
    y_left_is_closer = close[:, :-1, :][shift_select_xyz[1]] < close[:, 1:, :][shift_select_xyz[1]]
    z_left_is_closer = close[:, :, :-1][shift_select_xyz[2]] < close[:, :, 1:][shift_select_xyz[2]]

    x_left = regular_grid_xyz[:-1, :, :][shift_select_xyz[0]][x_left_is_closer]
    x_right = regular_grid_xyz[1:, :, :][shift_select_xyz[0]][~x_left_is_closer]
    y_left = regular_grid_xyz[:, :-1, :][shift_select_xyz[1]][y_left_is_closer]
    y_right = regular_grid_xyz[:, 1:, :][shift_select_xyz[1]][~y_left_is_closer]
    z_left = regular_grid_xyz[:, :, :-1][shift_select_xyz[2]][z_left_is_closer]  # .reshape(-1, 3)
    z_right = regular_grid_xyz[:, :, 1:][shift_select_xyz[2]][~z_left_is_closer]  # .reshape(-1, 3)

    xyz_anchor = np.vstack((
        x_left, x_right, y_left, y_right,
        z_left, z_right))
    return xyz_anchor




def _mark_edges_sparse(uv_8):
    shift_x = uv_8[:, :4] - uv_8[:, 4:]
    shift_y = uv_8[:, [0, 1, 4, 5]] - uv_8[:, [2, 3, 6, 7]]
    shift_z = uv_8[:, ::2] - uv_8[:, 1::2]

    shift_x_select = np.not_equal(shift_x, 0)  # np.zeros_like(shift_x, dtype=bool)#np.not_equal(shift_x, 0)
    shift_y_select = np.not_equal(shift_y, 0)  # np.zeros_like(shift_y, dtype=bool)#np.not_equal(shift_y, 0)
    shift_z_select = np.not_equal(shift_z, 0)
    shift_select_xyz = [shift_x_select, shift_y_select, shift_z_select]
    return shift_select_xyz


def _create_oct_level_sparse(shift_select_xyz, xyz_8):
    x_edg = (xyz_8[:, :4, :][shift_select_xyz[0]] + xyz_8[:, 4:, :][shift_select_xyz[0]]) / 2
    y_edg = (xyz_8[:, [0, 1, 4, 5], :][shift_select_xyz[1]] + xyz_8[:, [2, 3, 6, 7], :][shift_select_xyz[1]]) / 2
    z_edg = (xyz_8[:, ::2, :][shift_select_xyz[2]] + xyz_8[:, 1::2, :][shift_select_xyz[2]]) / 2
    new_xyz_edg = np.vstack((x_edg, y_edg, z_edg))
    return new_xyz_edg


def _mark_voxel_oct_level_sparse(shift_select_xyz, xyz_8):
    x_left = xyz_8[:, :4, :][shift_select_xyz[0]]
    x_right = xyz_8[:, 4:, :][shift_select_xyz[0]]
    y_left = xyz_8[:, [0, 1, 4, 5], :][shift_select_xyz[1]]
    y_right = xyz_8[:, [2, 3, 6, 7], :][shift_select_xyz[1]]
    z_left = xyz_8[:, ::2, :][shift_select_xyz[2]]
    z_right = xyz_8[:, 1::2, :][shift_select_xyz[2]]

    previous_xyz_cleaned = np.vstack((x_left, x_right, y_left, y_right, z_left, z_right))
    return previous_xyz_cleaned


def _create_oct_last_level_sparse(shift_select_xyz, xyz_8):
    x_edg = (xyz_8[:, :4, :][shift_select_xyz[0]] + xyz_8[:, 4:, :][shift_select_xyz[0]]) / 2
    y_edg = (xyz_8[:, [0, 1, 4, 5], :][shift_select_xyz[1]] + xyz_8[:, [2, 3, 6, 7], :][shift_select_xyz[1]]) / 2
    z_edg = (xyz_8[:, ::2, :][shift_select_xyz[2]] + xyz_8[:, 1::2, :][shift_select_xyz[2]]) / 2
    new_xyz_edg = np.vstack((x_edg, y_edg, z_edg))
    return new_xyz_edg




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


def foo(x_edg, y_edg, z_edg, dx, dy, dz, level=1):
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
