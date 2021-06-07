from typing import List

from gempy_engine.core.data.exported_structs import OctreeLevel
import numpy as np

from gempy_engine.core.data.grid import RegularGrid, Grid
from gempy_engine.modules.octrees_topology._octree_common import _generate_corners, _expand, _expand2


def _mark_voxel(uv_6):
    shift_x = uv_6[0]       - uv_6[1]
    shift_y = uv_6[2]       - uv_6[3]
    shift_z = uv_6[4]       - uv_6[5]

    shift_x_select = np.not_equal(shift_x, 0)  # np.zeros_like(shift_x, dtype=bool)#np.not_equal(shift_x, 0)
    shift_y_select = np.not_equal(shift_y, 0) * ~shift_x_select # np.zeros_like(shift_y, dtype=bool)#np.not_equal(shift_y, 0)
    shift_z_select = np.not_equal(shift_z, 0) * ~shift_y_select * ~shift_x_select
    shift_select_xyz2 = np.array([shift_x_select, shift_y_select, shift_z_select])

    # shift_y_select = np.bitwise_xor(shift_x_select, np.not_equal(shift_y, 0))
    # shift_z_select = np.bitwise_xor(shift_y_select, np.not_equal(shift_z, 0))
    # shift_select_xyz = np.array([shift_x_select, shift_y_select, shift_z_select])


    return shift_select_xyz2


def _create_oct_level(shift_select_xyz, xyz_6):
    # x_center = (xyz_6[0][shift_select_xyz[0]] + xyz_6[1][shift_select_xyz[0]]) / 2
    # y_center = (xyz_6[2][shift_select_xyz[1]] + xyz_6[3][shift_select_xyz[1]]) / 2
    # z_center = (xyz_6[4][shift_select_xyz[2]] + xyz_6[5][shift_select_xyz[2]]) / 2
    # new_xyz_center = np.vstack((
    #     x_center,
    #     y_center,
    #     z_center))

    new_xyz_center_a = ((xyz_6[::2] + xyz_6[1::2]) / 2)[0, shift_select_xyz.sum(axis=0, dtype=bool)]
    new_xyz_center_b = ((xyz_6[::2] + xyz_6[1::2]) / 2)[1, shift_select_xyz.sum(axis=0, dtype=bool)]
    new_xyz_center_c = ((xyz_6[::2] + xyz_6[1::2]) / 2)[2, shift_select_xyz.sum(axis=0, dtype=bool)]
    a = ((xyz_6[::2] + xyz_6[1::2]) / 2)[shift_select_xyz]
    new_xyz_center_d = np.unique(a, axis=0)
    return new_xyz_center_d


def compute_octree_root(prev_octree: OctreeLevel, compute_topology=False, debug=False) -> Grid:
    xyz = prev_octree.grid_faces.values
    dxdydz = prev_octree.dxdydz
    ids = prev_octree.output_faces.ids_block
    regular_grid_shape = prev_octree.grid_centers.regular_grid_shape


    xyz_6 = xyz.reshape((6, -1, 3))
    uv_6 = ids.reshape((6, -1))

    # Old octree
    shift_select_xyz = _mark_voxel(uv_6)
    prev_octree.marked_edges = shift_select_xyz


    if compute_topology: # TODO: Fix topology function
        prev_octree.edges_id, prev_octree.count_edges = _calculate_topology(shift_select_xyz, prev_octree.id_block)

    # New Octree

    xyz_anchor = _create_oct_level(shift_select_xyz, xyz_6)
    xyz_coords = _generate_corners(xyz_anchor, dxdydz, level=1)

    bool_regular_grid1= np.tile(np.repeat(shift_select_xyz, 2, axis=1), (1, 4)).sum(axis=0, dtype=bool)
    # bool_regular_grid = _expand2(shift_select_xyz[[0]],
    #                              shift_select_xyz[[1]],
    #                              shift_select_xyz[[2]]).sum(axis=1, dtype=bool)
   # bool_regular_grid = np.repeat(shift_select_xyz.sum(axis=0, dtype=bool), 8)

    bool_regular_grid = shift_select_xyz.sum(axis=0, dtype=bool)
    # bool_regular_grid2 = shift_select_xyz.sum(axis=0, dtype=bool)#_set_bool_array(shift_select_xyz, prev_octree.grid.regular_grid.resolution)
    if prev_octree.is_root:
        bool_regular_grid = bool_regular_grid.reshape(regular_grid_shape)  # TODO: This is not going to be possible in the next level
        # bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=0)
        # bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=1)
        # bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=2)
    else:
        bool_regular_grid = bool_regular_grid.reshape(-1, 2, 2, 2)

        # bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=1)
        # bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=2)
        # bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=3)
        #

    grid_next_centers = Grid(
        xyz_coords,
        regular_grid=RegularGrid(
            prev_octree.grid_centers.regular_grid.extent,
            prev_octree.grid_centers.regular_grid.resolution * 2,
            bool_regular_grid
        ),
    )
    #
    # new_regular_grid: RegularGrid = RegularGrid.init_regular_grid(prev_octree.grid.regular_grid.extent,
    #                                                               prev_octree.grid.regular_grid.resolution * 2)
    #
    # bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=0)
    # bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=1)
    # bool_regular_grid = np.repeat(bool_regular_grid, 2, axis=2)
    #
    # new_regular_grid.active_cells = bool_regular_grid
    #
    # # TODO: Check is active_cells Trues equal to xyzcoord? -> Yes
    #
    # new_octree_grid = Grid(xyz_coords2,
    #                        len_grids=[0, xyz_coords2.shape[0]],
    #                        regular_grid=new_regular_grid,
    #                        custom_grid={
    #                            "centers": xyz_coords,
    #                            "corners": xyz_coords2
    #                        }
    #                        )
    #
    # new_octree_level = OctreeLevel(new_octree_grid)
    if debug:
        return (xyz_coords, xyz_anchor, shift_select_xyz, bool_regular_grid)
    else:
        return grid_next_centers

def _create_oct_level_dense(shift_select_xyz: List[np.ndarray], regular_grid_xyz):
    x_edg = (regular_grid_xyz[:-1, :, :][shift_select_xyz[0]] + regular_grid_xyz[1:, :, :][shift_select_xyz[0]]) / 2
    y_edg = (regular_grid_xyz[:, :-1, :][shift_select_xyz[1]] + regular_grid_xyz[:, 1:, :][shift_select_xyz[1]]) / 2
    z_edg = (regular_grid_xyz[:, :, :-1][shift_select_xyz[2]] + regular_grid_xyz[:, :, 1:][shift_select_xyz[2]]) / 2

    new_xyz_edg = np.vstack((x_edg, y_edg, z_edg))

    return new_xyz_edg  # _create_oct_voxels(new_xyz_edg[:, 0], new_xyz_edg[:, 1], new_xyz_edg[:, 2], *dxdydz, level=1)



def _mark_edge_voxels_dense(ids):
    shift_x = ids[1:, :, :] - ids[:-1, :, :]
    shift_y = ids[:, 1:, :] - ids[:, :-1, :]
    shift_z = ids[:, :, 1:] - ids[:, :, :-1]

    shift_x_select = np.not_equal(shift_x, 0)
    shift_y_select = np.not_equal(shift_y, 0)
    shift_z_select = np.not_equal(shift_z, 0)

    shift_select_xyz = [shift_x_select, shift_y_select, shift_z_select]

    return shift_select_xyz


def _calculate_topology(shift_select_xyz: List[np.ndarray], ids: np.ndarray):
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

def _set_bool_array(shift_select_xyz, resolution):
    base_array = np.zeros(resolution, dtype=bool)
    shift_x_select, shift_y_select, shift_z_select = shift_select_xyz

    base_array[1:, :, :] = shift_x_select
    base_array[:-1, :, :] = shift_x_select

    base_array[:, 1:, :] = shift_y_select
    base_array[:, :-1, :] = shift_y_select

    base_array[:, :, 1:] = shift_z_select
    base_array[:, :, :-1] = shift_z_select

    return base_array
