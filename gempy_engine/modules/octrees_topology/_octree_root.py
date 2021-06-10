from typing import List

from gempy_engine.core.data.exported_structs import OctreeLevel
import numpy as np

from gempy_engine.core.data.grid import RegularGrid, Grid
from gempy_engine.modules.octrees_topology._octree_common import _generate_next_level_centers









def compute_octree_root(prev_octree: OctreeLevel, compute_topology=False, debug=False) -> Grid:
    def _mark_voxel(uv_8):
        shift_x = uv_8[:, :4] - uv_8[:, 4:]
        shift_y = uv_8[:, [0, 1, 4, 5]] - uv_8[:, [2, 3, 6, 7]]
        shift_z = uv_8[:, ::2] - uv_8[:, 1::2]

        shift_x_select = np.not_equal(shift_x, 0)
        shift_y_select = np.not_equal(shift_y, 0) #* ~shift_x_select
        shift_z_select = np.not_equal(shift_z, 0) #* ~shift_y_select * ~shift_x_select
        shift_select_xyz = np.array([shift_x_select, shift_y_select, shift_z_select])

        voxel_select = (shift_x_select + shift_y_select + shift_z_select).sum(axis=1, dtype=bool)
        return shift_select_xyz, voxel_select

    def _create_oct_level(shift_select_xyz, xyz_8):
        x_edg = (xyz_8[:, :4, :][shift_select_xyz[0]]           + xyz_8[:, 4:, :][shift_select_xyz[0]]) / 2
        y_edg = (xyz_8[:, [0, 1, 4, 5], :][shift_select_xyz[1]] + xyz_8[:, [2, 3, 6, 7], :][shift_select_xyz[1]]) / 2
        z_edg = (xyz_8[:, ::2, :][shift_select_xyz[2]]          + xyz_8[:, 1::2, :][shift_select_xyz[2]]) / 2
        new_xyz_edg = np.vstack((x_edg, y_edg, z_edg))
        return new_xyz_edg

    xyz = prev_octree.grid_faces.values
    dxdydz = prev_octree.dxdydz
    ids = prev_octree.output_faces.ids_block
    regular_grid_shape = prev_octree.grid_centers.regular_grid_shape


    xyz_8 = xyz.reshape((-1, 8, 3))
    uv_8 = ids.reshape((-1, 8))

    # Old octree
    shift_select_xyz, voxel_select = _mark_voxel(uv_8)
    prev_octree.marked_edges = shift_select_xyz


    if compute_topology: # TODO: Fix topology function
        prev_octree.edges_id, prev_octree.count_edges = _calculate_topology(shift_select_xyz, prev_octree.id_block)

    # New Octree

    xyz_anchor = prev_octree.grid_centers.values[voxel_select]
    #xyz_anchor = _create_oct_level(shift_select_xyz, xyz_8)
    xyz_coords = _generate_next_level_centers(xyz_anchor, dxdydz, level=1)

    bool_regular_grid = voxel_select
    if prev_octree.is_root:
        bool_regular_grid = bool_regular_grid.reshape(regular_grid_shape)
    else:
        bool_regular_grid = bool_regular_grid.reshape(-1, 2, 2, 2)

    grid_next_centers = Grid(
        xyz_coords,
        regular_grid=RegularGrid(
            prev_octree.grid_centers.regular_grid.extent,
            prev_octree.grid_centers.regular_grid.resolution * 2,
            bool_regular_grid
        ),
    )

    if debug:
        return (xyz_coords, xyz_anchor, shift_select_xyz, bool_regular_grid, voxel_select, grid_next_centers)
    else:
        return grid_next_centers


def compute_octree_root_on_faces(prev_octree: OctreeLevel, compute_topology=False, debug=False) -> Grid:
    def _mark_voxel(uv_6):
        shift_x = uv_6[0] - uv_6[1]
        shift_y = uv_6[2] - uv_6[3]
        shift_z = uv_6[4] - uv_6[5]

        shift_x_select = np.not_equal(shift_x, 0)
        shift_y_select = np.not_equal(shift_y, 0) * ~shift_x_select
        shift_z_select = np.not_equal(shift_z, 0) * ~shift_y_select * ~shift_x_select
        shift_select_xyz = np.array([shift_x_select, shift_y_select, shift_z_select])

        return shift_select_xyz

    def _create_oct_level(shift_select_xyz, xyz_6):
        return ((xyz_6[::2] + xyz_6[1::2]) / 2)[shift_select_xyz]

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
    xyz_coords = _generate_next_level_centers(xyz_anchor, dxdydz, level=1)

    bool_regular_grid = shift_select_xyz.sum(axis=0, dtype=bool)
    if prev_octree.is_root:
        bool_regular_grid = bool_regular_grid#.reshape(regular_grid_shape)
    else:
        bool_regular_grid = bool_regular_grid#.reshape(-1, 2, 2, 2)

    grid_next_centers = Grid(
        xyz_coords,
        regular_grid=RegularGrid(
            prev_octree.grid_centers.regular_grid.extent,
            prev_octree.grid_centers.regular_grid.resolution * 2,
            bool_regular_grid
        ),
    )

    if debug:
        return (xyz_coords, xyz_anchor, shift_select_xyz, bool_regular_grid)
    else:
        return grid_next_centers


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
