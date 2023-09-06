import warnings
from typing import List

from gempy_engine.core.data.octree_level import OctreeLevel
import numpy as np

from gempy_engine.core.data.grid import Grid
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.modules.octrees_topology._octree_common import _generate_next_level_centers


def compute_next_octree_locations(prev_octree: OctreeLevel, compute_topology=False) -> Grid:
    list_ixd_select = []

    def _mark_voxel(uv_8):
        shift_x = uv_8[:, :4] - uv_8[:, 4:]
        shift_y = uv_8[:, [0, 1, 4, 5]] - uv_8[:, [2, 3, 6, 7]]
        shift_z = uv_8[:, ::2] - uv_8[:, 1::2]

        shift_x_select = np.not_equal(shift_x, 0)
        shift_y_select = np.not_equal(shift_y, 0)
        shift_z_select = np.not_equal(shift_z, 0)
        shift_select_xyz = np.array([shift_x_select, shift_y_select, shift_z_select])

        idx_select_x = shift_x_select.sum(axis=1, dtype=bool)
        idx_select_y = shift_y_select.sum(axis=1, dtype=bool)
        idx_select_z = shift_z_select.sum(axis=1, dtype=bool)
        list_ixd_select.append(idx_select_x)
        list_ixd_select.append(idx_select_y)
        list_ixd_select.append(idx_select_z)

        voxel_select = (shift_x_select + shift_y_select + shift_z_select).sum(axis=1, dtype=bool)
        return shift_select_xyz, voxel_select

    dxdydz = prev_octree.dxdydz
    ids = prev_octree.last_output_corners.litho_faults_ids

    uv_8 = ids.reshape((-1, 8))

    # Old octree
    shift_select_xyz, voxel_select = _mark_voxel(uv_8)
    prev_octree.marked_edges = shift_select_xyz

    if compute_topology:  # TODO: Fix topology function
        prev_octree.edges_id, prev_octree.count_edges = _calculate_topology(shift_select_xyz, prev_octree.id_block)

    # New Octree
    xyz_anchor = prev_octree.grid_centers.regular_grid.values[voxel_select]
    xyz_coords, bool_idx = _generate_next_level_centers(xyz_anchor, dxdydz, level=1)

    grid_next_centers = Grid(
        regular_grid=RegularGrid.from_octree_level(
            xyz_coords_octree=xyz_coords,
            previous_regular_grid=prev_octree.grid_centers.regular_grid,
            active_cells=voxel_select,
            left_right=bool_idx
        ),
    )

    if True:
        grid_next_centers.debug_vals = (xyz_coords, xyz_anchor, shift_select_xyz, bool_idx, voxel_select, grid_next_centers)
        return grid_next_centers  # TODO: This is going to break the tests that were using this
    else:
        return grid_next_centers


def _calculate_topology(shift_select_xyz: List[np.ndarray], ids: np.ndarray):
    """This is for the typology of level 0. Probably for the rest of octtrees
    levels it will be a bit different
    """
    raise NotImplementedError

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
