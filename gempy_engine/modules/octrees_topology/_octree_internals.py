from typing import List

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_structs import OctreeLevel
import numpy as np

from gempy_engine.core.data.grid import RegularGrid, Grid
from gempy_engine.modules.octrees_topology._octree_common import _generate_next_level_centers

tfnp = BackendTensor.tfnp
tensor_types = BackendTensor.tensor_types

def compute_next_octree_locations(prev_octree: OctreeLevel, compute_topology=False, debug=False) -> Grid:
    def _mark_voxel(uv_8):
        """
        TF attempt:

        shift_x = uv_8[:, :4] - uv_8[:, 4:]
        if BackendTensor.engine_backend is AvailableBackends.tensorflow or False:
            shift_y_l = tfnp.gather(uv_8, [0, 1, 4, 5], axis=1)
            shift_y_r = tfnp.gather(uv_8, [0, 1, 4, 5], axis=1)
            shift_y = shift_y_l - shift_y_r

        else:
            shift_y = uv_8[:, [0, 1, 4, 5]] - uv_8[:, [2, 3, 6, 7]]
        shift_z = uv_8[:, ::2] - uv_8[:, 1::2]

        shift_x_select = tfnp.not_equal(shift_x, 0)
        shift_y_select = tfnp.not_equal(shift_y, 0)
        shift_z_select = tfnp.not_equal(shift_z, 0)
        shift_select_xyz = tfnp.concat([shift_x_select, shift_y_select, shift_z_select], axis=0)#np.array([shift_x_select, shift_y_select, shift_z_select])

        shift_select_x_int = tfnp.cast(shift_x_select, tfnp.int32)
        shift_select_y_int = tfnp.cast(shift_y_select, tfnp.int32)
        shift_select_z_int = tfnp.cast(shift_z_select, tfnp.int32)

        voxel_select_aux = tfnp.sum(shift_select_x_int + shift_select_y_int + shift_select_z_int, axis=1)#.sum(axis=1)
        voxel_select = tfnp.cast(voxel_select_aux, tfnp.bool)


        """

        shift_x = uv_8[:, :4] - uv_8[:, 4:]
        shift_y = uv_8[:, [0, 1, 4, 5]] - uv_8[:, [2, 3, 6, 7]]
        shift_z = uv_8[:, ::2] - uv_8[:, 1::2]

        shift_x_select = np.not_equal(shift_x, 0)
        shift_y_select = np.not_equal(shift_y, 0)
        shift_z_select = np.not_equal(shift_z, 0)
        shift_select_xyz = np.array([shift_x_select, shift_y_select, shift_z_select])

        voxel_select = (shift_x_select + shift_y_select + shift_z_select).sum(axis=1, dtype=bool)
        return shift_select_xyz, voxel_select

    dxdydz = prev_octree.dxdydz
    ids = prev_octree.output_corners.ids_block

    uv_8 = np.reshape(ids, (-1, 8))#.reshape((-1, 8))

    # Old octree
    shift_select_xyz, voxel_select = _mark_voxel(uv_8)
    prev_octree.marked_edges = shift_select_xyz

    if compute_topology: # TODO: Fix topology function
        prev_octree.edges_id, prev_octree.count_edges = _calculate_topology(shift_select_xyz, prev_octree.id_block)

    # New Octree
    xyz_anchor = prev_octree.grid_centers.values[voxel_select]
    xyz_coords = _generate_next_level_centers(xyz_anchor, dxdydz, level=1)

    grid_next_centers = Grid(
        xyz_coords,
        regular_grid=RegularGrid(
            prev_octree.grid_centers.regular_grid.extent,
            prev_octree.grid_centers.regular_grid.resolution * 2,
            voxel_select
        ),
    )

    if debug:
        return (xyz_coords, xyz_anchor, shift_select_xyz, None, voxel_select, grid_next_centers)
    else:
        return grid_next_centers


def compute_octree_root_on_faces(prev_octree: OctreeLevel, debug=False) -> Grid:
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

    xyz = prev_octree.grid_corners.values
    dxdydz = prev_octree.dxdydz
    ids = prev_octree.output_corners.ids_block

    xyz_6 = xyz.reshape((6, -1, 3))
    uv_6 = ids.reshape((6, -1))

    # Old octree
    shift_select_xyz = _mark_voxel(uv_6)
    prev_octree.marked_edges = shift_select_xyz

    # New Octree
    xyz_anchor = _create_oct_level(shift_select_xyz, xyz_6)
    xyz_coords = _generate_next_level_centers(xyz_anchor, dxdydz, level=1)

    bool_regular_grid = shift_select_xyz.sum(axis=0, dtype=bool)

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

