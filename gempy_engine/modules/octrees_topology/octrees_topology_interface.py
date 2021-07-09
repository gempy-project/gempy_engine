from typing import List

import numpy as np

from ._octree_internals import compute_next_octree_locations
from ...core.data.exported_structs import OctreeLevel
from ...core.data.grid import Grid


# TODO: [ ] Check if fortran order speeds up this function
# TODO: Substitute numpy for b.tfnp
# TODO: Remove all stack to be able to compile TF


def get_next_octree_grid(prev_octree: OctreeLevel, compute_topology=False, **kwargs) -> Grid:
    return compute_next_octree_locations(prev_octree, compute_topology, **kwargs)


def get_regular_grid_for_level(octree_list: List[OctreeLevel], level: int):

    # region Internal Functions
    def calculate_oct(shape, n_rep):

        f1 = shape[2] * shape[1] * 2 ** (n_rep - 1) * 2 ** (n_rep - 1)
        f2 = shape[2] * 2 ** (n_rep - 1)

        e = 2 ** n_rep

        n_voxel_per_dim = np.arange(e)

        d1 = np.repeat(n_voxel_per_dim, e ** 2) * f1
        d2 = np.tile(np.repeat(n_voxel_per_dim, e), e) * f2
        d3 = np.tile(n_voxel_per_dim, e ** 2)

        oct = d1 + d2 + d3
        return oct

    def get_global_anchor(activ, branch_res, n_rep):
        f1 = branch_res[2] * branch_res[1]
        f2 = branch_res[2]

        f1b = f1 * 2 ** (3 * n_rep)
        f2b = f2 * 2 ** (2 * n_rep)
        f3b =      2 ** (1 * n_rep)

        d1 = (activ // f1) * f1b
        d2 = (activ % f1 // f2) * f2b
        d3 = activ % f1 % f2 * f3b

        anchor = d1 + d2 + d3
        return anchor

    def _expand_regular_grid(active_cells, n_rep):
        active_cells = np.repeat(active_cells, 2 ** n_rep, axis=0)
        active_cells = np.repeat(active_cells, 2 ** n_rep, axis=1)
        active_cells = np.repeat(active_cells, 2 ** n_rep, axis=2)
        return active_cells.ravel()

    def _expand_octree(active_cells, n_rep):
        active_cells = np.repeat(active_cells, 2 ** n_rep, axis=1)
        active_cells = np.repeat(active_cells, 2 ** n_rep, axis=2)
        active_cells = np.repeat(active_cells, 2 ** n_rep, axis=3)

        return active_cells.ravel()

    # endregion
    if level > len(octree_list):
        raise ValueError("Level cannot be larger than the number of octrees.")

    # Octree - Level 0
    root = octree_list[0]
    regular_grid = _expand_regular_grid(root.output_centers.ids_block.reshape(root.grid_centers.regular_grid_shape), level)
    shape = root.grid_centers.regular_grid_shape

    aci = []

    # Octree - Level n
    for e, octree in enumerate(octree_list[1:level + 1]):
        n_rep = (level - e)
        active_cells = octree.grid_centers.regular_grid.active_cells

        local_active_cells =  np.where(active_cells)[0]
        shape = octree.grid_centers.regular_grid_shape
        oct = calculate_oct(shape, n_rep)
        ids = _expand_octree(octree.output_centers.ids_block.reshape(-1, 2, 2, 2), n_rep - 1)

        is_branch = e > 0
        if is_branch:
            local_shape = np.array([2,2,2])
            local_anchors = get_global_anchor(local_active_cells, local_shape, n_rep)
            global_anchors = global_active_cells_index[local_anchors]

            global_active_cells_index = (global_anchors.reshape(-1, 1) + oct).ravel()
            regular_grid[global_active_cells_index] = ids #+ (e * 2)

        else:
            local_shape = shape//2
            local_anchors = get_global_anchor(local_active_cells, local_shape, n_rep)
            global_active_cells_index = (local_anchors.reshape(-1, 1) + oct).ravel()
            regular_grid[global_active_cells_index] = ids #+ (e * 2)

        aci.append(global_active_cells_index)  # TODO: Unused

    return regular_grid.reshape(shape)


