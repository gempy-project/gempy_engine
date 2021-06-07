from typing import List

import numpy as np

from ...core.data.exported_structs import OctreeLevel
from ...core.data.grid import Grid, RegularGrid
from . import _octree_root

# TODO: [ ] Check if fortran order speeds up this function
# TODO: Substitute numpy for b.tfnp
# TODO: Remove all stack to be able to compile TF


def get_next_octree_grid(prev_octree: OctreeLevel, compute_topology=False, **kwargs) -> Grid:
    return _octree_root.compute_octree_root(prev_octree, compute_topology, **kwargs)


def get_regular_grid_for_level(octree_list: List[OctreeLevel], level: int):
    selected_octree = octree_list[level]
    regular_grid = np.zeros(selected_octree.grid_centers.regular_grid_shape.prod(), dtype=float)


    active_cells_index_foo = np.arange((selected_octree.grid_centers.regular_grid_shape.prod()))
    root = octree_list[0]

    regular_grid = _expand_regular_grid(root.output_centers.ids_block.reshape(root.grid_centers.regular_grid_shape), level)
    aci = []

    for e, octree in enumerate(octree_list[1:level+1]):
        n_rep = (level  - e)
        #n_rep = 1
        active_cells = octree.grid_centers.regular_grid.active_cells


        is_branch = len(active_cells.shape) == 4

        #if n_rep > 0:
        if is_branch:
#               ids = octree.output_centers.ids_block.reshape(-1, 2, 2)
            active_cells = _expand_octree(active_cells, n_rep)
        else:
#                ids = octree.output_centers.ids_block.reshape(octree.grid_centers.regular_grid_shape)
            active_cells = _expand_regular_grid(active_cells, n_rep)

        ids = _expand_octree(octree.output_centers.ids_block.reshape(-1, 2, 2, 2), n_rep - 1)
        active_cells_index = np.where(active_cells)

        active_cells_index_foo = active_cells_index_foo[active_cells_index]
        aci.append(active_cells_index) # TODO: Unused

        regular_grid[active_cells_index_foo] = ids


   # regular_grid[active_cells_index_foo] = selected_octree.output_centers.ids_block
    return regular_grid


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