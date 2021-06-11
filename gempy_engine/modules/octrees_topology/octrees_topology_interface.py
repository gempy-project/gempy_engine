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
    def calculate_oct(shape, n_rep):
        s_z = shape[2] * n_rep
        s_y = shape[1] * n_rep
        n_voxel_per_dim = np.arange(2 ** n_rep)

        f1 = shape[2] * shape[1] * 2 ** (n_rep - 1) * 2 ** (n_rep - 1)
        f2 = shape[2] * 2 ** (n_rep - 1)

        e = 2 ** n_rep

        n_voxel_per_dim = np.arange(e)

        d1 = np.repeat(n_voxel_per_dim, e ** 2) * f1
        d2 = np.tile(np.repeat(n_voxel_per_dim, e), e) * f2
        d3 = np.tile(n_voxel_per_dim, e ** 2)
        oct2 = np.array([n_voxel_per_dim,
                         s_z + n_voxel_per_dim,
                         (s_z * s_y) + n_voxel_per_dim,
                         (s_z * s_y) + s_z + n_voxel_per_dim]).reshape(1, -1)
        oct = d1 + d2 + d3
        return oct

    def foo_down(activ, branch_res, n_rep):
        original_res = (branch_res // 2)
        f1 = original_res[2] * original_res[1]
        f2 = original_res[2]

        f1b = f1 * 2 ** (3 * n_rep)  # 64 #** 2**(n_rep-1) * 2**(n_rep-1)
        f2b = f2 * 2 ** (2 * n_rep)  # 16 #** 2**(n_rep-1)
        f3b =      2 ** (1 * n_rep)

        d1 = (activ // f1) * f1b  # * 2
        d2 = (activ % f1 // f2) * f2b  # * 2
        d3 = activ % f1 % f2 * f3b

        anchor = d1 + d2 + d3
        return anchor

    selected_octree = octree_list[level]
    regular_grid = np.zeros(selected_octree.grid_centers.regular_grid_shape.prod(), dtype=float)

    active_cells_index_foo: np.ndarray = None  # np.arange((selected_octree.grid_centers.regular_grid_shape.prod()))
    root = octree_list[0]

    # TODO: Here I add the lvl 0
    regular_grid = _expand_regular_grid(root.output_centers.ids_block.reshape(root.grid_centers.regular_grid_shape), level)
    shape = root.grid_centers.regular_grid_shape

    aci = []

    for e, octree in enumerate(octree_list[1:level + 1]):
        n_rep = (level - e)
        # n_rep = 1
        active_cells = octree.grid_centers.regular_grid.active_cells

        is_branch = e > 0

        # # if n_rep > 0:
        if is_branch:
            a = np.where(active_cells.ravel())[0]

            def foo_down2(activ, branch_res, n_rep):
                original_res = (branch_res)
                f1 = original_res[2] * original_res[1]
                f2 = original_res[2]

                f1b = f1 * 2 ** (3 * n_rep)  # 64 #** 2**(n_rep-1) * 2**(n_rep-1)
                f2b = f2 * 2 ** (2 * n_rep)  # 16 #** 2**(n_rep-1)
                f3b =      2 ** (1 * n_rep)

                d1 = (activ // f1) * f1b  # * x
                d2 = (activ % f1 // f2) * f2b  # y
                d3 = activ % f1 % f2 * f3b # z

                anchor = d1 + d2 + d3
                return anchor

            shape = octree.grid_centers.regular_grid_shape
            b = foo_down2(a[:],np.array([2,2,2]), n_rep )
            c = active_cells_index_foo[b]

            oct = calculate_oct(shape, n_rep)
            active_cells_index = (c.reshape(-1, 1) + oct).ravel()
            active_cells_index_foo = active_cells_index
            ids = _expand_octree(octree.output_centers.ids_block.reshape(-1, 2, 2, 2), n_rep - 1)

            regular_grid[active_cells_index_foo] = (e * 2)  + ids
            # if e == 2:
            #     n = -48
            #     regular_grid[c[78+n]] = 10
            #     regular_grid[c[79+n]] = 9
            #     regular_grid[c[80+n]] = 8
            #     regular_grid[c[81+n]] = 7
            #     regular_grid[c[50:65]] = 6
            #regular_grid[active_cells_index] = ids

            #               ids = octree.output_centers.ids_block.reshape(-1, 2, 2)
            # foo = np.where(active_cells.ravel())[0]
            #
            # # bar = np.repeat(foo, 8)
            # shape = octree.grid_centers.regular_grid_shape
            #
            # def foo_down2(activ, branch_res, n_rep):
            #
            #     anchor = activ.reshape(-1,2,2,2)
            #     return anchor
            #
            # a = foo_down2(foo, shape, n_rep)
            # foo = _expand_octree2(active_cells, n_rep).reshape(5, 64) # TODO: This is expansion has to be more clever
            # a = np.where(foo)[0]
            #
            # act_bar = active_cells_index_foo.reshape(-1, 64)
            #
            # oct2 = calculate_oct(shape, n_rep)
            # oct = np.array([7,6,3,2,5,4,1,2])
            # #active_cells_index =( a + oct).ravel() # TODO: THIS IS WRONG!
            # n1 = 0
            # n2 = 500 #* 2
            # active_cells_index_foo2 = active_cells_index_foo[a[n1:n2]]
            # ids = _expand_octree(octree.output_centers.ids_block.reshape(-1, 2, 2, 2), n_rep - 1)
            #
            # aa = _expand_octree2(active_cells, n_rep).reshape(-1, 64)
            # active_cells_index_foo3 = act_bar[aa][n1:n2]
            # regular_grid[active_cells_index_foo3] = ids[n1:n2]
          #  regular_grid[active_cells_index_foo] = ids[:6]
            #  regular_grid[active_cells_index] = 150
            # regular_grid[a] = 300

        else:
            ac = active_cells


            foo = np.where(ac.ravel())[0]

            # bar = np.repeat(foo, 8)
            shape = octree.grid_centers.regular_grid_shape

            a = foo_down(foo, shape, n_rep)


            oct = calculate_oct(shape, n_rep)

            active_cells_index = (a.reshape(-1, 1) + oct).ravel()

            ids = _expand_octree(octree.output_centers.ids_block.reshape(-1, 2, 2, 2), n_rep - 1)
            #ids = octree.output_centers.ids_block


            if active_cells_index_foo is None:
                active_cells_index_foo = active_cells_index
            else:
                active_cells_index_foo = active_cells_index_foo[active_cells_index] # TODO: We should never get here

            aci.append(active_cells_index)  # TODO: Unused

            regular_grid[active_cells_index_foo] = ids + (e * 2)
          #  regular_grid[active_cells_index_foo] = ids
  #       regular_grid[active_cells_index_foo] = ids[:] * (e * 2 +1)
  #
  # #  regular_grid[active_cells_index] = 150
  #  # regular_grid[a] = 300

    return regular_grid.reshape(shape)


def _expand_regular_grid(active_cells, n_rep):
    active_cells = np.repeat(active_cells, 2 ** n_rep, axis=0)
    active_cells = np.repeat(active_cells, 2 ** n_rep, axis=1)
    active_cells = np.repeat(active_cells, 2 ** n_rep, axis=2)
    return active_cells.ravel()


def _expand_octree(active_cells, n_rep):
    active_cells = active_cells[:, :, :, :]
    #active_cells = np.swapaxes(active_cells, 1,3)
    active_cells = np.repeat(active_cells, 2 ** n_rep, axis=1)
    active_cells = np.repeat(active_cells, 2 ** n_rep, axis=2)
    active_cells = np.repeat(active_cells, 2 ** n_rep, axis=3)
#    active_cells = np.tile(active_cells, (2 ** n_rep, 2 ** n_rep, 2 ** n_rep))

    # active_cells = np.repeat(active_cells, 2 ** n_rep, axis=3)
    return active_cells.ravel()


def _expand_octree2(active_cells, n_rep):

    active_cells = np.repeat(active_cells, 2 ** n_rep, axis=1)
    active_cells = np.repeat(active_cells, 2 ** n_rep, axis=2)
    active_cells = np.repeat(active_cells, 2 ** n_rep, axis=3)


    return active_cells