import numpy as np
import pytest

from gempy_engine.core.data.exported_structs import OctreeLevel
from gempy_engine.modules.octrees_topology.octrees_topology_interface import _create_oct_level_dense, \
    calculate_topology, compute_octree_level_0
import matplotlib.pyplot as plt



def test_regular_grid_preparation(simple_grid_3d_more_points_grid):
    engine_grid = simple_grid_3d_more_points_grid
    print(engine_grid.regular_grid[45, 2, 4, 2])
    np.testing.assert_almost_equal(engine_grid.regular_grid[45, 2, 4, 2], .295)


def test_octree_and_topo_root(simple_model_values_block_output, simple_grid_3d_more_points_grid):

    grid = simple_grid_3d_more_points_grid
    output = simple_model_values_block_output

    octree_lvl0 = OctreeLevel(grid.values, output.ids_block_regular_grid,
                              output.exported_fields_regular_grid,
                              is_root=True)

    octree_lvl1 = compute_octree_level_0(octree_lvl0, grid.regular_grid, grid.dxdydz, compute_topology=True)

    np.testing.assert_array_almost_equal(octree_lvl0.count_edges, np.array([413, 32], dtype=int))

    print(f"Edges id: {octree_lvl0.edges_id}")
    print(f"Count edges: {octree_lvl0.count_edges}")

    slice = 2

    if True:
        plt.contourf(
            octree_lvl0.id_block[:, slice, :].T, N=40, cmap="viridis",
            extent=(0.25, .75, 0.25, .75)
        )

        new_xyz = octree_lvl1.xyz_coords
        plt.scatter(new_xyz[:, 0], new_xyz[:, 2], c="w", s= .5)

        plt.colorbar()
        plt.show()
