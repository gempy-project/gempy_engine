import numpy as np
import pytest

from gempy_engine.modules.octrees_topology.octrees_topology_interface import _create_oct_level_dense, compute_topology
import matplotlib.pyplot as plt



def test_regular_grid_preparation(simple_grid_3d_more_points_grid):
    engine_grid = simple_grid_3d_more_points_grid
    print(engine_grid.regular_grid[45, 2, 4, 2])
    np.testing.assert_almost_equal(engine_grid.regular_grid[45, 2, 4, 2], .295)


def test_octree_level_0(simple_model_values_block, simple_grid_3d_more_points_grid):
    new_xyz = _create_oct_level_dense(simple_model_values_block, simple_grid_3d_more_points_grid)
    print(new_xyz)

    slice = 2

    if True:
        plt.contourf(
            simple_model_values_block[0][:simple_grid_3d_more_points_grid.len_grids[0]]
                .reshape(50, 5, 50)[:, slice, :].T, N=40, cmap="viridis",
            extent=(0.25, .75, 0.25, .75)
        )

        plt.scatter(new_xyz[:, 0], new_xyz[:, 2], c="w", s= .5)

        plt.colorbar()
        plt.show()


def test_octree_level_1(simple_model_values_block, simple_grid_3d_more_points_grid):
    new_xyz = _create_oct_level_dense(simple_model_values_block, simple_grid_3d_more_points_grid)
    print(new_xyz)

    slice = 2

    if True:
        plt.contourf(
            simple_model_values_block[0][:simple_grid_3d_more_points_grid.len_grids[0]]
                .reshape(50, 5, 50)[:, slice, :].T, N=40, cmap="viridis",
            extent=(0.25, .75, 0.25, .75)
        )

        plt.scatter(new_xyz[:, 0], new_xyz[:, 2], c="w", s= .5)

        plt.colorbar()
        plt.show()


def test_topology_level_0(simple_model_values_block, simple_grid_3d_more_points_grid):
    edges_id, count_edges = compute_topology(simple_model_values_block, simple_grid_3d_more_points_grid)

    np.testing.assert_array_almost_equal(count_edges, np.array([413, 32], dtype=int))

    print(f"Edges id: {edges_id}")
    print(f"Count edges: {count_edges}")