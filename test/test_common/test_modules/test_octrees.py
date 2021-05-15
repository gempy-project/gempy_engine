import numpy as np
import pytest

from gempy_engine.modules.octrees.octrees_interface import create_oct_level_dense




def test_regular_grid_preparation(simple_grid_3d_more_points_grid):
    engine_grid = simple_grid_3d_more_points_grid
    print(engine_grid.regular_grid[45, 2, 4, 2])
    np.testing.assert_almost_equal(engine_grid.regular_grid[45, 2, 4, 2], .295)


def test_octree_1_level(simple_model_values_block, simple_grid_3d_more_points_grid):
    new_xyz = create_oct_level_dense(simple_model_values_block, simple_grid_3d_more_points_grid)
    print(new_xyz)



