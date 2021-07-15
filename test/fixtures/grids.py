import dataclasses

import pytest
import numpy as np

from gempy_engine.core.data.grid import RegularGrid, Grid


def simple_grid_2d_f():
    nx, ny = (5, 5)
    x = np.linspace(0, 5, nx)
    y = np.linspace(0, 5, ny)
    xv, yv = np.meshgrid(x, y)
    g = np.vstack((xv.ravel(), yv.ravel())).T
    return g


@pytest.fixture(scope='session')
def simple_grid_2d():
    return simple_grid_2d_f()


@pytest.fixture(scope="session")
def simple_grid_3d_more_points_grid():
    resolution = [50, 5, 50]
    extent = [0.25, .75, 0.25, .75, 0.25, .75]

    regular_grid = RegularGrid(extent, resolution)
    grid = Grid.from_regular_grid(regular_grid)
    return grid

@pytest.fixture(scope="session")
def simple_grid_3d_octree():
    resolution = [2, 2, 3]
    extent = [0.25, .75, 0.25, .75, 0.25, .75]

    regular_grid = RegularGrid(extent, resolution)
    grid = Grid.from_regular_grid(regular_grid)
    return dataclasses.replace(grid)
