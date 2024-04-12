import dataclasses

import pytest
import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.regular_grid import RegularGrid


def simple_grid_2d_f():
    nx, ny = (5, 5)
    x = np.linspace(0, 5, nx)
    y = np.linspace(0, 5, ny)
    xv, yv = np.meshgrid(x, y)
    g = np.vstack((xv.ravel(), yv.ravel())).T
    BackendTensor.t.array(g)
    return g


@pytest.fixture(scope='session')
def simple_grid_2d():
    return simple_grid_2d_f()


@pytest.fixture(scope="session")
def simple_grid_3d_more_points_grid():
    resolution = [50, 5, 50]
    extent = [0.25, .75, 0.25, .75, 0.25, .75]

    regular_grid = RegularGrid(extent, resolution)
    grid = EngineGrid.from_regular_grid(regular_grid)
    return grid


@pytest.fixture(scope="session")
def simple_grid_3d_octree():
    grid = _gen_grid()
    return grid


def _gen_grid():
    resolution = [2, 2, 2]
    extent = [0.25, .75, 0.25, .75, 0.25, .75]
    regular_grid = RegularGrid(extent, resolution)
    grid = EngineGrid.from_regular_grid(regular_grid)
    return grid
