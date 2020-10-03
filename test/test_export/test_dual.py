import pytest
import numpy as np

from gempy_engine.systems.evalution.dual_kriging import export_scalar
from gempy_engine.systems.kernel.kernel import input_dips_points


@pytest.fixture
def grid():
    nx, ny = (3, 2)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    g = np.vstack((xv.ravel(), yv.ravel())).T
    return g


def test_hu_simPoint(simple_model, grid):
    e = export_scalar(*simple_model, grid)