import pytest
import numpy as np

from gempy_engine.systems.evalution.dual_kriging import export_scalar, hv_x0, export
from gempy_engine.systems.kernel.kernel import input_dips_points


@pytest.fixture
def grid():
    nx, ny = (3, 2)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    g = np.vstack((xv.ravel(), yv.ravel())).T
    return g


def test_export_scalar(simple_model, grid):
    from gempy_engine.systems.kernel.kernel_functions import cubic_function_p_div_r
    kernel = cubic_function_p_div_r
    e = export_scalar(*simple_model, grid, kernel)


def test_hv_x0(simple_model, grid):
    e = hv_x0(*simple_model, grid, direction='x')


def test_export(simple_model, grid):
    simple_model[2].uni_degree = 1
    e = export(*simple_model, grid)


def test_export_drift2(simple_model, grid):
    simple_model[2].uni_degree = 2
    e = export(*simple_model, grid)