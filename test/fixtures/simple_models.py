import logging

import numpy as np
import pytest

from gempy_engine.config import BackendTensor
from gempy_engine.core.data.data_shape import TensorsStructure
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess


@pytest.fixture(scope='session')
def simple_model_2():
    print(BackendTensor.describe_conf())

    sp_coords = np.array([[4, 0],
                          [0, 0],
                          [2, 0],
                          [3, 0],
                          [3, 3],
                          [0, 2],
                          [2, 2]])

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp_coords, nugget_effect_scalar)

    dip_positions = np.array([[0, 6],
                              [2, 13]])

    dip_gradients = np.array([[0, 1], [0, .8]])

    nugget_effect_grad = 0.0000001
    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
                               number_dimensions=2, kernel_function=AvailableKernelFunctions.cubic)

    _ = np.ones(3)
    tensor_structure = TensorsStructure(np.array([4, 3]), _, _, _, _)

    return spi, ori_i, kri, tensor_structure

@pytest.fixture(scope="session")
def simple_model_2_internals(simple_model_2):
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    tensors_structure = simple_model_2[3]

    sp_internals = surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
    ori_internals = orientations_preprocess(orientations)
    return sp_internals, ori_internals, options

@pytest.fixture(scope='session')
def simple_grid():
    nx, ny = (3, 2)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    g = np.vstack((xv.ravel(), yv.ravel())).T
    return g


def test_simple_model_gempy_engine():
    g = gempy.create_data("test_engine", extent=[-2, 2, -2, 2, -2, 2], resolution=[2, 2, 2])
    sp = np.array([[4, 0, 0],
                   [0, 0, 0],
                   [2, 0, 0],
                   [3, 0, 0],
                   [3, 0, 3],
                   [0, 0, 2],
                   [2, 0, 2]])

    g.set_default_surfaces()

    for i in sp:
        g.add_surface_points(*i, surface="surface1")

    g.add_orientations(0, 0, 6, pole_vector=(0, 0, 1), surface="surface1")
    g.add_orientations(2, 0, 13, pole_vector=(0, 0, .8), surface="surface1")

    g.modify_kriging_parameters("range", 5)
    g.modify_kriging_parameters("$C_o$", 5 ** 2 / 14 / 3)

    gempy.set_interpolator(g, verbose=["covariance_matrix"])

    print(g.solutions.scalar_field_matrix)