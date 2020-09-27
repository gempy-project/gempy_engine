import pytest
import numpy as np

from gempy_engine.data_structures.private_structures import SurfacePointsInternals, OrientationsInternals
from gempy_engine.data_structures.public_structures import OrientationsInput, InterpolationOptions
from gempy_engine.systems.generators import tile_dip_positions
from gempy_engine.systems.kernel.kernel import vectors_preparation, create_covariance
from gempy_engine.systems.kernel.kernel_functions import *


@pytest.fixture()
def simple_model():
    spi = SurfacePointsInternals(
        ref_surface_points=np.array(
            [[4, 0],
             [4, 0],
             [4, 0],
             [3, 3],
             [3, 3]]),
        rest_surface_points=np.array([[0, 0],
                                      [2, 0],
                                      [3, 0],
                                      [0, 2],
                                      [2, 2]]),
        nugget_effect_ref_rest=0
    )

    ori_i = OrientationsInput(
        dip_positions=np.array([[0, 6],
                                [2, 13]]),
        nugget_effect_grad=0.0000001
    )
    dip_tiled = tile_dip_positions(ori_i.dip_positions, 2)
    ori_int = OrientationsInternals(dip_tiled)
    kri = InterpolationOptions(5, 5 ** 2 / 14 / 3, 0, i_res=1, gi_res=1,
                               number_dimensions=2)

    return spi, ori_int, kri


def test_vector_preparation(simple_model):
    ki = vectors_preparation(*simple_model)
    print(ki)


def test_kernel_numeric(simple_model):
    ki = vectors_preparation(*simple_model)
    c = create_covariance(ki, simple_model[2].range, simple_model[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)
    np.set_printoptions(precision=2, linewidth=150)
    print('\n', c)


def test_kernel_numeric_pykeops_cubic(simple_model):
    ki = vectors_preparation(*simple_model, backend='pykeops')
    c = create_covariance(ki, simple_model[2].range, simple_model[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)
    np.set_printoptions(precision=2, linewidth=150)
    print('\n', c)

    b = np.zeros((9, 1), dtype='float32')
    b[:4, :] = 1

    w = c.solve(b, alpha=0.01, dtype_acc='float32')
    print('\n', w)


def test_kernel_numeric_pykeops_cubic_heavy(moureze_internals):
    moureze_internals[2].uni_degree = 0

    ki = vectors_preparation(*moureze_internals, backend='pykeops')
    c = create_covariance(ki, moureze_internals[2].range, moureze_internals[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)
    np.set_printoptions(precision=2, linewidth=150)
    print('\n', c)

    b = np.zeros((moureze_internals[1].n_orientations_tiled+moureze_internals[0].n_points, 1),
                 dtype='float32')
    b[:moureze_internals[1].n_orientations_tiled, :] = 1
    c.backend = 'GPU'
    w = c.solve(b, alpha=10, dtype_acc='float32')
    print('\n', w)


def test_kernel_numeric_pykeops_gaussian(simple_model):
    ki = vectors_preparation(*simple_model, backend='pykeops')
    c = create_covariance(ki, simple_model[2].range, simple_model[2].c_o,
                          exp_function, exp_function_p_div_r,
                          exp_function_a)
    np.set_printoptions(precision=2, linewidth=150)
    print('\n', c)


