import pytest
import numpy as np
import gempy_engine.systems.kernel.kernel
from gempy_engine.data_structures.private_structures import SurfacePointsInternals, OrientationsInternals
from gempy_engine.data_structures.public_structures import OrientationsInput, InterpolationOptions
from gempy_engine.systems.generators import tile_dip_positions
from gempy_engine.systems.kernel.kernel import vectors_preparation, create_covariance, kernel_solver
from gempy_engine.systems.kernel.kernel_functions import *
import copy


def test_kernel_numeric_numpy_cubic(simple_model, save_result=False):
    # Test numpy
    gempy_engine.systems.kernel.kernel.pykeops_imported = False
    simple_model_ = copy.deepcopy(simple_model)

    simple_model_[2].uni_degree = 1
    w = kernel_solver(*simple_model_)
    if save_result:
        np.save('./test_kernel_numeric_weights', w)
    print(w)

    # Test TF:
    simple_model_[2].uni_degree = 1
    w = kernel_solver(*simple_model_)
    print(w)

    # Test pykeops
    gempy_engine.systems.kernel.kernel.pykeops_imported = True
    simple_model_[2].uni_degree = 1
    w = kernel_solver(*simple_model_)
    print(w)


def test_vector_preparation(simple_model):
    simple_model_ = copy.deepcopy(simple_model)
    gempy_engine.systems.kernel.kernel.pykeops_imported = True
    ki = vectors_preparation(*simple_model_)
    print(ki)

    gempy_engine.systems.kernel.kernel.pykeops_imported = False
    ki = vectors_preparation(*simple_model_)
    print(ki)


def test_kernel_numeric_jax(simple_model, save_result=False):
    simple_model_ = copy.deepcopy(simple_model)
    gempy_engine.systems.kernel.kernel.pykeops_imported = False
    ki = vectors_preparation(*simple_model_, backend='numpy')
    c = create_covariance(ki, simple_model_[2].range, simple_model_[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)

    np.set_printoptions(precision=2, linewidth=150)
    if save_result:
        np.save('./test_kernel_numeric2', c)
    l = np.load('test_kernel_numeric2.npy')
    np.testing.assert_array_almost_equal(np.asarray(c), l, decimal=3)


def test_kernel_numeric(simple_model, save_result=False):
    gempy_engine.systems.kernel.kernel.pykeops_imported = False
    simple_model_ = copy.deepcopy(simple_model)
    ki = vectors_preparation(*simple_model_, backend='numpy')
    c = create_covariance(ki, simple_model_[2].range, simple_model_[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)

    np.set_printoptions(precision=2, linewidth=150)
    if save_result:
        np.save('./test_kernel_numeric2', c)
    l = np.load('test_kernel_numeric2.npy')
    np.testing.assert_array_almost_equal(np.asarray(c), l, decimal=3)

    # ============================== PYKEOPS ==================
    gempy_engine.systems.kernel.kernel.pykeops_imported = True
    ki = vectors_preparation(*simple_model_, backend='pykeops')
    c = create_covariance(ki, simple_model_[2].range, simple_model_[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)
    np.testing.assert_array_almost_equal(
        c.sum_reduction(axis=0).reshape(-1),
        np.load('test_kernel_numeric2.npy').sum(axis=0), decimal=3)
    print('\n', c)


def test_kernel_numeric_uni(simple_model):
    simple_model_ = copy.deepcopy(simple_model)
    simple_model_[2].uni_degree = 1

    gempy_engine.systems.kernel.kernel.pykeops_imported = False
    ki = vectors_preparation(*simple_model_, backend='numpy')
    c = create_covariance(ki, simple_model_[2].range, simple_model_[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)
    print('\n', c)

    # ============================== PYKEOPS ==================
    gempy_engine.systems.kernel.kernel.pykeops_imported = True
    ki = vectors_preparation(*simple_model_, backend='pykeops')
    c = create_covariance(ki, simple_model_[2].range, simple_model_[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)

    print('\n', c.sum_reduction(0))


def test_kernel_numeric_pykeops_cubic(simple_model):
    simple_model_ = copy.deepcopy(simple_model)
    gempy_engine.systems.kernel.kernel.pykeops_imported = True
    ki = vectors_preparation(*simple_model_, backend='pykeops')
    c = create_covariance(ki, simple_model_[2].range, simple_model_[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)
    np.set_printoptions(precision=2, linewidth=150)
    print('\n', c)

    b = np.zeros((9, 1), dtype='float32')
    b[:4, :] = 1

    w = c.solve(b, alpha=0.01, dtype_acc='float32')
    print('\n', w)


def test_kernel_numeric_numpy_cubic2(simple_model):
    simple_model_ = copy.deepcopy(simple_model)
    gempy_engine.systems.kernel.kernel.pykeops_imported = False
    ki = vectors_preparation(*simple_model_, backend='numpy')
    c = create_covariance(ki, simple_model_[2].range, simple_model_[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)
    np.set_printoptions(precision=2, linewidth=150)
    print('\n', c)

    b = np.zeros((9, 1), dtype='float32')
    b[:4, :] = 1

    w = np.linalg.solve(c, b)
    # w = c.solve(b, alpha=0.01, dtype_acc='float32')
    print('\n', w)


def test_kernel_numeric_pykeops_cubic_heavy(moureze_internals):
    gempy_engine.systems.kernel.kernel.pykeops_imported = True
    moureze_internals_ = copy.deepcopy(moureze_internals)
    moureze_internals_[2].uni_degree = 0
    ki = vectors_preparation(*moureze_internals_, backend='pykeops')
    c = create_covariance(ki, moureze_internals_[2].range, moureze_internals_[2].c_o,
                          cubic_function, cubic_function_p_div_r,
                          cubic_function_a)
    np.set_printoptions(precision=2, linewidth=150)
    print('\n', c)

    b = np.zeros((moureze_internals_[1].n_orientations_tiled+moureze_internals_[0].n_points, 1),
                 dtype='float32')
    b[:moureze_internals_[1].n_orientations_tiled, :] = 1
    c.backend = 'GPU'
    w = c.solve(b, alpha=10, dtype_acc='float32')
    print('\n', w)


def test_kernel_numeric_pykeops_gaussian(simple_model):
    gempy_engine.systems.kernel.kernel.pykeops_imported = True
    simple_model_ = copy.deepcopy(simple_model)
    ki = vectors_preparation(*simple_model_, backend='pykeops')
    c = create_covariance(ki, simple_model_[2].range, simple_model_[2].c_o,
                          exp_function, exp_function_p_div_r,
                          exp_function_a)
    np.set_printoptions(precision=2, linewidth=150)
    print('\n', c)


def test_kernel_numeric_pykeops_gaussian_drift(simple_model):
    simple_model_ = copy.deepcopy(simple_model)
    gempy_engine.systems.kernel.kernel.pykeops_imported = True
    simple_model_[2].uni_degree = 1
    ki = vectors_preparation(*simple_model_, backend='pykeops')
    c = create_covariance(ki, simple_model_[2].range, simple_model_[2].c_o,
                          exp_function, exp_function_p_div_r,
                          exp_function_a)
    np.set_printoptions(precision=2, linewidth=150)
    print('\n', c)
    b = np.zeros((11, 1), dtype='float32')
    b[:4, :] = 1
    w = c.solve(b, alpha=10, dtype_acc='float32')
    print('\n', w)


