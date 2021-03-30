import numpy as np
import pytest

from gempy_engine.config import BackendTensor, AvailableBackends
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.modules.kernel_constructor._covariance_assembler import _test_covariance_items, create_covariance
from gempy_engine.modules.kernel_constructor._input_preparation import surface_points_preprocess, \
    orientations_preprocess
from gempy_engine.modules.kernel_constructor._vectors_preparation import _vectors_preparation
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance


def test_covariance_cubic_kernel(simple_model_2):
    # Cubic kernel
    # Euclidean distance

    l = np.load('test_kernel_numeric2.npy')
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    tensors_structure = simple_model_2[3]
    sp_internals = surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
    ori_internals = orientations_preprocess(orientations)

    cov = yield_covariance(sp_internals, ori_internals, options)
    print(cov)
    print(l)

    np.testing.assert_array_almost_equal(np.asarray(cov), l, decimal=3)


# TODO: By default we are not testing if the graph works with tf.function
def test_covariance_spline_kernel(simple_model_2):
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    tensors_structure = simple_model_2[3]

    options.kernel_function = AvailableKernelFunctions.exponential

    sp_internals = surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
    ori_internals = orientations_preprocess(orientations)

    cov = yield_covariance(sp_internals, ori_internals, options)
    cov_sum = cov.sum(axis=1).reshape(-1, 1)
    print(cov_sum)
    return cov_sum


class TestPykeopsNumPyEqual():

    @pytest.fixture(scope="class")
    def preprocess_data(self, simple_model_2):
        surface_points = simple_model_2[0]
        orientations = simple_model_2[1]
        options = simple_model_2[2]
        tensors_structure = simple_model_2[3]

        # Prepare kernel
        sp_internals = surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
        ori_internals = orientations_preprocess(orientations)

        # Prepare options
        options.kernel_function = AvailableKernelFunctions.exponential

        return sp_internals, ori_internals, options

    def test_compare_cg(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_grad", cov_func = _test_covariance_items)

    def test_compare_ci(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_sp", cov_func = _test_covariance_items)

    def test_compare_cgi(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_grad_sp", cov_func = _test_covariance_items)

    def test_compare_drift(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="drift", cov_func = _test_covariance_items)

    def test_copare_full_cov(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov", cov_func = create_covariance)

    def _compare_covariance_item_numpy_pykeops(self, preprocess_data, item, cov_func):
        sp_internals, ori_internals, options = preprocess_data
        # numpy
        BackendTensor.change_backend(AvailableBackends.numpy, pykeops_enabled=False)
        kernel_data = _vectors_preparation(sp_internals, ori_internals, options)
        c_n = cov_func(kernel_data, options, item=item)
        c_n_sum = c_n.sum(0).reshape(-1, 1)
        print(c_n, c_n_sum)

        # pykeops
        BackendTensor.change_backend(AvailableBackends.numpy, pykeops_enabled=True)
        kernel_data = _vectors_preparation(sp_internals, ori_internals, options)
        c_k = cov_func(kernel_data, options, item=item)
        c_k_sum = c_n.sum(0).reshape(-1, 1)
        print(c_k, c_k_sum)
        np.testing.assert_array_almost_equal(c_n_sum, c_k_sum, decimal=3)
