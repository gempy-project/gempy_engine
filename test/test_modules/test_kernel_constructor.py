import numpy as np
import pytest

from gempy_engine.config import BackendTensor, AvailableBackends
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.modules.kernel_constructor._covariance_assembler import _test_covariance_items, create_kernel
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, \
    orientations_preprocess
from gempy_engine.modules.kernel_constructor._vectors_preparation import cov_vectors_preparation
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, yield_b_vector


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

# TODO:
def test_b_vector(simple_model_2):
    orientations = simple_model_2[1]
    ori_internals = orientations_preprocess(orientations)

    b_vec = yield_b_vector(ori_internals, 9)
    print(b_vec)

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


# TODO: (bug) When running test_covariance_spline_kernel the running the class test breaks for some weird state change
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

    def test_cartesian_selector(self, preprocess_data):
        sp_, ori_, options = preprocess_data
        cov_size = ori_.n_orientations_tiled + sp_.n_points + options.n_uni_eq

        from gempy_engine.modules.kernel_constructor._kernel_selectors import dips_sp_cartesian_selector
        from gempy_engine.modules.kernel_constructor._kernel_selectors import hu_hv_sel

        sel_hu_input, sel_hv_input, sel_hu_points_input = dips_sp_cartesian_selector(cov_size,
                                                                                     options.number_dimensions,
                                                                                     ori_.n_orientations, sp_.n_points)

        cartesian_selector = hu_hv_sel(sel_hu_input, sel_hv_input,
                                       sel_hv_input, sel_hu_input,
                                       sel_hu_points_input, sel_hu_points_input)
        import pickle
        with open('solutions/cartesian_selector.pickle', 'rb') as handle:
            cartesian_selector_sol = pickle.load(handle)

        np.testing.assert_array_almost_equal(cartesian_selector.hu_sel_i,        cartesian_selector_sol.hu_sel_i, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hu_sel_j,        cartesian_selector_sol.hu_sel_j, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hv_sel_i,        cartesian_selector_sol.hv_sel_i, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hv_sel_j,        cartesian_selector_sol.hv_sel_j, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hu_sel_points_i, cartesian_selector_sol.hv_sel_points_i, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hu_sel_points_j, cartesian_selector_sol.hu_sel_points_j, decimal=3)


    def test_compare_cg(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_grad", cov_func = _test_covariance_items)

    def test_compare_ci(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_sp", cov_func = _test_covariance_items)

    def test_compare_cgi(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_grad_sp", cov_func = _test_covariance_items)

    def test_compare_drift(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="drift", cov_func = _test_covariance_items)

    def test_copare_full_cov(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="all", cov_func = _test_covariance_items)

    def _compare_covariance_item_numpy_pykeops(self, preprocess_data, item, cov_func):
        sp_internals, ori_internals, options = preprocess_data
        # numpy
        BackendTensor.change_backend(AvailableBackends.numpy, pykeops_enabled=False)
        kernel_data = cov_vectors_preparation(sp_internals, ori_internals, options)
        c_n = cov_func(kernel_data, options, item=item)
        if False:
            np.save(f"./solutions/{item}", c_n)
        l =  np.load(f"./solutions/{item}.npy")
        c_n_sum = c_n.sum(0).reshape(-1, 1)

        print(c_n, c_n_sum)
        np.testing.assert_array_almost_equal(c_n, l, decimal=3)


        # pykeops
        BackendTensor.change_backend(AvailableBackends.numpy, pykeops_enabled=True)
        kernel_data = cov_vectors_preparation(sp_internals, ori_internals, options)
        c_k = cov_func(kernel_data, options, item=item)
        c_k_sum = c_n.sum(0).reshape(-1, 1)
        print(c_k, c_k_sum)
        np.testing.assert_array_almost_equal(c_n_sum, c_k_sum, decimal=3)
