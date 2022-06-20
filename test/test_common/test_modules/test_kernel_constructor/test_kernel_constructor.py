import numpy as np
import pytest

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.modules.kernel_constructor._kernels_assembler import _compute_all_distance_matrices, create_scalar_kernel, create_grad_kernel
from gempy_engine.modules.kernel_constructor._test_assembler import _test_covariance_items
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, \
    orientations_preprocess
from gempy_engine.modules.kernel_constructor._structs import CartesianSelector
from gempy_engine.modules.kernel_constructor._vectors_preparation import cov_vectors_preparation, \
    evaluation_vectors_preparations
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, yield_b_vector

import pickle
import os
dir_name = os.path.dirname(__file__)


def test_covariance_cubic_kernel(simple_model_2):
    # Cubic kernel
    # Euclidean distance

    l = np.load(dir_name + '/../solutions/test_kernel_numeric2.npy')
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    input_data_descriptor: InputDataDescriptor = simple_model_2[3]

    options.i_res = 1
    options.gi_res = 1

    sp_internals = surface_points_preprocess(surface_points, input_data_descriptor.tensors_structure)
    ori_internals = orientations_preprocess(orientations)

    cov = yield_covariance(SolverInput(sp_internals, ori_internals, options.kernel_options))
    print(cov)
    print(l)
    np.save(dir_name + '/../solutions/test_kernel_numeric2.npy', cov)

    np.testing.assert_array_almost_equal(np.asarray(cov), l, decimal=3)


def test_b_vector(simple_model_2):
    orientations = simple_model_2[1]
    ori_internals = orientations_preprocess(orientations)

    b_vec = yield_b_vector(ori_internals, 9)
    print(b_vec)


def test_eval_kernel(simple_model_2, simple_grid_2d):
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    input_data_descriptor: InputDataDescriptor = simple_model_2[3]

    sp_internals = surface_points_preprocess(surface_points, input_data_descriptor.tensors_structure)
    ori_internals = orientations_preprocess(orientations)
    kernel_data = evaluation_vectors_preparations(simple_grid_2d, SolverInput(sp_internals, ori_internals, options))
    export_kernel = create_scalar_kernel(kernel_data, options)
    print(export_kernel)

    export_gradient_ = create_grad_kernel(kernel_data, options)
    print(export_gradient_)


pykeops_enabled = True


# TODO: (bug) When running test_covariance_spline_kernel the running the class test breaks for some weird state change
class TestPykeopsNumPyEqual():
    
    @pytest.fixture(scope="class")
    def preprocess_data(self, simple_model_2):
        surface_points = simple_model_2[0]
        orientations = simple_model_2[1]
        options = simple_model_2[2]
        input_data_descriptor: InputDataDescriptor = simple_model_2[3]
        # Prepare options
        options.kernel_function = AvailableKernelFunctions.exponential

        # Prepare kernel
        sp_internals = surface_points_preprocess(surface_points, input_data_descriptor.tensors_structure)
        ori_internals = orientations_preprocess(orientations)

        return sp_internals, ori_internals, options

    def test_cartesian_selector(self, preprocess_data):
        sp_, ori_, options = preprocess_data
        cov_size = ori_.n_orientations_tiled + sp_.n_points + options.n_uni_eq

        from gempy_engine.modules.kernel_constructor._kernel_selectors import dips_sp_cartesian_selector

        sel_hu_input, sel_hv_input, sel_hu_points_input = dips_sp_cartesian_selector(cov_size,
                                                                                     options.number_dimensions,
                                                                                     ori_.n_orientations, sp_.n_points)

        cartesian_selector = CartesianSelector(sel_hu_input, sel_hv_input, sel_hv_input, sel_hu_input, sel_hu_points_input,
                                       sel_hu_points_input, sel_hu_points_input, sel_hu_points_input)

        with open(dir_name + '/../solutions/cartesian_selector.pickle', 'rb') as handle:
            cartesian_selector_sol = pickle.load(handle)

        np.testing.assert_array_almost_equal(cartesian_selector.hu_sel_i,        cartesian_selector_sol.hu_sel_i, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hu_sel_j,        cartesian_selector_sol.hu_sel_j, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hv_sel_i,        cartesian_selector_sol.hv_sel_i, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hv_sel_j,        cartesian_selector_sol.hv_sel_j, decimal=3)

    def test_distance_matrices(self, preprocess_data):
        sp_, ori_, options = preprocess_data
        cov_size = ori_.n_orientations_tiled + sp_.n_points + options.n_uni_eq

        ki = cov_vectors_preparation(SolverInput(sp_, ori_, options.kernel_options))

        with open(dir_name + '/../solutions/distance_matrices.pickle', 'rb') as handle:
            dm_sol = pickle.load(handle)
        dm = _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices)

        np.testing.assert_array_almost_equal(dm.dif_ref_ref , dm_sol.dif_ref_ref, decimal=3)
        np.testing.assert_array_almost_equal(dm.dif_rest_rest, dm_sol.dif_rest_rest, decimal=3)
        np.testing.assert_array_almost_equal(dm.hu, dm_sol.hu, decimal=3)
        np.testing.assert_array_almost_equal(dm.huv_ref, dm_sol.huv_ref, decimal=3)
        np.testing.assert_array_almost_equal(dm.huv_rest, dm_sol.huv_rest, decimal=3)
        np.testing.assert_array_almost_equal(dm.perp_matrix, dm_sol.perp_matrix, decimal=3)
        np.testing.assert_array_almost_equal(dm.r_ref_ref, dm_sol.r_ref_ref, decimal=3)
        np.testing.assert_array_almost_equal(dm.r_ref_rest, dm_sol.r_ref_rest, decimal=3)
        np.testing.assert_array_almost_equal(dm.r_rest_ref, dm_sol.r_rest_ref, decimal=3)
        np.testing.assert_array_almost_equal(dm.r_rest_rest, dm_sol.r_rest_rest, decimal=3)


    def test_compare_cg(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_grad", cov_func = _test_covariance_items)

    def test_compare_ci(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_sp", cov_func = _test_covariance_items)

    def test_compare_cgi(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_grad_sp", cov_func = _test_covariance_items)

    def test_compare_drift(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="drift", cov_func = _test_covariance_items)

    @pytest.mark.skip("This test is broken: the stored covariance has a different c_o")
    def test_copare_full_cov(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov", cov_func = _test_covariance_items)

    def _compare_covariance_item_numpy_pykeops(self, preprocess_data, item, cov_func):
        sp_internals, ori_internals, options = preprocess_data
        
        # numpy
        BackendTensor.change_backend(AvailableBackends.numpy, pykeops_enabled=False)
        kernel_data = cov_vectors_preparation(SolverInput(sp_internals, ori_internals, options))
        c_n = cov_func(kernel_data, options, item=item)
        if False:
            np.save(f"./solutions/{item}", c_n)

        l =  np.load(dir_name + f"/../solutions/{item}.npy")
        c_n_sum = c_n.sum(0).reshape(-1, 1)

        print(c_n, c_n_sum)
        np.testing.assert_array_almost_equal(np.asarray(c_n), l, decimal=3)


        # pykeops
        BackendTensor.change_backend(AvailableBackends.numpy, pykeops_enabled=pykeops_enabled)
        kernel_data = cov_vectors_preparation(SolverInput(sp_internals, ori_internals, options))
        c_k = cov_func(kernel_data, options, item=item)
        c_k_sum = c_n.sum(0).reshape(-1, 1)
        print(c_k, c_k_sum)
        np.testing.assert_array_almost_equal(c_n_sum, c_k_sum, decimal=3)
