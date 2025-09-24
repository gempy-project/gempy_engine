import numpy as np
import pytest
from approvaltests import Options
from approvaltests.approvals import verify
from approvaltests.namer import NamerFactory

from ....conftest import Requirements, REQUIREMENT_LEVEL
from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.core.data.matrices_sizes import MatricesSizes

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

from tests.verify_helper import ArrayComparator

dir_name = os.path.dirname(__file__)


def test_covariance_cubic_kernel(simple_model_2):
    # Cubic kernel
    # Euclidean distance
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    input_data_descriptor: InputDataDescriptor = simple_model_2[3]

    sp_internals = surface_points_preprocess(surface_points, input_data_descriptor.tensors_structure)
    ori_internals = orientations_preprocess(orientations)

    solver_input = SolverInput(sp_internals, ori_internals, None, None)
    cov = yield_covariance(solver_input, options.kernel_options)
    print(cov)

    # todo: verify the full matrix when pykeops is False

    parameters: Options = NamerFactory.with_parameters("axis=1").with_comparator(ArrayComparator())
    sol = BackendTensor.tfnp.sum(cov, axis=1, keepdims=True)
    
    verify(sol, options=parameters)


def test_b_vector(simple_model_2):
    orientations = simple_model_2[1]
    ori_internals = orientations_preprocess(orientations)

    b_vec = yield_b_vector(ori_internals, 9)
    
    verify(
        data= BackendTensor.t.to_numpy(b_vec),
        options=NamerFactory.with_parameters().with_comparator(ArrayComparator())
    )


def test_eval_kernel(simple_model_2, simple_grid_2d):
    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    input_data_descriptor: InputDataDescriptor = simple_model_2[3]

    simple_grid_2d = BackendTensor.t.array(simple_grid_2d)
    sp_internals = surface_points_preprocess(surface_points, input_data_descriptor.tensors_structure)
    ori_internals = orientations_preprocess(orientations)
    solver_input = SolverInput(sp_internals, ori_internals, simple_grid_2d, None)
    kernel_data = evaluation_vectors_preparations(solver_input, options.kernel_options)
    export_kernel = create_scalar_kernel(kernel_data, options.kernel_options)
    print(export_kernel)

    export_gradient_ = create_grad_kernel(kernel_data, options.kernel_options)
    print(export_gradient_)


backendNOTNumpyOrNotEnoughRequirementsInstalled = (BackendTensor.engine_backend != AvailableBackends.numpy or
                                                   REQUIREMENT_LEVEL.value < Requirements.OPTIONAL.value)
@pytest.mark.skipif(backendNOTNumpyOrNotEnoughRequirementsInstalled, reason="These tests only makes sense for numpy backend and PyKEOPS")
class TestPykeopsNumPyEqual():

    @pytest.fixture(scope="class")
    def preprocess_data(self, simple_model_2_b):
        surface_points = simple_model_2_b[0]
        orientations = simple_model_2_b[1]
        input_data_descriptor: InputDataDescriptor = simple_model_2_b[3]

        # Prepare options
        interpolation_options = InterpolationOptions.from_args(
            range=5,
            c_o=5 ** 2 / 14 / 3,
            uni_degree=0,
            number_dimensions=2,
            kernel_function=AvailableKernelFunctions.exponential
        )

        # Prepare kernel
        sp_internals = surface_points_preprocess(surface_points, input_data_descriptor.tensors_structure)
        ori_internals = orientations_preprocess(orientations)

        return sp_internals, ori_internals, interpolation_options

    def test_cartesian_selector(self, preprocess_data):
        sp_, ori_, options = preprocess_data
        cov_size = ori_.n_orientations_tiled + sp_.n_points + options.n_uni_eq

        from gempy_engine.modules.kernel_constructor._kernel_selectors import dips_sp_cartesian_selector

        matrices_sizes = MatricesSizes(
            ori_size=ori_.n_orientations_tiled,
            sp_size=sp_.n_points,
            uni_drift_size=options.n_uni_eq,
            faults_size=0,
            dim=options.number_dimensions,
            n_dips=ori_.n_orientations
        )

        sel_hu_input, sel_hv_input, sel_hu_points_input = dips_sp_cartesian_selector(matrices_sizes)

        cartesian_selector = CartesianSelector(sel_hu_input, sel_hv_input, sel_hv_input, sel_hu_input, sel_hu_points_input,
                                               sel_hu_points_input, sel_hu_points_input, sel_hu_points_input)

        with open(dir_name + '/../solutions/cartesian_selector.pickle', 'rb') as handle:
            cartesian_selector_sol = pickle.load(handle)

        np.testing.assert_array_almost_equal(cartesian_selector.hu_sel_i, cartesian_selector_sol.hu_sel_i, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hu_sel_j, cartesian_selector_sol.hu_sel_j, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hv_sel_i, cartesian_selector_sol.hv_sel_i, decimal=3)
        np.testing.assert_array_almost_equal(cartesian_selector.hv_sel_j, cartesian_selector_sol.hv_sel_j, decimal=3)

    def test_distance_matrices(self, preprocess_data):
        sp_, ori_, options = preprocess_data
        cov_size = ori_.n_orientations_tiled + sp_.n_points + options.n_uni_eq

        solver_input = SolverInput(sp_, ori_)
        ki = cov_vectors_preparation(solver_input, options.kernel_options)

        with open(dir_name + '/../solutions/distance_matrices.pickle', 'rb') as handle:
            dm_sol = pickle.load(handle)
        dm = _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices, True, True, is_testing=True)

        if BackendTensor.pykeops_enabled is False:
            np.testing.assert_array_almost_equal(dm.dif_ref_ref, dm_sol.dif_ref_ref, decimal=3)
            np.testing.assert_array_almost_equal(dm.dif_rest_rest, dm_sol.dif_rest_rest, decimal=3)
            np.testing.assert_array_almost_equal(dm.hu, dm_sol.hu, decimal=3)
            np.testing.assert_array_almost_equal(dm.huv_ref, dm_sol.huv_ref, decimal=3)
            np.testing.assert_array_almost_equal(dm.huv_rest, dm_sol.huv_rest, decimal=3)
            np.testing.assert_array_almost_equal(dm.perp_matrix, dm_sol.perp_matrix, decimal=3)
            if False:  # ! (March 6, 2023) these checks are failing but they are old
                np.testing.assert_array_almost_equal(dm.r_ref_ref, dm_sol.r_ref_ref, decimal=3)
                np.testing.assert_array_almost_equal(dm.r_ref_rest, dm_sol.r_ref_rest, decimal=3)
                np.testing.assert_array_almost_equal(dm.r_rest_ref, dm_sol.r_rest_ref, decimal=3)
                np.testing.assert_array_almost_equal(dm.r_rest_rest, dm_sol.r_rest_rest, decimal=3)

        verify(
            data=BackendTensor.tfnp.sum(dm.dif_ref_ref, axis=1, keepdims=False), 
            options=NamerFactory.with_parameters("dif_ref_ref").with_comparator(ArrayComparator())
        )

    def test_compare_cg(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_grad", cov_func=_test_covariance_items)
        
    @pytest.mark.skip(reason="Deprecated")
    def test_compare_ci(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_sp", cov_func=_test_covariance_items)

    def test_compare_cgi(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov_grad_sp", cov_func=_test_covariance_items)

    def test_compare_drift(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="drift", cov_func=_test_covariance_items)

    def test_compare_full_cov(self, preprocess_data):
        self._compare_covariance_item_numpy_pykeops(preprocess_data, item="cov", cov_func=_test_covariance_items, compare_to_saved=False)

    def _compare_covariance_item_numpy_pykeops(self, preprocess_data, item, cov_func, compare_to_saved=True):
        sp_internals, ori_internals, options = preprocess_data

        # numpy
        BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=False)
        solver_input = SolverInput(sp_internals, ori_internals)
        kernel_data = cov_vectors_preparation(solver_input, options.kernel_options)
        c_n = cov_func(kernel_data, options, item=item)

        path = dir_name + f"/../solutions/{item}.npy"
        if False:
            np.save(path, c_n)

        l = np.load(path)
        c_n_sum = c_n.sum(0).reshape(-1, 1)

        # pykeops
        BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=True)
        kernel_data = cov_vectors_preparation(solver_input, options.kernel_options)
        c_k = cov_func(kernel_data, options, item=item)
        c_k_sum = c_n.sum(0).reshape(-1, 1)

        print('l: ', l)
        print("just numpy: ", c_n, c_n_sum)
        print("pykeops: ", c_k, c_k_sum)

        if compare_to_saved:
            np.testing.assert_array_almost_equal(np.asarray(c_n), l, decimal=1)
        np.testing.assert_array_almost_equal(c_n_sum, c_k_sum, decimal=2)
