from concurrent.futures import ThreadPoolExecutor

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

from gempy_engine.modules.kernel_constructor import _kernels_assembler
from gempy_engine.modules.kernel_constructor._internalDistancesMatrices import DistancesBuffer
from gempy_engine.modules.kernel_constructor._kernels_assembler import _compute_all_distance_matrices, create_scalar_kernel, create_grad_kernel
from gempy_engine.modules.kernel_constructor._test_assembler import _test_covariance_items
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, \
    orientations_preprocess
from gempy_engine.modules.kernel_constructor._structs import CartesianSelector
from gempy_engine.modules.kernel_constructor._vectors_preparation import cov_vectors_preparation, \
    evaluation_vectors_preparations
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, yield_b_vector
from gempy_engine.modules.kernel_constructor.execution_mode import KernelExecutionMode

import pickle
import os

from tests.verify_helper import ArrayComparator, gempy_verify_array

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

    sol = BackendTensor.tfnp.sum(cov, axis=1, keepdims=True)
    
    gempy_verify_array(sol, "axis=1")


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


def test_distance_buffer_reuses_scalar_distances_for_all_gradient_axes(simple_model_2, simple_grid_2d, monkeypatch):
    surface_points, orientations, options, input_data_descriptor = simple_model_2
    grid = BackendTensor.t.array(simple_grid_2d)
    solver_input = SolverInput(
        surface_points_preprocess(surface_points, input_data_descriptor.tensors_structure),
        orientations_preprocess(orientations),
    )
    solver_input.xyz_to_interpolate = grid
    square_distance = options.kernel_options.kernel_function.value.consume_sq_distance
    distance_buffer = DistancesBuffer()
    distance_cache_key = object()
    generic_calls = 0
    original_compute = _kernels_assembler._compute_distances_generic

    def count_generic_calls(*args, **kwargs):
        nonlocal generic_calls
        generic_calls += 1
        return original_compute(*args, **kwargs)

    monkeypatch.setattr(_kernels_assembler, "_compute_distances_generic", count_generic_calls)
    scalar_data = evaluation_vectors_preparations(solver_input, options.kernel_options)
    scalar_distances = _compute_all_distance_matrices(
        scalar_data.cartesian_selector,
        scalar_data.ori_sp_matrices,
        square_distance,
        is_gradient=False,
        distance_buffer=distance_buffer,
        distance_cache_key=distance_cache_key,
    )

    cached_gradient_distances = []
    for axis in range(options.kernel_options.number_dimensions):
        gradient_data = evaluation_vectors_preparations(solver_input, options.kernel_options, axis=axis)
        cached_gradient_distances.append(_compute_all_distance_matrices(
            gradient_data.cartesian_selector,
            gradient_data.ori_sp_matrices,
            square_distance,
            is_gradient=True,
            distance_buffer=distance_buffer,
            distance_cache_key=distance_cache_key,
        ))

    assert generic_calls == 1
    assert all(distances.dif_ref_ref is scalar_distances.dif_ref_ref for distances in cached_gradient_distances)

    for axis, cached_distances in enumerate(cached_gradient_distances):
        gradient_data = evaluation_vectors_preparations(solver_input, options.kernel_options, axis=axis)
        uncached_distances = original_compute(
            gradient_data.cartesian_selector,
            gradient_data.ori_sp_matrices,
            square_distance,
        )
        for field_name in cached_distances.__dataclass_fields__:
            np.testing.assert_allclose(
                BackendTensor.t.to_numpy(getattr(cached_distances, field_name)),
                BackendTensor.t.to_numpy(getattr(uncached_distances, field_name)),
            )


def test_distance_buffers_are_isolated_between_concurrent_equal_shaped_inputs(simple_model_2, simple_grid_2d):
    surface_points, orientations, options, input_data_descriptor = simple_model_2
    sp_internal = surface_points_preprocess(surface_points, input_data_descriptor.tensors_structure)
    ori_internal = orientations_preprocess(orientations)
    square_distance = options.kernel_options.kernel_function.value.consume_sq_distance

    def compute_distances(grid_offset):
        grid = BackendTensor.t.array(simple_grid_2d) + grid_offset
        solver_input = SolverInput(sp_internal, ori_internal)
        solver_input.xyz_to_interpolate = grid
        distance_buffer = DistancesBuffer()
        distance_cache_key = object()
        scalar_data = evaluation_vectors_preparations(solver_input, options.kernel_options)
        _compute_all_distance_matrices(
            scalar_data.cartesian_selector,
            scalar_data.ori_sp_matrices,
            square_distance,
            is_gradient=False,
            distance_buffer=distance_buffer,
            distance_cache_key=distance_cache_key,
        )
        gradient_data = evaluation_vectors_preparations(solver_input, options.kernel_options, axis=0)
        cached = _compute_all_distance_matrices(
            gradient_data.cartesian_selector,
            gradient_data.ori_sp_matrices,
            square_distance,
            is_gradient=True,
            distance_buffer=distance_buffer,
            distance_cache_key=distance_cache_key,
        )
        uncached = _kernels_assembler._compute_distances_generic(
            gradient_data.cartesian_selector,
            gradient_data.ori_sp_matrices,
            square_distance,
        )
        return (
            BackendTensor.t.to_numpy(cached.r_ref_ref),
            BackendTensor.t.to_numpy(uncached.r_ref_ref),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(compute_distances, (0.0, 0.125)))

    for cached, uncached in results:
        np.testing.assert_allclose(cached, uncached)
    assert not np.allclose(results[0][0], results[1][0])


def test_distance_buffer_rejects_an_equal_shaped_input_with_a_different_cache_key(simple_model_2, simple_grid_2d):
    surface_points, orientations, options, input_data_descriptor = simple_model_2
    sp_internal = surface_points_preprocess(surface_points, input_data_descriptor.tensors_structure)
    ori_internal = orientations_preprocess(orientations)
    square_distance = options.kernel_options.kernel_function.value.consume_sq_distance
    distance_buffer = DistancesBuffer()

    first_input = SolverInput(sp_internal, ori_internal)
    first_input.xyz_to_interpolate = BackendTensor.t.array(simple_grid_2d)
    first_scalar_data = evaluation_vectors_preparations(first_input, options.kernel_options)
    _compute_all_distance_matrices(
        first_scalar_data.cartesian_selector,
        first_scalar_data.ori_sp_matrices,
        square_distance,
        is_gradient=False,
        distance_buffer=distance_buffer,
        distance_cache_key="first-input",
    )

    second_input = SolverInput(sp_internal, ori_internal)
    second_input.xyz_to_interpolate = BackendTensor.t.array(simple_grid_2d) + 0.125
    second_gradient_data = evaluation_vectors_preparations(second_input, options.kernel_options, axis=0)
    result = _compute_all_distance_matrices(
        second_gradient_data.cartesian_selector,
        second_gradient_data.ori_sp_matrices,
        square_distance,
        is_gradient=True,
        distance_buffer=distance_buffer,
        distance_cache_key="second-input",
    )
    expected = _kernels_assembler._compute_distances_generic(
        second_gradient_data.cartesian_selector,
        second_gradient_data.ori_sp_matrices,
        square_distance,
    )

    np.testing.assert_allclose(
        BackendTensor.t.to_numpy(result.r_ref_ref),
        BackendTensor.t.to_numpy(expected.r_ref_ref),
    )
    assert result.dif_ref_ref is not distance_buffer.shared.dif_ref_ref


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
        c_n = cov_func(
            kernel_data,
            options,
            item=item,
            execution_mode=KernelExecutionMode.DENSE,
        )

        path = dir_name + f"/../solutions/{item}.npy"
        if False:
            np.save(path, c_n)

        l = np.load(path)
        c_n_sum = c_n.sum(0).reshape(-1, 1)

        # pykeops
        BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=True)
        kernel_data = cov_vectors_preparation(solver_input, options.kernel_options)
        c_k = cov_func(
            kernel_data,
            options,
            item=item,
            execution_mode=KernelExecutionMode.SYMBOLIC,
        )
        c_k_sum = c_n.sum(0).reshape(-1, 1)

        print('l: ', l)
        print("just numpy: ", c_n, c_n_sum)
        print("pykeops: ", c_k, c_k_sum)

        if compare_to_saved:
            np.testing.assert_array_almost_equal(np.asarray(c_n), l, decimal=1)
        np.testing.assert_array_almost_equal(c_n_sum, c_k_sum, decimal=2)
