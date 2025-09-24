import numpy as np
import pytest

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, \
    orientations_preprocess
from gempy_engine.modules.kernel_constructor._kernels_assembler import create_grad_kernel, \
    create_scalar_kernel
from gempy_engine.modules.kernel_constructor._test_assembler import _test_covariance_items
from gempy_engine.modules.kernel_constructor._vectors_preparation import \
    evaluation_vectors_preparations
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, \
    yield_b_vector
from gempy_engine.modules.solver.solver_interface import kernel_reduction

cov_sol = np.array(
    [[0.115, 0.073, 0., 0., 0., -0.004, -0.025, -0.039, -0.042, -0.045, -0.032, -0.044],
     [0.073, 0.115, 0., 0., -0.004, 0., -0.021, -0.038, -0.042, -0.047, -0.029, -0.045],
     [0., 0., 0.115, 0.093, 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0.093, 0.115, 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., -0.004, 0., 0., 0.115, 0.092, -0.001, -0.006, -0.012, -0.018, -0.003, -0.014],
     [-0.004, 0., 0., 0., 0.092, 0.115, 0.002, -0.001, -0.006, -0.013, 0.001, -0.009],
     [-0.025, -0.021, 0., 0., -0.001, 0.002, 0.007, 0.011, 0.011, 0.012, 0.009, 0.012],
     [-0.039, -0.038, 0., 0., -0.006, -0.001, 0.011, 0.018, 0.019, 0.021, 0.014, 0.02],
     [-0.042, -0.042, 0., 0., -0.012, -0.006, 0.011, 0.019, 0.022, 0.024, 0.015, 0.023],
     [-0.045, -0.047, 0., 0., -0.018, -0.013, 0.012, 0.021, 0.024, 0.027, 0.017, 0.026],
     [-0.032, -0.029, 0., 0., -0.003, 0.001, 0.009, 0.014, 0.015, 0.017, 0.012, 0.016],
     [-0.044, -0.045, 0., 0., -0.014, -0.009, 0.012, 0.02, 0.023, 0.026, 0.016, 0.024],
     ]
)

scalar_sol = np.array(
    [-0.286616, -0.218132, -0.107467, 0.050246, -0.257588, -0.204464, -0.114943, 0.020915, -0.227138, -0.193918,
     -0.147019, -0.046563, -0.206186, -0.189268, -0.176999, -0.136825])

grid = np.array([
    [0.25010, 0.50010, 0.12510],
    [0.25010, 0.50010, 0.29177],
    [0.25010, 0.50010, 0.45843],
    [0.25010, 0.50010, 0.62510],
    [0.41677, 0.50010, 0.12510],
    [0.41677, 0.50010, 0.29177],
    [0.41677, 0.50010, 0.45843],
    [0.41677, 0.50010, 0.62510],
    [0.58343, 0.50010, 0.12510],
    [0.58343, 0.50010, 0.29177],
    [0.58343, 0.50010, 0.45843],
    [0.58343, 0.50010, 0.62510],
    [0.75010, 0.50010, 0.12510],
    [0.75010, 0.50010, 0.29177],
    [0.75010, 0.50010, 0.45843],
    [0.75010, 0.50010, 0.62510]
])

plot = False


class TestCompareWithGempy_v2:
    @pytest.fixture(scope="class")
    def internals(self, simple_model):
        BackendTensor._change_backend(AvailableBackends.numpy, use_pykeops=False)

        surface_points = simple_model[0]
        orientations = simple_model[1]
        options = simple_model[2]
        tensors_structure = simple_model[3]

        # Prepare options
        interpolation_options = InterpolationOptions.from_args(
            range=5,
            c_o=5 ** 2 / 14 / 3,
            uni_degree=0,
            number_dimensions=2,
            kernel_function=AvailableKernelFunctions.cubic,
            i_res=1,
            gi_res=1
        )

    
        sp_internals = surface_points_preprocess(surface_points, tensors_structure.tensors_structure)
        ori_internals = orientations_preprocess(orientations)
        return sp_internals, ori_internals, options

    @pytest.fixture(scope="class")
    def weights(self, internals):
        sp_internals, ori_internals, options = internals
        solver_input = SolverInput(sp_internals, ori_internals)
        cov = yield_covariance(solver_input, options.kernel_options)
        b_vec = yield_b_vector(ori_internals, cov.shape[0])
        weights = kernel_reduction(cov, b_vec)
        return weights

    @pytest.mark.skipif(BackendTensor.engine_backend != AvailableBackends.numpy, reason="These tests only makes sense for numpy backend")
    def test_reduction(self, internals):
        sp_internals, ori_internals, options = internals
        # Test cov
        solver_input = SolverInput(sp_internals, ori_internals)
        cov = yield_covariance(solver_input, options.kernel_options)

        print("\n")
        print(cov)

        np.testing.assert_array_almost_equal(np.asarray(cov), cov_sol, decimal=1)

        # Test weights and b vector
        b_vec = yield_b_vector(ori_internals, cov.shape[0])
        weights = kernel_reduction(cov, b_vec, options.kernel_options)
        print(weights)

        weights_gempy_v2 = [6.402e+00, -1.266e+01, 2.255e-15, -2.784e-15, 1.236e+01, 2.829e+01, -6.702e+01, -6.076e+02,
                            1.637e+03, 1.053e+03, 2.499e+02, -2.266e+03]
        np.testing.assert_allclose(np.asarray(weights), weights_gempy_v2, rtol=2)
