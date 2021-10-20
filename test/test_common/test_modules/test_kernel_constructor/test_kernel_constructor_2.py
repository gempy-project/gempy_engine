import numpy as np
import pykeops
import pytest

from gempy_engine.core.backend_tensor import BackendTensor, AvailableBackends
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, \
    orientations_preprocess
from gempy_engine.modules.kernel_constructor._covariance_assembler import create_grad_kernel, \
    create_scalar_kernel, \
    _test_covariance_items, _compute_all_distance_matrices
from gempy_engine.modules.kernel_constructor._vectors_preparation import \
    evaluation_vectors_preparations, cov_vectors_preparation
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, \
    yield_b_vector
from gempy_engine.modules.solver.solver_interface import kernel_reduction

tfnp = BackendTensor.tfnp
tensor_types = BackendTensor.tensor_types


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

        surface_points = simple_model[0]
        orientations = simple_model[1]
        options = simple_model[2]
        tensors_structure = simple_model[3]

        options.i_res = 1
        options.gi_res = 1

        sp_internals = surface_points_preprocess(surface_points,
                                                 tensors_structure.number_of_points_per_surface)
        ori_internals = orientations_preprocess(orientations)
        return sp_internals, ori_internals, options

    @pytest.fixture(scope="class")
    def weights(self, internals):
        sp_internals, ori_internals, options = internals
        cov = yield_covariance(SolverInput(sp_internals, ori_internals, options))
        b_vec = yield_b_vector(ori_internals, cov.shape[0])
        weights = kernel_reduction(cov, b_vec)
        return tfnp.reshape(weights,(1, -1))

    def test_reduction(self, internals):
        sp_internals, ori_internals, options = internals
        # Test cov
        cov = yield_covariance(SolverInput(sp_internals, ori_internals, options))
        print("\n")
        print(cov)

        if BackendTensor.pykeops_enabled is not True:
            np.testing.assert_array_almost_equal(np.asarray(cov), cov_sol, decimal=3)

        # Test weights and b vector
        b_vec = yield_b_vector(ori_internals, cov.shape[0])
        weights = kernel_reduction(cov, b_vec)
        print(weights)

        weights_gempy_v2 = [6.402e+00, -1.266e+01, 2.255e-15, -2.784e-15, 1.236e+01, 2.829e+01, -6.702e+01, -6.076e+02,
                            1.637e+03, 1.053e+03, 2.499e+02, -2.266e+03]
        np.testing.assert_allclose(np.asarray(weights).reshape(-1), weights_gempy_v2, rtol=2)

    def test_export_to_scalar(self, internals, weights):
        sp_internals, ori_internals, options = internals

        options.uni_degree = 0

        # Test sigma 0 sp
        kernel_data = evaluation_vectors_preparations(grid, SolverInput(sp_internals, ori_internals, options))
        export_sp_contr = _test_covariance_items(kernel_data, options, item="sigma_0_sp")
        sp_contr = weights @ export_sp_contr

        # TODO: Add test
        print(f"\n Scalar field sp contr: {sp_contr}")

        # Test sigma grad - sp
        export_grad_sp_contr = _test_covariance_items(kernel_data, options, item="sigma_0_grad_sp")
        grad_sp_contr = weights @ export_grad_sp_contr
        print(f"\n Scalar field grad contr: {grad_sp_contr}")

        # Test scalar field
        export_scalar_ff = create_scalar_kernel(kernel_data, options)
        scalar_ff = weights @ export_scalar_ff

        if BackendTensor.engine_backend == AvailableBackends.tensorflowCPU or BackendTensor.engine_backend == AvailableBackends.tensorflowGPU:
            scalar_ff = scalar_ff.numpy()

        print(f"\n Scalar field: {scalar_ff.reshape(4, 1, 4)}")

        if plot or True:
            import matplotlib.pyplot as plt

            plt.contourf(scalar_ff.reshape(4, 1, 4)[:, 0, :].T, N=40, cmap="autumn",
                         extent=[0.25, 0.75, .12510, .62510])
            plt.scatter(sp_internals.rest_surface_points[:, 0], sp_internals.rest_surface_points[:, 2])

            plt.show()

        np.testing.assert_allclose(np.asarray(scalar_ff)[0], scalar_sol, rtol=1)

    @pytest.mark.skipif(BackendTensor.engine_backend is AvailableBackends.tensorflowCPU or
                        BackendTensor.engine_backend is AvailableBackends.tensorflowGPU, reason="Only test against numpy")
    def test_export_to_grad(self, internals, weights):
        # Test gradient x
        np_grad_x = np.gradient(scalar_sol.reshape((4, 1, 4)), axis=0)
        np_grad_y = np.gradient(scalar_sol.reshape((4, 1, 4)), axis=2)

        grad_x_sol = np.array(
            [0.154, 0.08, 0.012, -0.048, 0.178, 0.064, -0.138, -0.307, 0.153, 0.052, -0.225, -0.521, 0.049, -0.066,
             -0.183, -0.475])
        grad_z_sol = np.array(
            [0.328, 0.526, 0.818, 0.949, 0.257, 0.412, 0.684, 0.876, 0.182, 0.23, 0.378, 0.803, 0.107, 0.101, 0.086,
             0.578])

        print(f"\n Grad x 'sol': {np_grad_x}")

        sp_internals, ori_internals, options = internals

        # Gradient x
        kernel_data = evaluation_vectors_preparations(grid, SolverInput(sp_internals, ori_internals, options),
                                                      axis=0)
        export_grad_scalar = create_grad_kernel(kernel_data, options)
        grad_x = weights @ export_grad_scalar

        print(f"\n Grad x: {grad_x.reshape(4, 1, 4)}")
        np.testing.assert_array_almost_equal(grad_x.reshape(-1), grad_x_sol, decimal=3)

        kernel_data = evaluation_vectors_preparations(grid, SolverInput(sp_internals, ori_internals, options), axis=2)
        export_grad_scalar = create_grad_kernel(kernel_data, options)
        grad_z = weights @ export_grad_scalar
        print(grad_z)
        print(f"\n Grad z: {grad_z.reshape(4, 1, 4)}")
        np.testing.assert_array_almost_equal(grad_z.reshape(-1), grad_z_sol, decimal=3)
        if plot or True:
            import matplotlib.pyplot as plt

            plt.contourf(scalar_sol.reshape((4, 1, 4))[:, 0, :].T, N=40, cmap="autumn",
                         extent=[0.25, 0.75, .12510, .62510]
                         )
            plt.quiver(grid[:, 0], grid[:,2], np_grad_x[:, 0, :], np_grad_y[:, 0, :],
                       pivot="tail",
                       color='blue', alpha=.6,   )

            plt.scatter(sp_internals.rest_surface_points[:, 0], sp_internals.rest_surface_points[:, 2])


            plt.quiver(grid[:, 0], grid[:,2], grad_x.reshape(4, 4), grad_z.reshape(4, 4),
                       pivot="tail",
                       color='green', alpha=.6,  )

            plt.show()


@pytest.mark.skipif(BackendTensor.engine_backend is not AvailableBackends.numpyPykeopsCPU and
                    BackendTensor.engine_backend is not AvailableBackends.numpyPykeopsGPU, reason="Only test against pykeops")
class TestPykeops:
    @pytest.fixture(scope="class")
    def internals(self, simple_model):

        surface_points = simple_model[0]
        orientations = simple_model[1]
        options = simple_model[2]
        tensors_structure = simple_model[3]

        # options.i_res = 1
        # options.gi_res = 1

        sp_internals = surface_points_preprocess(surface_points,
                                                 tensors_structure.number_of_points_per_surface)
        ori_internals = orientations_preprocess(orientations)
        return sp_internals, ori_internals, options

    @pytest.fixture(scope="class")
    def weights(self, internals):
        sp_internals, ori_internals, options = internals
        cov = yield_covariance(SolverInput(sp_internals, ori_internals, options))
        b_vec = yield_b_vector(ori_internals, cov.shape[0])
        weights = kernel_reduction(cov, b_vec)
        return tfnp.reshape(weights,(1, -1))

    def test_kernel(self, internals):
        """
        This test should compile quite fast and it is meant to check if:
        - Range is cte
        - Distances are squared
        """
        sp_internals, ori_internals, options = internals

        # Test kernel
        solver_input = SolverInput(sp_internals, ori_internals, options)
        kernel_data = cov_vectors_preparation(solver_input)
        kernel = _test_covariance_items(kernel_data, options, "cov_sp")

        print("\n")
        print(kernel)
        test_pykeops = kernel.sum(axis=0, backend="CPU")
        print(test_pykeops)

    def test_cov(self, internals):
        """
        This test is meant to be used to check with how many terms the compilation time raises too much

        Times:
            -  cov_grad and cov_sp: 4s
            -  cov_grad and cov_sp and cov_grad_sp: 4s
            -  cov_grad cov_sp cov_grad_sp drift: 6s
        """
        sp_internals, ori_internals, options = internals

        # Test kernel
        solver_input = SolverInput(sp_internals, ori_internals, options)
        cov = yield_covariance(solver_input)

        print("\n")
        print(cov)
        test_pykeops = cov.sum(axis=0, backend="CPU")

        print(test_pykeops)

    def test_distances(self, internals):
        sp_internals, ori_internals, options = internals

        # Test kernel
        solver_input = SolverInput(sp_internals, ori_internals, options)
        kernel_data = cov_vectors_preparation(solver_input)

        dm = _compute_all_distance_matrices(kernel_data.cartesian_selector, kernel_data.ori_sp_matrices)
        print(dm.r_ref_ref.sum(axis=0, backend="CPU"))


    def test_reduction(self, internals):
        """
        This test is meant to be used to check with how many terms the compilation time raises too much

        Times:    in brakets number of Variables    //dynamicRange-cubic//staticRange-cubic//staticRange-exp//DynamicRange-exp
            -  cov_grad:                               (37) 8 sec       // (30) 4 sec      // (10) 3 sec    //
            -  cov_grad and cov_sp:                     >5 min          // (32) 4 sec      // (14) 3 sec    //
            -  cov_grad and cov_sp and cov_grad_sp:                     //  >5 min         // (15) 3 sec    //
            -  cov_grad cov_sp cov_grad_sp drift:                                          // (42) 4 sec    // (49) 5 sec
        """
        #BackendTensor.change_backend(AvailableBackends.numpy, use_gpu=False, pykeops_enabled=True)


        sp_internals, ori_internals, options = internals
        options.kernel_function = AvailableKernelFunctions.exponential
        # options.range = 4.464646446464646464
        # options.i_res = 4
        # options.gi_res =2

        # Test cov
        cov = yield_covariance(SolverInput(sp_internals, ori_internals, options))
        print("\n")
        print(cov)


        # Test weights and b vector
        b_vec = yield_b_vector(ori_internals, cov.shape[0])
        weights = kernel_reduction(cov, b_vec, compute=True)
        print(weights)

        weights_gempy_v2 = [6.402e+00, -1.266e+01, 2.255e-15, -2.784e-15, 1.236e+01, 2.829e+01, -6.702e+01, -6.076e+02,
                            1.637e+03, 1.053e+03, 2.499e+02, -2.266e+03]
        if options.kernel_function is AvailableKernelFunctions.cubic:
            np.testing.assert_allclose(np.asarray(weights).reshape(-1), weights_gempy_v2, rtol=2)
