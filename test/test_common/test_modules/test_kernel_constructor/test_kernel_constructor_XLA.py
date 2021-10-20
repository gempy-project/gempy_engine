import numpy as np
import pytest
import tensorflow

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess
from gempy_engine.modules.kernel_constructor._covariance_assembler import _test_covariance_items, create_scalar_kernel
from gempy_engine.modules.kernel_constructor._vectors_preparation import evaluation_vectors_preparations
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, yield_b_vector
from gempy_engine.modules.solver.solver_interface import kernel_reduction
from test.test_common.test_modules.test_kernel_constructor.test_kernel_constructor_2 import cov_sol, grid, plot, \
    scalar_sol


@pytest.mark.skipif(BackendTensor.engine_backend != AvailableBackends.tensorflowCPU or BackendTensor.engine_backend != AvailableBackends.tensorflowGPU, reason="only with tensorflow")
class TestXLACompareWithGempy_v2:
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
        return weights

    def test_reduction(self, internals):
        sp_internals, ori_internals, options = internals

        @tensorflow.function
        def tf_function1(internals):
            sp_internals, ori_internals, options = internals

            solver_input = SolverInput(sp_internals, ori_internals, options)
            cov = yield_covariance(solver_input)
            return cov

        # Test cov
        cov = tf_function1(internals)
        print("\n")
        print(cov)

        np.testing.assert_array_almost_equal(np.asarray(cov), cov_sol, decimal=3)

        # Test weights and b vector
        b_vec = yield_b_vector(ori_internals, cov.shape[0])
        weights = kernel_reduction(cov, b_vec)
        print("Numpy weights: ", weights)

        @tensorflow.function
        def tf_function_weights(internals):
            sp_internals, ori_internals, options = internals

            solver_input = SolverInput(sp_internals, ori_internals, options)
            cov = yield_covariance(solver_input)
            b_vec = yield_b_vector(ori_internals, cov.shape[0])
            weights = kernel_reduction(cov, b_vec)
            return weights
        weights_tf = tf_function_weights(internals)
        print("TF weights: ", weights_tf)

        weights_gempy_v2 = [6.402e+00, -1.266e+01, 2.255e-15, -2.784e-15, 1.236e+01, 2.829e+01, -6.702e+01, -6.076e+02,
                            1.637e+03, 1.053e+03, 2.499e+02, -2.266e+03]
        np.testing.assert_allclose(np.asarray(weights)[:, 0], weights_gempy_v2, rtol=2)
        np.testing.assert_allclose(weights, weights_tf, rtol=2)


    def test_export_to_scalar(self, internals, weights):
        sp_internals, ori_internals, options = internals
        weights = tensorflow.reshape(weights, (1,-1))
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
        scalar_ff = scalar_ff.numpy()
        print(f"\n Scalar field: {scalar_ff.reshape(4, 1, 4)}")

        @tensorflow.function
        def tf_function_scalar(internals, weights):
            sp_internals, ori_internals, options = internals
            export_scalar_ff = create_scalar_kernel(kernel_data, options)
            scalar_ff = weights @ export_scalar_ff
            return scalar_ff

        scalar_ff_tf = tf_function_scalar(internals, weights)
        scalar_ff_tf = scalar_ff_tf.numpy()
        print(f"\n Scalar field TF: {scalar_ff_tf.reshape(4, 1, 4)}")

        if plot or True:
            import matplotlib.pyplot as plt

            plt.contourf(scalar_ff.reshape(4, 1, 4)[:, 0, :].T, N=40, cmap="autumn",
                         extent=[0.25, 0.75, .12510, .62510])
            plt.scatter(sp_internals.rest_surface_points[:, 0], sp_internals.rest_surface_points[:, 2])

            plt.show()

        np.testing.assert_allclose(np.asarray(scalar_ff)[0, :], scalar_sol, rtol=1)
        np.testing.assert_allclose(np.asarray(scalar_ff)[0, :], scalar_ff_tf[0, :], rtol=1)

    # def test_export_to_grad(self, internals, weights):
    #     # Test gradient x
    #     np_grad_x = np.gradient(scalar_sol.reshape((4, 1, 4)), axis=0)
    #     np_grad_y = np.gradient(scalar_sol.reshape((4, 1, 4)), axis=2)
    #
    #     grad_x_sol = np.array(
    #         [0.154, 0.08, 0.012, -0.048, 0.178, 0.064, -0.138, -0.307, 0.153, 0.052, -0.225, -0.521, 0.049, -0.066,
    #          -0.183, -0.475])
    #     grad_z_sol = np.array(
    #         [0.328, 0.526, 0.818, 0.949, 0.257, 0.412, 0.684, 0.876, 0.182, 0.23, 0.378, 0.803, 0.107, 0.101, 0.086,
    #          0.578])
    #
    #     print(f"\n Grad x 'sol': {np_grad_x}")
    #
    #     sp_internals, ori_internals, options = internals
    #
    #     # Gradient x
    #     kernel_data = evaluation_vectors_preparations(grid, SolverInput(sp_internals, ori_internals, options),
    #                                                   axis=0)
    #     export_grad_scalar = create_grad_kernel(kernel_data, options)
    #     grad_x = weights @ export_grad_scalar
    #
    #     print(f"\n Grad x: {grad_x.reshape(4, 1, 4)}")
    #     np.testing.assert_array_almost_equal(grad_x, grad_x_sol, decimal=3)
    #
    #     kernel_data = evaluation_vectors_preparations(grid, SolverInput(sp_internals, ori_internals, options), axis=2)
    #     export_grad_scalar = create_grad_kernel(kernel_data, options)
    #     grad_z = weights @ export_grad_scalar
    #     print(grad_z)
    #     print(f"\n Grad z: {grad_z.reshape(4, 1, 4)}")
    #     np.testing.assert_array_almost_equal(grad_z, grad_z_sol, decimal=3)
    #     if plot or True:
    #         import matplotlib.pyplot as plt
    #
    #         plt.contourf(scalar_sol.reshape((4, 1, 4))[:, 0, :].T, N=40, cmap="autumn",
    #                      extent=[0.25, 0.75, .12510, .62510]
    #                      )
    #         plt.quiver(grid[:, 0], grid[:,2], np_grad_x[:, 0, :], np_grad_y[:, 0, :],
    #                    pivot="tail",
    #                    color='blue', alpha=.6,   )
    #
    #         plt.scatter(sp_internals.rest_surface_points[:, 0], sp_internals.rest_surface_points[:, 2])
    #
    #
    #         plt.quiver(grid[:, 0], grid[:,2], grad_x.reshape(4, 4), grad_z.reshape(4, 4),
    #                    pivot="tail",
    #                    color='green', alpha=.6,  )
    #
    #         plt.show()