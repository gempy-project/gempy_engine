import numpy as np
import pytest

from gempy_engine.config import BackendTensor, AvailableBackends
from gempy_engine.core.data import SurfacePoints, Orientations, InterpolationOptions, TensorsStructure
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess, orientations_preprocess
from gempy_engine.modules.kernel_constructor._covariance_assembler import create_grad_kernel, create_scalar_kernel, \
    _test_covariance_items
from gempy_engine.modules.kernel_constructor._vectors_preparation import evaluation_vectors_preparations, \
    evaluation_vectors_preparations_grad
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance, yield_b_vector
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

scalar_sol = [[-0.43369979, - 0.30058757, - 0.09845786, 0.17851481, - 0.38803627, - 0.27854967,
               - 0.11301122, 0.11947617, - 0.34158771, - 0.26367544, - 0.16590076, 0.00206044,
               - 0.31537991, - 0.26678353, - 0.2292369, - 0.15412643]]

scalar_gempy_v2 = np.array([-0.287, -0.218, -0.107, 0.05, -0.258, -0.204, -0.115, 0.021, -0.227, -0.194, -0.147, -0.047,
                            -0.206, -0.189, -0.177, -0.137])
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


@pytest.fixture(scope="module")
def gempy_v2_model_res():
    import numpy

    numpy.set_printoptions(precision=3, linewidth=200)

    dip_positions = np.array([
        [0.25010, 0.50010, 0.54177],
        [0.66677, 0.50010, 0.62510],
    ])
    sp = np.array([
        [0.25010, 0.50010, 0.37510],
        [0.50010, 0.50010, 0.37510],
        [0.66677, 0.50010, 0.41677],
        [0.70843, 0.50010, 0.47510],
        [0.75010, 0.50010, 0.54177],
        [0.58343, 0.50010, 0.39177],
        [0.73343, 0.50010, 0.50010],
    ])

    nugget_effect_scalar = 0
    spi = SurfacePoints(sp, nugget_effect_scalar)

    dip_gradients = np.array([[0, 0, 1],
                              [-.6, 0, .8]])
    nugget_effect_grad = 0

    range_ = 4.166666666667
    co = 0.1428571429

    ori_i = Orientations(dip_positions, dip_gradients, nugget_effect_grad)

    kri = InterpolationOptions(range_, co, 0, i_res=1, gi_res=1,
                               number_dimensions=3, kernel_function=AvailableKernelFunctions.cubic)
    _ = np.ones(3)
    tensor_structure = TensorsStructure(np.array([7]), _, _, _, _)
    return spi, ori_i, kri, tensor_structure


plot = False


class TestCompareWithGempy_v2:
    @pytest.fixture(scope="class")
    def internals(self, gempy_v2_model_res):
        BackendTensor.change_backend(AvailableBackends.numpy, pykeops_enabled=False)

        surface_points = gempy_v2_model_res[0]
        orientations = gempy_v2_model_res[1]
        options = gempy_v2_model_res[2]
        tensors_structure = gempy_v2_model_res[3]
        sp_internals = surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
        ori_internals = orientations_preprocess(orientations)
        return sp_internals, ori_internals, options

    @pytest.fixture(scope="class")
    def weights(self, internals):
        sp_internals, ori_internals, options = internals
        cov = yield_covariance(sp_internals, ori_internals, options)
        b_vec = yield_b_vector(ori_internals, cov.shape[0])
        weights = kernel_reduction(cov, b_vec)
        return weights

    def test_reduction(self, internals):
        sp_internals, ori_internals, options = internals
        # Test cov
        cov = yield_covariance(sp_internals, ori_internals, options)
        print("\n")
        print(cov)

        np.testing.assert_array_almost_equal(np.asarray(cov), cov_sol, decimal=3)

        # Test weights and b vector
        b_vec = yield_b_vector(ori_internals, cov.shape[0])
        weights = kernel_reduction(cov, b_vec)
        print(weights)

        weights_gempy_v2 = [6.402e+00, -1.266e+01, 2.255e-15, -2.784e-15, 1.236e+01, 2.829e+01, -6.702e+01, -6.076e+02,
                            1.637e+03, 1.053e+03, 2.499e+02, -2.266e+03]
        np.testing.assert_allclose(np.asarray(weights), weights_gempy_v2, rtol=2)

    def test_export_to_scalar(self, internals, weights):
        sp_internals, ori_internals, options = internals
        # Test sigma 0 sp
        kernel_data = evaluation_vectors_preparations(grid, sp_internals, ori_internals, options)
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
        print(f"\n Scalar field: {scalar_ff.reshape(4, 1, 4)}")

        np.testing.assert_allclose(np.asarray(scalar_ff), scalar_gempy_v2, rtol=1)

        if plot:
            import matplotlib.pyplot as plt

            plt.contourf(scalar_ff.reshape(4, 1, 4)[:, 0, :].T, N=40, cmap="autumn")
            plt.show()

    def test_export_to_grad(self, internals, weights):
        # Test gradient x
        np_grad_x = np.gradient(scalar_gempy_v2.reshape(4, 1, 4), axis=0)
        np_grad_y = np.gradient(scalar_gempy_v2.reshape(4, 1, 4), axis=2)
        print(f"\n Grad x 'sol': {np_grad_x}")

        sp_internals, ori_internals, options = internals

        # Gradient x
        kernel_data = evaluation_vectors_preparations_grad(grid, sp_internals, ori_internals,
                                                           options, axis=0)
        export_grad_scalar = create_grad_kernel(kernel_data, options)
        grad_x = weights @ export_grad_scalar
        print(f"\n Grad x: {grad_x.reshape(4, 1, 4)}")

        kernel_data = evaluation_vectors_preparations_grad(grid, sp_internals, ori_internals,
                                                           options, axis=2)
        export_grad_scalar = create_grad_kernel(kernel_data, options)
        grad_z = weights @ export_grad_scalar
        print(f"\n Grad z: {grad_z.reshape(4, 1, 4)}")

        if plot:
            import matplotlib.pyplot as plt

            plt.contourf(scalar_gempy_v2.reshape(4, 1, 4)[:, 0, :].T, N=40, cmap="autumn")
            plt.quiver(np_grad_x[:, 0, :], np_grad_y[:, 0, :],
                       pivot="tail",
                       color='blue', alpha=.6)

            plt.quiver(grad_x.reshape(4, 4), grad_z.reshape(4, 4),
                       pivot="tail",
                       color='green', alpha=.6)

            plt.show()
