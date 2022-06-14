import numpy as np
import pytest
from matplotlib import pyplot as plt

from gempy_engine.core.data.exported_structs import InterpOutput
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.API.interp_single._interp_single_internals import _input_preprocess, _solve_interpolation
from gempy_engine.API.interp_single.interp_single_interface import interpolate_single_field
from gempy_engine.modules.kernel_constructor._covariance_assembler import _test_covariance_items, create_grad_kernel
from gempy_engine.modules.kernel_constructor._vectors_preparation import cov_vectors_preparation, \
    evaluation_vectors_preparations
from test.conftest import TEST_SPEED
from test.helper_functions import plot_2d_scalar_y_direction
from test.test_common.test_geometries.test_geometries import plot

from test.test_common.test_geometries.solutions import recumbent_weights, recumbent_weights_d1


class TestRecumbentFoldCovConstructionWithDrift:

    def test_recumbent_fold_scaled_grad(self, recumbent_fold_scaled):
        """ From old gempy
         [[666.67522329, 289.80267087,   0.        ,   0.        ,   0. ,   0.        ],
         [289.80267087, 666.67522329,   0.        ,   0.        ,   0. ,   0.        ],
         [  0.        ,   0.        , 666.67522329, 289.80267087,   0. ,   0.        ],
         [  0.        ,   0.        , 289.80267087, 666.67522329,   0. ,   0.        ],
         [  0.        ,   0.        ,   0.        ,   0.        , 666.67522329,  -34.26541266],
         [  0.        ,   0.        ,   0.        ,   0.        , -34.26541266,  666.67522329],]
        """
        interpolation_input, options, structure = recumbent_fold_scaled

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)
        solver_input = SolverInput(sp_internal, ori_internal, options)

        kernel_data = cov_vectors_preparation(solver_input)
        cov = _test_covariance_items(kernel_data, options, "cov_grad")

        print(options.c_o * cov[:6, :6])
        sol = np.array(
            [[666.67522329, 289.80267087, 0., 0., 0., 0.],
             [289.80267087, 666.67522329, 0., 0., 0., 0.],
             [0., 0., 666.67522329, 289.80267087, 0., 0.],
             [0., 0., 289.80267087, 666.67522329, 0., 0.],
             [0., 0., 0., 0., 666.67522329, -34.26541266],
             [0., 0., 0., 0., -34.26541266, 666.67522329]]
        )
        np.testing.assert_allclose((options.c_o * cov)[:6, :6], sol, rtol=.01)

    # def test_recumbent_fold_scaled_ci(self, recumbent_fold_scaled):
    #     """ From old GemPy
    #
    #     """
    #     interpolation_input, options, structure = recumbent_fold_scaled
    #
    #     # Within series
    #     xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
    #                                                             interpolation_input.grid,
    #                                                             interpolation_input.orientations,
    #                                                             interpolation_input.surface_points)
    #     solver_input = SolverInput(sp_internal, ori_internal, options)
    #
    #     kernel_data = cov_vectors_preparation(solver_input)
    #     cov = _test_covariance_items(kernel_data, options, "cov_sp")
    #
    #     val = options.i_res * options.c_o * cov
    #     print(val)
    #
    #     sol = np.array(
    #
    #     )
    #
    #     np.testing.assert_allclose(val[:6, 6:-9], sol, rtol=.03)

    def test_recumbent_fold_scaled_rest_ref(self, recumbent_fold_scaled):
        interpolation_input, options, structure = recumbent_fold_scaled

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)

        from test.test_common.test_geometries.solutions import recumbent_ref, recumbent_rest, recumbent_dips
        np.testing.assert_allclose(sp_internal.ref_surface_points, recumbent_ref, rtol=1e-7)
        np.testing.assert_allclose(sp_internal.rest_surface_points, recumbent_rest, rtol=1e-7)
        np.testing.assert_allclose(ori_internal.dip_positions_tiled, recumbent_dips, rtol=1e-7)

    def test_recumbent_fold_scaled_cig(self, recumbent_fold_scaled):
        """ Old gempy

        Gempy Engine
        [-1.295e+02, -1.210e+02,  3.881e+01,  3.395e+01,  0.000e+00, -2.144e+01],
        [-2.590e+02, -2.421e+02,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00],
        [ 3.965e-14,  0.000e+00, -1.968e+02, -1.840e+02,  0.000e+00,  0.000e+00],
        [-1.295e+02, -1.210e+02, -2.357e+02, -2.179e+02,  0.000e+00, -2.144e+01],
        [-2.590e+02, -2.421e+02, -1.968e+02, -1.840e+02,  0.000e+00,  0.000e+00],
        [-1.210e+02, -1.295e+02,  3.395e+01,  3.881e+01,  2.144e+01,  0.000e+00],
        [-2.421e+02, -2.590e+02,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00],
        [ 0.000e+00,  3.965e-14, -1.840e+02, -1.968e+02,  0.000e+00,  0.000e+00],
        [-1.210e+02, -1.295e+02, -2.179e+02, -2.357e+02,  2.144e+01,  0.000e+00],
        [-2.421e+02, -2.590e+02, -1.840e+02, -1.968e+02,  0.000e+00,  0.000e+00],

        """

        interpolation_input, options, structure = recumbent_fold_scaled

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)
        solver_input = SolverInput(sp_internal, ori_internal, options)

        kernel_data = cov_vectors_preparation(solver_input)
        cov = _test_covariance_items(kernel_data, options, "cov_grad_sp")
        #   print(cov)
        val = options.c_o * cov

        print(val[6:-9, :6])

        from test.test_common.test_geometries.solutions import recumbent_cgi
        print(val[6:-9, :6] - recumbent_cgi)

        np.testing.assert_allclose(val[6:-9, :6], recumbent_cgi, atol=.000001)

    def test_recumbent_fold_scaled_weights(self, recumbent_fold_scaled):
        """

        """
        interpolation_input, options, structure = recumbent_fold_scaled

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)

        solver_input = SolverInput(sp_internal, ori_internal, options)
        kernel_data = cov_vectors_preparation(solver_input)

        cov = _test_covariance_items(kernel_data, options, "cov")
        print(cov)

        weights = _solve_interpolation(solver_input)

        print(weights)


        weights_sol = recumbent_weights

        print(weights - weights_sol)
        np.testing.assert_allclose(weights, weights_sol, atol=1e-4)

    @pytest.mark.skipif(TEST_SPEED.value <= 2, reason="Global test speed below this test value.")
    def test_recumbent_fold_scaled(self, recumbent_fold_scaled):
        """
        Old gempy:
        Z_x:
        [1.3106523  1.34144988 1.37089355 ... 0.93999536, 0.90814532,   0.87492965  ]

        """

        interpolation_input, options, structure = recumbent_fold_scaled

        output: InterpOutput = interpolate_single_field(interpolation_input, options, structure)

        Z_x = output.exported_fields.scalar_field
        print(Z_x)

        np.testing.assert_allclose(Z_x[:3], np.array([1.3106523,  1.34144988, 1.37089355]), rtol=.02)
        np.testing.assert_allclose(Z_x[-3:], np.array([0.93999536, 0.90814532,  0.87492965 ]), rtol=.02)

        if plot:
            plot_2d_scalar_y_direction(interpolation_input, Z_x)
            plt.show()

    def test_recumbent_fold_universal_degree_2(self, recumbent_fold_scaled):
        """
            U_G __str__ =
            [[1. ,   0.   ,  0.  ,   1.7004, 0.    , 0.    , 1.0002, 1.2802, 0.    ],
             [1. ,   0.   ,  0.  ,   1.7004, 0.    , 0.    , 1.0002, 0.7202, 0.    ],
             [0. ,   1.   ,  0.  ,   0.    , 2.0004, 0.    , 0.8502, 0.    , 1.2802],
             [0. ,   1.   ,  0.  ,   0.    , 2.0004, 0.    , 0.8502, 0.    , 0.7202],
             [0. ,   0.   ,  1.  ,   0.    , 0.    , 2.5604, 0.    , 0.8502, 1.0002],
             [0. ,   0.   ,  1.  ,   0.    , 0.    , 1.4404, 0.    , 0.8502, 1.0002],]
        """
        interpolation_input, options, structure = recumbent_fold_scaled

        options.uni_degree = 2

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)

        solver_input = SolverInput(sp_internal, ori_internal, options)
        kernel_data = cov_vectors_preparation(solver_input)
        kernel = _test_covariance_items(kernel_data, options, "drift_ug")
        print(kernel)

        kernel_ug = np.array(
            [[1., 0., 0., 1.7004, 0., 0., 1.0002, 1.2802, 0.],
             [1., 0., 0., 1.7004, 0., 0., 1.0002, 0.7202, 0.],
             [0., 1., 0., 0., 2.0004, 0., 0.8502, 0., 1.2802],
             [0., 1., 0., 0., 2.0004, 0., 0.8502, 0., 0.7202],
             [0., 0., 1., 0., 0., 2.5604, 0., 0.8502, 1.0002],
             [0., 0., 1., 0., 0., 1.4404, 0., 0.8502, 1.0002], ]
        )

        np.testing.assert_allclose(kernel[:6, -9:], kernel_ug, atol=.02)

        kernel_sp = _test_covariance_items(kernel_data, options, "drift_usp")

        print(kernel_sp[6:-9, -9:])
        from test.test_common.test_geometries.solutions import recumbent_ui
        np.testing.assert_allclose(kernel_sp[6:-9, -9:], recumbent_ui, atol=.02)

    @pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Test speed to low")
    def test_recumbent_fold_universal_degree_2_scalar_kernel(self, recumbent_fold_scaled):
        """
        universal_kernel __str__ =
         [[0.6602     0.6602     0.6602     ... 0.9802     0.9802     0.9802    ]
         [0.5102     0.5102     0.5102     ... 1.4502     1.4502     1.4502    ]
         [0.5102     0.5302     0.5502     ... 0.8902     0.9102     0.9302    ]
         ...
         [0.33683404 0.33683404 0.33683404 ... 1.42148604 1.42148604 1.42148604]
         [0.33683404 0.35003804 0.36324204 ... 0.87257404 0.89217804 0.91178204]
         [0.26030404 0.27050804 0.28071204 ... 1.29096804 1.31997204 1.34897604]]


        Universal terms contribution __str__ = [1.34620525 1.37810425 1.40867413 ... 1.64379458 1.65044021 1.65575671]

        First 10 terms degree 2:
        array([[1.34620525, 1.37810425, 1.40867413, 1.43791488, 1.4658265 ,
                1.49240901, 1.51766238, 1.54158663, 1.56418176, 1.58544776]])

        First 10 terms degree 1
        array([[-0.11850139, -0.11850139, -0.11850139, -0.11850139, -0.11850139,
        -0.11850139, -0.11850139, -0.11850139, -0.11850139, -0.11850139]])

        """
        interpolation_input, options, structure = recumbent_fold_scaled

        options.uni_degree = 2

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)

        solver_input = SolverInput(sp_internal, ori_internal, options)
        kernel_data = evaluation_vectors_preparations(xyz_lvl0, solver_input)
        kernel = _test_covariance_items(kernel_data, options, "sigma_0_u_sp")

        print(kernel[-9:])

        if options.uni_degree == 1:
            contribution = (recumbent_weights_d1[-3:].reshape(-1, 1) * kernel[-3:]).sum(axis=0)
        elif options.uni_degree == 2:
            contribution = (recumbent_weights[-9:].reshape(-1, 1) * kernel[-9:]).sum(axis=0)

        print(contribution)

    @pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
    def test_recumbent_fold_universal_degree_2_gradient(self, recumbent_fold_scaled):
        interpolation_input, options, structure = recumbent_fold_scaled

        # region Reduce grid res
        resolution = [10, 10, 10]
        extent = [0.3301 - 0.005, .8201 + 0.005,
                  0.2551 - 0.005, 0.7451 + 0.005,
                  0.2551 - 0.005, 0.7451 + 0.005]

        from gempy_engine.core.data.grid import RegularGrid
        from gempy_engine.core.data.grid import Grid

        regular_grid = RegularGrid(extent, resolution)
        grid = Grid(regular_grid.values, regular_grid=regular_grid)

        interpolation_input.grid = grid
        # endregion

        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)

        options.uni_degree = 2

        # Scalar
        output: InterpOutput = interpolate_single_field(interpolation_input, options, structure)

        weights = output.weights
        Z_x = output.exported_fields.scalar_field

        # Gradient x
        kernel_data = evaluation_vectors_preparations(xyz_lvl0, SolverInput(sp_internal, ori_internal, options),
                                                      axis=0)

        export_grad_scalar = create_grad_kernel(kernel_data, options)
        grad_x = (weights @ export_grad_scalar)[:-105]

        print(f"\n Grad x: {grad_x.reshape(resolution)}")
        #np.testing.assert_array_almost_equal(grad_x, grad_x_sol, decimal=3)

        # Gradient Y
        kernel_data = evaluation_vectors_preparations(xyz_lvl0, SolverInput(sp_internal, ori_internal, options), axis=1)
        export_grad_scalar = create_grad_kernel(kernel_data, options)
        grad_y = (weights @ export_grad_scalar)[:-105]

        print(grad_y)
        print(f"\n Grad y: {grad_y.reshape(resolution)}")


        # Gradient Z
        kernel_data = evaluation_vectors_preparations(xyz_lvl0, SolverInput(sp_internal, ori_internal, options), axis=2)
        export_grad_scalar = create_grad_kernel(kernel_data, options)
        grad_z = (weights @ export_grad_scalar)[:-105]

        print(grad_z)
        print(f"\n Grad z: {grad_z.reshape(resolution)}")
        #np.testing.assert_array_almost_equal(grad_z, grad_z_sol, decimal=3)

        if plot or True:
            import matplotlib.pyplot as plt

            extent = [interpolation_input.grid.regular_grid.extent[0],
                      interpolation_input.grid.regular_grid.extent[1],
                      interpolation_input.grid.regular_grid.extent[4],
                      interpolation_input.grid.regular_grid.extent[5]]

            # region Plot GxGz
            plt.contourf(Z_x.reshape(resolution)[:, 5, :].T, N=40, cmap="autumn",
                         extent= extent
                         )

            plt.scatter(sp_internal.rest_surface_points[:, 0], sp_internal.rest_surface_points[:, 2])

            g_x = xyz_lvl0[:-105,0].reshape(resolution)
            g_z = xyz_lvl0[:-105,2].reshape(resolution)

            plt.quiver(g_x[:, 5, :], g_z[:, 5, :],
                       grad_x.reshape(resolution)[:,5,:],
                       grad_z.reshape(resolution)[:,5,:],
                       pivot="tail",
                       color='green', alpha=.6, )

            plt.show()
            # region Plot GxGz

            # region Plot GyGz
            extent = [interpolation_input.grid.regular_grid.extent[2],
                      interpolation_input.grid.regular_grid.extent[3],
                      interpolation_input.grid.regular_grid.extent[4],
                      interpolation_input.grid.regular_grid.extent[5]]


            plt.contourf(Z_x.reshape(resolution)[5, :, :].T, N=40, cmap="autumn",
                         extent= extent
                         )

            plt.scatter(sp_internal.rest_surface_points[:, 1], sp_internal.rest_surface_points[:, 2])

            g_y = xyz_lvl0[:-105,1].reshape(resolution)

            plt.quiver(g_y[5, :, :], g_z[5, :, :],
                       grad_y.reshape(resolution)[5, :, :],
                       grad_z.reshape(resolution)[5, :, :],
                       pivot="tail",
                       color='green', alpha=.6, )

            plt.show()
            # endregion


