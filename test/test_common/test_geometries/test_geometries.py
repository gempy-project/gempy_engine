import pytest

from gempy_engine.core.data.exported_structs import InterpOutput
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.integrations.interp_single._interp_single_internals import _input_preprocess
from gempy_engine.integrations.interp_single.interp_single_interface import interpolate_single_field
from gempy_engine.modules.kernel_constructor._covariance_assembler import _test_covariance_items
from gempy_engine.modules.kernel_constructor._vectors_preparation import cov_vectors_preparation, \
    evaluation_vectors_preparations
from test.helper_functions import plot_2d_scalar_y_direction

import numpy as np
np.set_printoptions(precision=3, linewidth=300)

plot = True


@pytest.mark.skip(reason="Not rescaled values seems broken")
def test_horizontal_stratigraphic(horizontal_stratigraphic):
    interpolation_input, options, structure = horizontal_stratigraphic

    output: InterpOutput = interpolate_single_field(interpolation_input, options, structure)
    Z_x = output.exported_fields.scalar_field

    if plot:
        plot_2d_scalar_y_direction(interpolation_input, Z_x)


class TestHorizontalStatCovConstruction:
    weights: np.array = np.array(
        [-1.437e-18,  2.359e-18, - 2.193e-18, 2.497e-18,  1.481e-03,  1.481e-03,
         5.969e-03, - 2.984e-03, - 2.984e-03, 5.969e-03, - 2.984e-03, - 5.969e-03,
         2.984e-03,  2.984e-03,  - 5.969e-03, 2.984e-03]
    )

    def test_horizontal_stratigraphic_scaled_grad(self, horizontal_stratigraphic_scaled):
        """ From old gempy
           [[533.332 418.886   0.      0.      0.      0.   ]
            [418.886 533.332   0.      0.      0.      0.   ]
            [  0.      0.    533.332 418.886   0.      0.   ]
            [  0.      0.    418.886 533.332   0.      0.   ]
            [  0.      0.      0.      0.    533.332 306.474]
            [  0.      0.      0.      0.    306.474 533.332]]
        """
        interpolation_input, options, structure = horizontal_stratigraphic_scaled

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)
        solver_input = SolverInput(sp_internal, ori_internal, options)

        kernel_data = cov_vectors_preparation(solver_input)
        cov = _test_covariance_items(kernel_data, options, "cov_grad")

        print(options.c_o * cov)

    def test_horizontal_stratigraphic_scaled_ci(self, horizontal_stratigraphic_scaled):
        """ From old GemPy
        [[ 95.655 125.244  24.476  71.179  89.584  85.742 114.988  21.054  64.687  81.867]
         [125.244 250.488  60.136 125.244 190.352 114.988 229.976  54.176 114.988 175.8  ]
         [ 24.476  60.136 176.171 151.695 116.035  21.054  54.176 160.523 139.469 106.348]
         [ 71.179 125.244 151.695 222.874 216.803  64.687 114.988 139.469 204.156 200.281]
         [ 89.584 190.352 116.035 216.803 306.387  81.867 175.8   106.348 200.281 282.148]
         [ 85.742 114.988  21.054  64.687  81.867  95.655 125.244  24.476  71.179  89.584]
         [114.988 229.976  54.176 114.988 175.8   125.244 250.488  60.136 125.244 190.352]
         [ 21.054  54.176 160.523 139.469 106.348  24.476  60.136 176.171 151.695 116.035]
         [ 64.687 114.988 139.469 204.156 200.281  71.179 125.244 151.695 222.874 216.803]
         [ 81.867 175.8   106.348 200.281 282.148  89.584 190.352 116.035 216.803 306.387]]
        """



        interpolation_input, options, structure = horizontal_stratigraphic_scaled

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)
        solver_input = SolverInput(sp_internal, ori_internal, options)

        kernel_data = cov_vectors_preparation(solver_input)
        cov = _test_covariance_items(kernel_data, options, "cov_sp")
        print(options.i_res * options.c_o * cov)

    def test_horizontal_stratigraphic_scaled_cig(self, horizontal_stratigraphic_scaled):
        """ Old gempy
        [[-1.301e+02 -1.209e+02  3.872e+01  3.346e+01  0.000e+00 -2.231e+01]
         [-2.603e+02 -2.419e+02  0.000e+00 -8.527e-14  0.000e+00  5.684e-14]
         [ 0.000e+00 -1.137e-13 -1.952e+02 -1.814e+02  0.000e+00  5.684e-14]
         [-1.301e+02 -1.209e+02 -2.339e+02 -2.149e+02  0.000e+00 -2.231e+01]
         [-2.603e+02 -2.419e+02 -1.952e+02 -1.814e+02  0.000e+00  1.279e-13]
         [-1.209e+02 -1.301e+02  3.346e+01  3.872e+01  2.231e+01  0.000e+00]
         [-2.419e+02 -2.603e+02  0.000e+00 -1.421e-13  0.000e+00  0.000e+00]
         [ 0.000e+00 -2.842e-14 -1.814e+02 -1.952e+02  0.000e+00  0.000e+00]
         [-1.209e+02 -1.301e+02 -2.149e+02 -2.339e+02  2.231e+01  0.000e+00]
         [-2.419e+02 -2.603e+02 -1.814e+02 -1.952e+02 -7.105e-14  0.000e+00]]

        Gempy Engine
        [-1.295e+02 -1.210e+02  3.881e+01  3.395e+01  0.000e+00 -2.144e+01
        [-2.590e+02 -2.421e+02  0.000e+00  0.000e+00  0.000e+00  0.000e+00
        [ 3.965e-14  0.000e+00 -1.968e+02 -1.840e+02  0.000e+00  0.000e+00
        [-1.295e+02 -1.210e+02 -2.357e+02 -2.179e+02  0.000e+00 -2.144e+01
        [-2.590e+02 -2.421e+02 -1.968e+02 -1.840e+02  0.000e+00  0.000e+00
        [-1.210e+02 -1.295e+02  3.395e+01  3.881e+01  2.144e+01  0.000e+00
        [-2.421e+02 -2.590e+02  0.000e+00  0.000e+00  0.000e+00  0.000e+00
        [ 0.000e+00  3.965e-14 -1.840e+02 -1.968e+02  0.000e+00  0.000e+00
        [-1.210e+02 -1.295e+02 -2.179e+02 -2.357e+02  2.144e+01  0.000e+00
        [-2.421e+02 -2.590e+02 -1.840e+02 -1.968e+02  0.000e+00  0.000e+00

        """

        interpolation_input, options, structure = horizontal_stratigraphic_scaled

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)
        solver_input = SolverInput(sp_internal, ori_internal, options)

        kernel_data = cov_vectors_preparation(solver_input)
        cov = _test_covariance_items(kernel_data, options, "cov_grad_sp")
        print(cov)
        print(options.gi_res * options.c_o * cov)

    def test_horizontal_stratigraphic_scaled_eval_i(self, horizontal_stratigraphic_scaled):
        """
        [0.013 0.013 0.012 ... 0.005 0.005 0.005]

        """
        interpolation_input, options, structure = horizontal_stratigraphic_scaled

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)
        solver_input = SolverInput(sp_internal, ori_internal, options)
        kernel_data = evaluation_vectors_preparations(xyz_lvl0, solver_input)
        kernel = 4 * _test_covariance_items(kernel_data, options, "sigma_0_sp")
        print(kernel)

        print(self.weights @ kernel)

    def test_horizontal_stratigraphic_scaled_eval_gi(self, horizontal_stratigraphic_scaled):
        """
         [-0.206 -0.204 -0.201 ... -0.091 -0.091 -0.091]

        """

        interpolation_input, options, structure = horizontal_stratigraphic_scaled

        # Within series
        xyz_lvl0, ori_internal, sp_internal = _input_preprocess(structure,
                                                                interpolation_input.grid,
                                                                interpolation_input.orientations,
                                                                interpolation_input.surface_points)
        solver_input = SolverInput(sp_internal, ori_internal, options)
        kernel_data = evaluation_vectors_preparations(xyz_lvl0, solver_input)
        kernel = 2 * _test_covariance_items(kernel_data, options, "sigma_0_grad_sp")
        print(kernel)
        print(self.weights @ kernel)

    @pytest.mark.skip(reason="Trigger only manually since it is too slow")
    def test_horizontal_stratigraphic_scaled(self, horizontal_stratigraphic_scaled):
        """
        Old gempy:
        [ 1.180e-17 -9.998e-18  3.805e-18  2.459e-17  1.503e-03  1.503e-03  5.872e-03
         -2.936e-03 -2.936e-03  5.872e-03 -2.936e-03 -5.872e-03  2.936e-03  2.936e-03
          -5.872e-03  2.936e-03]

        New Gempy
        [-1.437e-18  2.359e-18 -2.193e-18  2.497e-18  1.481e-03  1.481e-03  5.969e-03
         -2.984e-03 -2.984e-03  5.969e-03 -2.984e-03 -5.969e-03  2.984e-03  2.984e-03
          -5.969e-03  2.984e-03]
        """


        interpolation_input, options, structure = horizontal_stratigraphic_scaled

        output: InterpOutput = interpolate_single_field(interpolation_input, options, structure)

        weights = output.weights
        print(weights)

        Z_x = output.exported_fields.scalar_field

        if plot:
            plot_2d_scalar_y_direction(interpolation_input, Z_x)


def test_horizontal_stratigraphic_universal_1(horizontal_stratigraphic_scaled):
    """
    U_G __str__ = [[1.    0.    0.    2.    0.    0.    1.    1.125 0.   ]
                   [1.    0.    0.    2.    0.    0.    1.    0.875 0.   ]
                   [0.    1.    0.    0.    2.    0.    1.    0.    1.125]
                   [0.    1.    0.    0.    2.    0.    1.    0.    0.875]
                   [0.    0.    1.    0.    0.    2.25  0.    1.    1.   ]
                   [0.    0.    1.    0.    0.    1.75  0.    1.    1.   ]]
    U_I __str__ = [[-0.5   -0.    -0.    -0.75  -0.    -0.    -0.313 -0.563 -0.   ]
                 [-1.    -0.    -0.    -2.    -0.    -0.    -0.625 -1.125 -0.   ]
                 [-0.    -0.75  -0.    -0.    -1.5   -0.    -0.375 -0.    -0.844]
                 [-0.5   -0.75  -0.    -0.75  -1.5   -0.    -1.063 -0.563 -0.844]
                 [-1.    -0.75  -0.    -2.    -1.5   -0.    -1.75  -1.125 -0.844]
    """
    interpolation_input, options, structure = horizontal_stratigraphic_scaled

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


