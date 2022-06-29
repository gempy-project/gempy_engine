import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, StackRelationType, TensorsStructure, StacksStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.interp_output import InterpOutput
from test import helper_functions_pyvista
from test.conftest import TEST_SPEED
from test.helper_functions import plot_block, plot_2d_scalar_y_direction

PLOT = False


def test_creating_one_fault_kernel_with_dummy_data(simple_model_2):
    from gempy_engine.modules.data_preprocess._input_preparation import orientations_preprocess

    surface_points = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    input_data_descriptor: InputDataDescriptor = simple_model_2[3]

    ori_internals = orientations_preprocess(orientations)

    sp_points_fault_values = np.array([0, 0, 1, 1, 0, 0, 1])

    # region Preprocess
    # * This dadata has to go to SolverInput

    partitions_bool = input_data_descriptor.tensors_structure.partitions_bool
    number_repetitions = input_data_descriptor.tensors_structure.number_of_points_per_surface - 1

    ref_points = sp_points_fault_values[partitions_bool]

    ref_matrix_val_repeated = np.repeat(ref_points, number_repetitions, 0)
    rest_matrix_val = sp_points_fault_values[~partitions_bool]

    # endregion

    # region vectors_preparation
    # * The closest I need I guess is dips_ref_d1 dips_rest_d1

    matrix_val = rest_matrix_val.reshape(-1, 1)
    fault_vector_rest = _fault_assembler(matrix_val, options, ori_internals)
    fault_vector_ref = _fault_assembler(ref_matrix_val_repeated.reshape(-1, 1), options, ori_internals)

    fault_vector_ref_i = fault_vector_ref[:, None, :]
    fault_vector_rest_i = fault_vector_rest[:, None, :]

    fault_vector_ref_j = fault_vector_ref[None, :, :]
    fault_vector_rest_j = fault_vector_rest[None, :, :]

    # endregion

    # region matrix
    fault_ref = (fault_vector_ref_i * fault_vector_ref_j).sum(axis=-1)
    fault_rest = (fault_vector_rest_i * fault_vector_rest_j).sum(axis=-1)

    bar = (fault_rest - fault_ref)
    # endregion

    # region Selector
    from gempy_engine.modules.kernel_constructor import _structs
    selector_components = _structs.DriftMatrixSelector(10, 10, 1)
    selector = (selector_components.sel_ui * (selector_components.sel_vj + 1)).sum(-1)

    # endregion

    fault_drift = selector * bar

    pass


def test_creating_scalar_kernel_with_dummy_data(simple_model_interpolation_input):
    interpolation_input = simple_model_interpolation_input[0]
    options = simple_model_interpolation_input[1]
    input_data_descriptor: InputDataDescriptor = simple_model_interpolation_input[2]

    grid = interpolation_input.grid.values

    # * This is the fault input
    fault_values = np.random.randint(0, 2, grid.shape[0])
    sp_points_fault_values = np.array([0, 0, 1, 1, 0, 0, 1])

    # region Preprocess

    partitions_bool = input_data_descriptor.tensors_structure.partitions_bool
    number_repetitions = input_data_descriptor.tensors_structure.number_of_points_per_surface - 1

    ref_points = sp_points_fault_values[partitions_bool]
    ref_matrix_val_repeated = np.repeat(ref_points, number_repetitions, 0)

    # endregion

    # region vectors_preparation    
    fault_vector_ref = _fault_assembler(ref_matrix_val_repeated.reshape(-1, 1), options, 2)

    fault_vector_ref_i = fault_vector_ref[:, None, :]
    fault_vector_ref_j = fault_values.reshape(-1, 1)[None, :, :]

    fault_drift = (fault_vector_ref_i * fault_vector_ref_j).sum(axis=-1)
    print("Fault drift:", fault_drift)


def _fault_assembler(matrix_val, options, ori_size):
    interpolation_options = None
    n_dim = 1
    n_uni_eq = options.n_uni_eq  # * Number of equations. This should be how many graben_data are active
    n_faults = 1
    z = np.zeros((ori_size, n_dim))
    z2 = np.zeros((n_uni_eq, n_dim))
    z3 = np.ones((n_faults, n_dim))
    # Degree 1
    return np.vstack((z, matrix_val, z2, z3))


def test_creating_several_faults_kernel_with_dummy_data(simple_model_2):
    from gempy_engine.modules.data_preprocess._input_preparation import orientations_preprocess

    matrix_val = simple_model_2[0]
    orientations = simple_model_2[1]
    options = simple_model_2[2]
    input_data_descriptor: InputDataDescriptor = simple_model_2[3]

    ori_internals = orientations_preprocess(orientations)

    # ! For each fault we are going to need to compute everything again
    sp_points_fault_values = np.array([[0, 0, 1, 1, 0, 0, 1],
                                       [0, 1, 1, 1, 1, 0, 1],
                                       [0, 0, 1, 0, 0, 1, 1]]).T

    # region Preprocess
    # * This dadata has to go to SolverInput

    partitions_bool = input_data_descriptor.tensors_structure.partitions_bool
    number_repetitions = input_data_descriptor.tensors_structure.number_of_points_per_surface - 1

    ref_points = sp_points_fault_values[partitions_bool]

    ref_matrix_val_repeated = np.repeat(ref_points, number_repetitions, 0)
    rest_matrix_val = sp_points_fault_values[~partitions_bool]

    # endregion

    # region vectors_preparation
    # * The closest I need I guess is dips_ref_d1 dips_rest_d1

    n_faults = 3

    def foo(matrix_val, options, ori_internals):
        ori_size = ori_internals.n_orientations_tiled
        interpolation_options = None
        n_dim = n_faults
        n_uni_eq = options.n_uni_eq  # * Number of equations. This should be how many graben_data are active

        z = np.zeros((ori_size, n_dim))
        z2 = np.zeros((n_uni_eq, n_dim))
        z3 = np.ones((n_faults, n_dim))
        # Degree 1
        return np.vstack((z, matrix_val, z2, z3))

    matrix_val = rest_matrix_val.reshape(-1, n_faults)
    fault_vector_rest = foo(matrix_val, options, ori_internals)
    fault_vector_ref = foo(ref_matrix_val_repeated.reshape(-1, n_faults), options, ori_internals)

    fault_vector_ref_i = fault_vector_ref[:, None, :]
    fault_vector_rest_i = fault_vector_rest[:, None, :]

    fault_vector_ref_j = fault_vector_ref[None, :, :]
    fault_vector_rest_j = fault_vector_rest[None, :, :]

    # endregion

    # region matrix
    fault_ref = (fault_vector_ref_i * fault_vector_ref_j).sum(axis=-1).astype(bool)
    fault_rest = (fault_vector_rest_i * fault_vector_rest_j).sum(axis=-1).astype(bool)

    bar = (fault_rest ^ fault_ref)

    # endregion

    # region Selector
    from gempy_engine.modules.kernel_constructor import _structs
    cov_size = 12
    foo = _structs.DriftMatrixSelector(cov_size, cov_size, n_faults)
    selector = (foo.sel_ui * (foo.sel_vj + 1)).sum(-1)  # * This has to be a list

    # endregion

    fault_matrix = selector * bar

    pass


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_fault_kernel(unconformity_complex, n_oct_levels=1):
    interpolation_input, options, structure = unconformity_complex
    structure.stack_structure.masking_descriptor = [StackRelationType.ERODE, StackRelationType.FAULT, False]

    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    # TODO: Grab second scalar and create fault kernel
    outputs = solutions.octrees_output[0].outputs_centers
    output: InterpOutput = outputs[1]

    all_values = output.scalar_fields._values_block
    sp_val = all_values[0, -2:]

    if PLOT or True:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].values_block, grid)
        plot_block(outputs[1].values_block, grid)
        plot_block(outputs[2].values_block, grid)

    if True:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].final_block, grid)
        plot_block(outputs[1].final_block, grid)
        plot_block(outputs[2].final_block, grid)

    if True:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes
        )


def test_one_fault_model(one_fault_model, n_oct_levels=1):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    options.dual_contouring = False

    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    # TODO: Grab second scalar and create fault kernel
    outputs = solutions.octrees_output[0].outputs_centers

    last_cov = outputs[-1].exported_fields.debug
    gempy_v2_cov = test_whaaaat()

    diff = last_cov - gempy_v2_cov

    if True:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

    if PLOT or True:
        grid = interpolation_input.grid.regular_grid

        plot_block(outputs[0].values_block, grid)
        plot_block(outputs[0].squeezed_mask_array, grid)
        print(outputs[0].squeezed_mask_array)
        plot_block(outputs[1].values_block, grid)
        plot_block(outputs[2].values_block, grid)

    if True:
        #plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        #plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

        # plot_block(outputs[0].final_block, grid)
        # plot_block(outputs[1].final_block, grid)
        # plot_block(outputs[2].final_block, grid)


def plot_scalar_and_input_2d(foo, interpolation_input, outputs, structure: StacksStructure):
    structure.stack_number = foo
    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(interpolation_input, structure)
    plot_2d_scalar_y_direction(interpolation_input_i, outputs[foo].exported_fields.scalar_field)


def plot_block_and_input_2d(stack_number, interpolation_input, outputs, structure: StacksStructure):
    grid = interpolation_input.grid.regular_grid

    structure.stack_number = stack_number
    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(interpolation_input, structure)
    plot_block(outputs[stack_number].final_block, grid, interpolation_input_i)


def test_whaaaat():
    foo = np.array(
        [[5.33e+02, -1.11e+02, -1.08e+02, 5.04e+02, 4.76e+02, -1.05e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 3.00e+01, 4.91e+01, 0.00e+00, 0.00e+00, 6.70e+01, 1.63e+02, 1.72e+02, 1.63e+02, 1.18e+02, 1.09e+02, 1.18e+02, 8.07e+00, -1.37e+01, 8.07e+00, 0.00e+00, 1.97e+01, 1.97e+01, -8.34e+00, -3.55e-14, 5.32e+01, 4.49e+01, 4.49e+01, -9.64e+01, -1.16e+02, -9.64e+01, -1.08e+02, -1.29e+02, -1.08e+02, 7.27e+00, -1.78e-14, -4.41e+01, -5.14e+01, -4.41e+01, -1.38e+02,
          -1.57e+02, -1.38e+02, -1.49e+02, -1.70e+02, -1.49e+02, 1.00e+00, 0.00e+00, 0.00e+00, 1.00e-04],
         [-1.11e+02, 5.33e+02, 5.04e+02, -1.12e+02, -1.12e+02, 4.76e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 3.00e+01, 0.00e+00, 0.00e+00, 1.01e+01, -1.01e+01, 0.00e+00, 1.52e+02, 1.72e+02, 1.52e+02, 1.64e+02, 1.86e+02, 1.64e+02, 5.41e+01, 6.31e+01, 5.41e+01, 0.00e+00, 8.95e+00, 8.95e+00, 2.20e+01, 1.99e-13, 8.13e+00, -1.17e+01, -1.17e+01,
          -1.56e+02, -1.64e+02, -1.56e+02, -1.11e+02, -1.03e+02, -1.11e+02, 1.99e+01, -8.53e-14, 1.17e+01, 3.38e+01, 1.17e+01, -1.44e+02, -1.51e+02, -1.44e+02, -9.99e+01, -9.26e+01, -9.99e+01, 1.00e+00, 0.00e+00, 0.00e+00, 1.00e-04],
         [-1.08e+02, 5.04e+02, 5.33e+02, -1.11e+02, -1.12e+02, 5.04e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 4.91e+01, 0.00e+00, 0.00e+00, 3.00e+01, 1.01e+01, 0.00e+00, 1.50e+02, 1.69e+02, 1.50e+02, 1.61e+02, 1.83e+02, 1.61e+02, 5.32e+01, 6.16e+01, 5.32e+01, 0.00e+00, 8.34e+00, 8.34e+00, 2.18e+01, -1.14e-13, 8.07e+00, -1.16e+01, -1.16e+01,
          -1.55e+02, -1.64e+02, -1.55e+02, -1.10e+02, -1.01e+02, -1.10e+02, 1.99e+01, -1.42e-13, 1.17e+01, 3.38e+01, 1.17e+01, -1.44e+02, -1.53e+02, -1.44e+02, -9.95e+01, -9.11e+01, -9.95e+01, 1.00e+00, 0.00e+00, 0.00e+00, 1.00e-04],
         [5.04e+02, -1.12e+02, -1.11e+02, 5.33e+02, 5.04e+02, -1.08e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.01e+01, 3.00e+01, 0.00e+00, 0.00e+00, 4.91e+01, 1.64e+02, 1.73e+02, 1.64e+02, 1.19e+02, 1.11e+02, 1.19e+02, 8.13e+00, -1.39e+01, 8.13e+00, 0.00e+00, 1.99e+01, 1.99e+01, -8.95e+00, -1.42e-14, 5.41e+01, 4.52e+01, 4.52e+01,
          -9.83e+01, -1.18e+02, -9.83e+01, -1.10e+02, -1.32e+02, -1.10e+02, 8.34e+00, -2.13e-14, -4.49e+01, -5.32e+01, -4.49e+01, -1.41e+02, -1.61e+02, -1.41e+02, -1.53e+02, -1.74e+02, -1.53e+02, 1.00e+00, 0.00e+00, 0.00e+00, 1.00e-04],
         [4.76e+02, -1.12e+02, -1.12e+02, 5.04e+02, 5.33e+02, -1.11e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, -1.01e+01, 1.01e+01, 0.00e+00, 0.00e+00, 3.00e+01, 1.64e+02, 1.71e+02, 1.64e+02, 1.20e+02, 1.12e+02, 1.20e+02, 8.13e+00, -1.39e+01, 8.13e+00, 0.00e+00, 1.99e+01, 1.99e+01, -8.34e+00, -2.13e-14, 5.32e+01, 4.49e+01, 4.49e+01,
          -9.95e+01, -1.19e+02, -9.95e+01, -1.11e+02, -1.33e+02, -1.11e+02, 8.95e+00, -1.42e-14, -4.52e+01, -5.41e+01, -4.52e+01, -1.43e+02, -1.63e+02, -1.43e+02, -1.55e+02, -1.77e+02, -1.55e+02, 1.00e+00, 0.00e+00, 0.00e+00, 1.00e-04],
         [-1.05e+02, 4.76e+02, 5.04e+02, -1.08e+02, -1.11e+02, 5.33e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 6.70e+01, 0.00e+00, 0.00e+00, 4.91e+01, 3.00e+01, 0.00e+00, 1.45e+02, 1.64e+02, 1.45e+02, 1.56e+02, 1.77e+02, 1.56e+02, 5.14e+01, 5.86e+01, 5.14e+01, 0.00e+00, 7.27e+00, 7.27e+00, 2.14e+01, 1.71e-13, 7.94e+00, -1.15e+01, -1.15e+01,
          -1.53e+02, -1.61e+02, -1.53e+02, -1.08e+02, -9.95e+01, -1.08e+02, 1.97e+01, 0.00e+00, 1.16e+01, 3.35e+01, 1.16e+01, -1.43e+02, -1.52e+02, -1.43e+02, -9.83e+01, -8.94e+01, -9.83e+01, 1.00e+00, 0.00e+00, 0.00e+00, 1.00e-04],
         [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 5.33e+02, 1.69e+02, 1.67e+02
             , 5.04e+02, 4.76e+02, 1.62e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 1.35e+02, 0.00e+00, -1.35e+02, 1.35e+02, 0.00e+00, -1.35e+02, 6.12e+01, 0.00e+00, -6.12e+01
             , 0.00e+00, 4.84e+01, -4.84e+01, 1.35e+02, 2.69e+02, 1.35e+02, 0.00e+00, 2.69e+02, 1.82e+02
             , 1.35e+02, 8.71e+01, 1.95e+02, 1.35e+02, 7.45e+01, -1.32e+02, -2.64e+02, 0.00e+00, -1.32e+02
             , -2.64e+02, -8.59e+01, -1.32e+02, -1.79e+02, -7.36e+01, -1.32e+02, -1.91e+02, 0.00e+00, 1.00e+00
             , 0.00e+00, 1.00e-04],
         [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.69e+02, 5.33e+02, 5.04e+02
             , 1.71e+02, 1.71e+02, 4.76e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 4.84e+01, 0.00e+00, -4.84e+01, 6.12e+01, 0.00e+00, -6.12e+01, 1.35e+02, 0.00e+00, -1.35e+02
             , 0.00e+00, 1.35e+02, -1.35e+02, 6.17e+01, 1.23e+02, 6.17e+01, 1.29e+01, 1.10e+02, 1.96e+02
             , 6.17e+01, -7.30e+01, 1.96e+02, 6.17e+01, -7.30e+01, -4.88e+01, -9.75e+01, 1.29e+01, -4.88e+01
             , -1.10e+02, 8.35e+01, -4.88e+01, -1.81e+02, 8.35e+01, -4.88e+01, -1.81e+02, 0.00e+00, 1.00e+00
             , 0.00e+00, 1.00e-04],
         [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.67e+02, 5.04e+02, 5.33e+02
             , 1.69e+02, 1.71e+02, 5.04e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 4.75e+01, 0.00e+00, -4.75e+01, 6.02e+01, 0.00e+00, -6.02e+01, 1.35e+02, 0.00e+00, -1.35e+02
             , 0.00e+00, 1.35e+02, -1.35e+02, 6.12e+01, 1.22e+02, 6.12e+01, 1.28e+01, 1.10e+02, 1.97e+02
             , 6.12e+01, -7.43e+01, 1.97e+02, 6.12e+01, -7.43e+01, -4.88e+01, -9.75e+01, 1.29e+01, -4.88e+01
             , -1.10e+02, 8.59e+01, -4.88e+01, -1.83e+02, 8.59e+01, -4.88e+01, -1.83e+02, 0.00e+00, 1.00e+00
             , 0.00e+00, 1.00e-04],
         [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 5.04e+02, 1.71e+02, 1.69e+02
             , 5.33e+02, 5.04e+02, 1.67e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 1.35e+02, 0.00e+00, -1.35e+02, 1.35e+02, 0.00e+00, -1.35e+02, 6.17e+01, 0.00e+00, -6.17e+01
             , 0.00e+00, 4.88e+01, -4.88e+01, 1.35e+02, 2.71e+02, 1.35e+02, 0.00e+00, 2.71e+02, 1.84e+02
             , 1.35e+02, 8.71e+01, 1.97e+02, 1.35e+02, 7.43e+01, -1.35e+02, -2.69e+02, 0.00e+00, -1.35e+02
             , -2.69e+02, -8.71e+01, -1.35e+02, -1.82e+02, -7.45e+01, -1.35e+02, -1.95e+02, 0.00e+00, 1.00e+00
             , 0.00e+00, 1.00e-04],
         [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 4.76e+02, 1.71e+02, 1.71e+02
             , 5.04e+02, 5.33e+02, 1.69e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 1.32e+02, 0.00e+00, -1.32e+02, 1.32e+02, 0.00e+00, -1.32e+02, 6.17e+01, 0.00e+00, -6.17e+01
             , 0.00e+00, 4.88e+01, -4.88e+01, 1.35e+02, 2.69e+02, 1.35e+02, 0.00e+00, 2.69e+02, 1.83e+02
             , 1.35e+02, 8.59e+01, 1.96e+02, 1.35e+02, 7.30e+01, -1.35e+02, -2.71e+02, 0.00e+00, -1.35e+02
             , -2.71e+02, -8.71e+01, -1.35e+02, -1.84e+02, -7.43e+01, -1.35e+02, -1.97e+02, 0.00e+00, 1.00e+00
             , 0.00e+00, 1.00e-04],
         [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.62e+02, 4.76e+02, 5.04e+02
             , 1.67e+02, 1.69e+02, 5.33e+02, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 4.64e+01, 0.00e+00, -4.64e+01, 5.87e+01, 0.00e+00, -5.87e+01, 1.32e+02, 0.00e+00, -1.32e+02
             , 0.00e+00, 1.32e+02, -1.32e+02, 6.02e+01, 1.20e+02, 6.02e+01, 1.26e+01, 1.08e+02, 1.95e+02
             , 6.02e+01, -7.45e+01, 1.95e+02, 6.02e+01, -7.45e+01, -4.84e+01, -9.67e+01, 1.28e+01, -4.84e+01
             , -1.10e+02, 8.71e+01, -4.84e+01, -1.84e+02, 8.71e+01, -4.84e+01, -1.84e+02, 0.00e+00, 1.00e+00
             , 0.00e+00, 1.00e-04],
         [0.00e+00, 3.00e+01, 4.91e+01, 0.00e+00, 0.00e+00, 6.70e+01, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 5.33e+02, 1.66e+02, 1.58e+02, 4.76e+02, 4.18e+02, 1.46e+02
             , -1.41e+01, -1.41e+01, -1.41e+01, -1.41e+01, -1.41e+01, -1.41e+01, 1.23e+00, 3.75e+00, 1.23e+00
             , 0.00e+00, -1.97e+00, -1.97e+00, 8.34e+00, 3.55e-14, 8.34e+00, 0.00e+00, 2.13e-14, -2.63e+00
             , 6.01e-01, -2.63e+00, 2.62e+00, 6.74e+00, 2.62e+00, 1.45e+01, -3.55e-14, 0.00e+00, 1.45e+01
             , -3.55e-14, -1.70e+01, -1.26e+01, -1.70e+01, -9.87e+00, -4.26e+00, -9.87e+00, 0.00e+00, 0.00e+00
             , 1.00e+00, 1.00e-04],
         [3.00e+01, 0.00e+00, 0.00e+00, 1.01e+01, -1.01e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 1.66e+02, 5.33e+02, 4.76e+02, 1.70e+02, 1.70e+02, 4.18e+02
             , -1.21e+01, -1.41e+01, -1.21e+01, -1.53e+01, -1.78e+01, -1.53e+01, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, -8.47e-01, -7.11e-15, 4.14e-01, 1.08e+00, 1.08e+00, 2.76e+01
             , 3.59e+01, 2.76e+01, 2.76e+01, 3.59e+01, 2.76e+01, 6.63e-01, -2.66e-15, 1.08e+00, 1.92e+00
             , 1.08e+00, 4.00e+01, 5.46e+01, 4.00e+01, 4.00e+01, 5.46e+01, 4.00e+01, 0.00e+00, 0.00e+00
             , 1.00e+00, 1.00e-04],
         [4.91e+01, 0.00e+00, 0.00e+00, 3.00e+01, 1.01e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 1.58e+02, 4.76e+02, 5.33e+02, 1.66e+02, 1.70e+02, 4.76e+02
             , 1.10e+01, 7.74e+00, 1.10e+01, 5.72e+00, 1.60e+00, 5.72e+00, 8.34e+00, 1.42e-13, 8.34e+00
             , 0.00e+00, 8.34e+00, 8.34e+00, -2.52e+00, 1.42e-14, 1.23e+00, 3.20e+00, 3.20e+00, 1.53e+01
             , 1.53e+01, 1.53e+01, 1.53e+01, 1.53e+01, 1.53e+01, -6.63e-01, 4.44e-15, -1.08e+00, -1.92e+00
             , -1.08e+00, 2.65e+01, 3.48e+01, 2.65e+01, 2.65e+01, 3.48e+01, 2.65e+01, 0.00e+00, 0.00e+00
             , 1.00e+00, 1.00e-04],
         [0.00e+00, 1.01e+01, 3.00e+01, 0.00e+00, 0.00e+00, 4.91e+01, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 4.76e+02, 1.70e+02, 1.66e+02, 5.33e+02, 4.76e+02, 1.58e+02
             , -2.72e+01, -3.55e+01, -2.72e+01, -2.72e+01, -3.55e+01, -2.72e+01, 4.14e-01, 1.26e+00, 4.14e-01
             , 0.00e+00, -6.63e-01, -6.63e-01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.21e+01
             , 1.41e+01, 1.21e+01, 1.53e+01, 1.78e+01, 1.53e+01, 8.34e+00, -2.13e-14, 0.00e+00, 8.34e+00
             , 1.42e-14, -2.63e+00, 6.01e-01, -2.63e+00, 2.62e+00, 6.74e+00, 2.62e+00, 0.00e+00, 0.00e+00
             , 1.00e+00, 1.00e-04],
         [0.00e+00, -1.01e+01, 1.01e+01, 0.00e+00, 0.00e+00, 3.00e+01, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 4.18e+02, 1.70e+02, 1.70e+02, 4.76e+02, 5.33e+02, 1.66e+02
             , -3.94e+01, -5.39e+01, -3.94e+01, -3.94e+01, -5.39e+01, -3.94e+01, -4.14e-01, -1.26e+00, -4.14e-01
             , 0.00e+00, 6.63e-01, 6.63e-01, -8.34e+00, 1.42e-14, -8.34e+00, 3.55e-14, 1.42e-14, 2.65e+01
             , 2.72e+01, 2.65e+01, 2.76e+01, 2.84e+01, 2.76e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 1.21e+01, 1.41e+01, 1.21e+01, 1.53e+01, 1.78e+01, 1.53e+01, 0.00e+00, 0.00e+00
             , 1.00e+00, 1.00e-04],
         [6.70e+01, 0.00e+00, 0.00e+00, 4.91e+01, 3.00e+01, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 1.46e+02, 4.18e+02, 4.76e+02, 1.58e+02, 1.66e+02, 5.33e+02
             , 3.16e+01, 2.72e+01, 3.16e+01, 2.44e+01, 1.88e+01, 2.44e+01, 1.45e+01, -1.78e-13, 1.45e+01
             , 0.00e+00, 1.45e+01, 1.45e+01, -4.12e+00, -5.33e-14, 2.02e+00, 5.25e+00, 5.25e+00, 2.62e+00
             , -5.72e+00, 2.62e+00, 2.62e+00, -5.72e+00, 2.62e+00, -1.97e+00, 0.00e+00, -3.20e+00, -5.72e+00
             , -3.20e+00, 1.21e+01, 1.21e+01, 1.21e+01, 1.21e+01, 1.21e+01, 1.21e+01, 0.00e+00, 0.00e+00
             , 1.00e+00, 1.00e-04],
         [1.63e+02, 1.52e+02, 1.50e+02, 1.64e+02, 1.64e+02, 1.45e+02, 1.35e+02, 4.84e+01, 4.75e+01
             , 1.35e+02, 1.32e+02, 4.64e+01, -1.41e+01, -1.21e+01, 1.10e+01, -2.72e+01, -3.94e+01, 3.16e+01
             , 2.69e+02, 2.31e+02, 1.81e+02, 2.50e+02, 2.10e+02, 1.64e+02, 5.84e+01, 1.90e+01, 2.27e+01
             , 0.00e+00, 3.81e+01, 1.04e+01, 4.60e+01, 8.54e+01, 6.65e+01, 1.69e+01, 1.04e+02, -1.25e+02
             , -1.63e+02, -1.52e+02, -1.05e+02, -1.44e+02, -1.40e+02, -3.71e+01, -8.59e+01, -1.88e+01, -5.75e+01
             , -1.03e+02, -2.26e+02, -2.63e+02, -2.53e+02, -2.07e+02, -2.45e+02, -2.41e+02, 1.00e+00, 3.75e-01
             , -9.38e-02, -1.00e+01],
         [1.72e+02, 1.72e+02, 1.69e+02, 1.73e+02, 1.71e+02, 1.64e+02, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, -1.41e+01, -1.41e+01, 7.74e+00, -3.55e+01, -5.39e+01, 2.72e+01
             , 2.31e+02, 2.52e+02, 2.31e+02, 2.13e+02, 2.30e+02, 2.13e+02, 3.88e+01, 2.14e+01, 3.88e+01
             , 1.98e-14, 2.09e+01, 2.09e+01, 1.70e+01, 0.00e+00, 3.85e+01, 1.80e+01, 1.80e+01, -1.91e+02
             , -2.12e+02, -1.91e+02, -1.73e+02, -1.91e+02, -1.73e+02, 1.98e+01, 4.96e-15, -1.80e+01, -1.56e+00
             , -1.80e+01, -2.06e+02, -2.26e+02, -2.06e+02, -1.89e+02, -2.06e+02, -1.89e+02, 1.00e+00, -0.00e+00
             , -9.38e-02, -1.00e+01],
         [1.63e+02, 1.52e+02, 1.50e+02, 1.64e+02, 1.64e+02, 1.45e+02, -1.35e+02, -4.84e+01, -4.75e+01
             , -1.35e+02, -1.32e+02, -4.64e+01, -1.41e+01, -1.21e+01, 1.10e+01, -2.72e+01, -3.94e+01, 3.16e+01
             , 1.81e+02, 2.31e+02, 2.69e+02, 1.64e+02, 2.10e+02, 2.50e+02, 2.27e+01, 1.90e+01, 5.84e+01
             , 0.00e+00, 1.04e+01, 3.81e+01, -3.93e+01, -8.54e+01, -1.88e+01, 1.90e+01, -6.85e+01, -2.38e+02
             , -2.48e+02, -2.11e+02, -2.26e+02, -2.30e+02, -1.91e+02, 4.88e+01, 8.59e+01, -1.68e+01, 2.84e+01
             , 6.70e+01, -1.67e+02, -1.77e+02, -1.40e+02, -1.55e+02, -1.59e+02, -1.21e+02, 1.00e+00, -3.75e-01
             , -9.38e-02, -1.00e+01],
         [1.18e+02, 1.64e+02, 1.61e+02, 1.19e+02, 1.20e+02, 1.56e+02, 1.35e+02, 6.12e+01, 6.02e+01
             , 1.35e+02, 1.32e+02, 5.87e+01, -1.41e+01, -1.53e+01, 5.72e+00, -2.72e+01, -3.94e+01, 2.44e+01
             , 2.50e+02, 2.13e+02, 1.64e+02, 2.39e+02, 1.98e+02, 1.51e+02, 6.26e+01, 2.04e+01, 1.81e+01
             , 1.98e-14, 4.05e+01, 4.80e+00, 4.72e+01, 8.75e+01, 6.21e+01, 1.36e+01, 9.89e+01, -1.10e+02
             , -1.50e+02, -1.45e+02, -8.79e+01, -1.30e+02, -1.32e+02, -3.62e+01, -8.39e+01, -1.16e+01, -5.12e+01
             , -9.75e+01, -2.06e+02, -2.45e+02, -2.40e+02, -1.85e+02, -2.25e+02, -2.27e+02, 8.75e-01, 3.75e-01
             , -9.38e-02, -1.00e+01],
         [1.09e+02, 1.86e+02, 1.83e+02, 1.11e+02, 1.12e+02, 1.77e+02, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, -1.41e+01, -1.78e+01, 1.60e+00, -3.55e+01, -5.39e+01, 1.88e+01
             , 2.10e+02, 2.30e+02, 2.10e+02, 1.98e+02, 2.17e+02, 1.98e+02, 3.78e+01, 2.31e+01, 3.78e+01
             , 1.98e-14, 1.84e+01, 1.84e+01, 1.80e+01, 1.59e-13, 3.19e+01, 1.24e+01, 1.24e+01, -1.79e+02
             , -1.97e+02, -1.79e+02, -1.60e+02, -1.74e+02, -1.60e+02, 1.89e+01, -5.45e-14, -1.25e+01, 4.79e+00
             , -1.25e+01, -1.89e+02, -2.07e+02, -1.89e+02, -1.70e+02, -1.85e+02, -1.70e+02, 8.75e-01, -0.00e+00
             , -9.38e-02, -1.00e+01],
         [1.18e+02, 1.64e+02, 1.61e+02, 1.19e+02, 1.20e+02, 1.56e+02, -1.35e+02, -6.12e+01, -6.02e+01
             , -1.35e+02, -1.32e+02, -5.87e+01, -1.41e+01, -1.53e+01, 5.72e+00, -2.72e+01, -3.94e+01, 2.44e+01
             , 1.6e+02, 2.13e+02, 2.50e+02, 1.51e+02, 1.98e+02, 2.39e+02, 1.81e+01, 2.04e+01, 6.26e+01
             , 1.98e-14, 4.80e+00, 4.05e+01, -4.03e+01, -8.75e+01, -2.54e+01, 1.14e+01, -7.39e+01, -2.32e+02
             , -2.37e+02, -1.97e+02, -2.19e+02, -2.17e+02, -1.75e+02, 4.77e+01, 8.39e+01, -1.36e+01, 3.27e+01
             , 7.23e+01, -1.56e+02, -1.61e+02, -1.22e+02, -1.43e+02, -1.41e+02, -1.01e+02, 8.75e-01, -3.75e-01
             , -9.38e-02, -1.00e+01],
         [8.07e+00, 5.41e+01, 5.32e+01, 8.13e+00, 8.13e+00, 5.14e+01, 6.12e+01, 1.35e+02, 1.35e+02
             , 6.17e+01, 6.17e+01, 1.32e+02, 1.23e+00, 0.00e+00, 8.34e+00, 4.14e-01, -4.14e-01, 1.45e+01
             , 5.84e+01, 3.88e+01, 2.27e+01, 6.26e+01, 3.78e+01, 1.81e+01, 6.46e+01, 6.67e+00, -2.35e+01
             , 0.00e+00, 5.79e+01, -2.80e+01, 1.98e+01, 4.49e+01, 2.09e+01, 4.64e+00, 4.07e+01, 3.97e+01
             , -1.77e+01, -4.56e+01, 4.63e+01, -1.11e+01, -4.12e+01, -1.98e+01, -3.60e+01, 4.21e+00, -2.08e+01
             , -4.07e+01, -1.56e+00, -5.75e+01, -8.54e+01, 4.79e+00, -5.12e+01, -8.11e+01, 1.25e-01, 3.75e-01
             , -0.00e+00, 1.00e-04],
         [-1.37e+01, 6.31e+01, 6.16e+01, -1.39e+01, -1.39e+01, 5.86e+01, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 3.75e+00, 0.00e+00, 1.42e-13, 1.26e+00, -1.26e+00, -1.78e-13
             , 1.90e+01, 2.14e+01, 1.90e+01, 2.04e+01, 2.31e+01, 2.04e+01, 6.67e+00, 7.73e+00, 6.67e+00
             , 0.00e+00, 1.06e+00, 1.06e+00, 2.75e+00, 1.09e-13, 1.03e+00, -1.46e+00, -1.46e+00, -1.95e+01
             , -2.05e+01, -1.95e+01, -1.39e+01, -1.29e+01, -1.39e+01, 2.48e+00, 0.00e+00, 1.46e+00, 4.21e+00
             , 1.46e+00, -1.80e+01, -1.88e+01, -1.80e+01, -1.25e+01, -1.16e+01, -1.25e+01, 1.25e-01, -0.00e+00
             , -0.00e+00, 1.00e-04],
         [8.07e+00, 5.41e+01, 5.32e+01, 8.13e+00, 8.13e+00, 5.14e+01, -6.12e+01, -1.35e+02, -1.35e+02
             , -6.17e+01, -6.17e+01, -1.32e+02, 1.23e+00, 0.00e+00, 8.34e+00, 4.14e-01, -4.14e-01, 1.45e+01
             , 2.27e+01, 3.88e+01, 5.84e+01, 1.81e+01, 3.78e+01, 6.26e+01, -2.35e+01, 6.67e+00, 6.46e+01
             , 0.00e+00, -2.80e+01, 5.79e+01, -2.50e+01, -4.49e+01, -2.40e+01, -4.21e+00, -4.02e+01, -9.05e+01
             , -6.25e+01, -5.17e+00, -8.61e+01, -5.60e+01, 1.40e+00, 1.62e+01, 3.60e+01, -4.64e+00, 1.52e+01
             , 4.02e+01, -4.94e+01, -2.15e+01, 3.45e+01, -4.50e+01, -1.51e+01, 4.08e+01, 1.25e-01, -3.75e-01
             , -0.00e+00, 1.00e-04],
         [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 2.00e-07, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, -0.00e+00, -0.00e+00
             , -0.00e+00, 1.00e-04],
         [1.97e+01, 8.95e+00, 8.34e+00, 1.99e+01, 1.99e+01, 7.27e+00, 4.84e+01, 1.35e+02, 1.35e+02
             , 4.88e+01, 4.88e+01, 1.32e+02, -1.97e+00, 0.00e+00, 8.34e+00, -6.63e-01, 6.63e-01, 1.45e+01
             , 3.81e+01, 2.09e+01, 1.04e+01, 4.05e+01, 1.84e+01, 4.80e+00, 5.79e+01, 1.06e+00, -2.80e+01
             , 0.00e+00, 5.90e+01, -2.91e+01, 1.37e+01, 3.60e+01, 1.62e+01, 5.63e+00, 3.36e+01, 5.41e+01
             , -4.29e+00, -3.34e+01, 5.31e+01, -3.30e+00, -3.23e+01, -1.73e+01, -2.79e+01, 2.48e+00, -1.98e+01
             , -3.36e+01, 1.98e+01, -3.71e+01, -6.61e+01, 1.89e+01, -3.62e+01, -6.50e+01, -0.00e+00, 3.75e-01
             , -0.00e+00, 1.00e-04],
         [1.97e+01, 8.95e+00, 8.34e+00, 1.99e+01, 1.99e+01, 7.27e+00, -4.84e+01, -1.35e+02, -1.35e+02
             , -4.88e+01, -4.88e+01, -1.32e+02, -1.97e+00, 0.00e+00, 8.34e+00, -6.63e-01, 6.63e-01, 1.45e+01
             , 1.04e+01, 2.09e+01, 3.81e+01, 4.80e+00, 1.84e+01, 4.05e+01, -2.80e+01, 1.06e+00, 5.79e+01
             , 0.00e+00, -2.91e+01, 5.90e+01, -2.23e+01, -3.60e+01, -1.98e+01, -2.48e+00, -3.04e+01, -6.95e+01
             , -4.03e+01, 1.80e+01, -6.83e+01, -3.93e+01, 1.70e+01, 1.06e+01, 2.79e+01, -5.63e+00, 8.11e+00
             , 3.04e+01, -3.82e+01, -9.16e+00, 4.77e+01, -3.71e+01, -8.28e+00, 4.68e+01, -0.00e+00, -3.75e-01
             , -0.00e+00, 1.00e-04],
         [-8.34e+00, 2.20e+01, 2.18e+01, -8.95e+00, -8.34e+00, 2.14e+01, 1.35e+02, 6.17e+01, 6.12e+01
             , 1.35e+02, 1.35e+02, 6.02e+01, 8.34e+00, -8.47e-01, -2.52e+00, 0.00e+00, -8.34e+00, -4.12e+00
             , 4.60e+01, 1.70e+01, -3.93e+01, 4.72e+01, 1.80e+01, -4.03e+01, 1.98e+01, 2.75e+00, -2.50e+01
             , 0.00e+00, 1.37e+01, -2.23e+01, 5.90e+01, 8.81e+01, 5.79e+01, 1.06e+00, 8.69e+01, 5.42e+01
             , 4.05e+01, 1.84e+01, 6.02e+01, 4.33e+01, 1.57e+01, -2.90e+01, -8.54e+01, 1.14e+00, -2.80e+01
             , -8.64e+01, -3.29e+01, -4.63e+01, -6.80e+01, -2.70e+01, -4.36e+01, -7.07e+01, -0.00e+00, 3.75e-01
             , -0.00e+00, 1.00e-04],
         [-3.55e-14, 1.99e-13, -1.14e-13, -1.42e-14, -2.13e-14, 1.71e-13, 2.69e+02, 1.23e+02, 1.22e+02
             , 2.71e+02, 2.69e+02, 1.20e+02, 3.55e-14, -7.11e-15, 1.42e-14, 0.00e+00, 1.42e-14, -5.33e-14
             , 8.54e+01, 0.00e+00, -8.54e+01, 8.75e+01, 1.59e-13, -8.75e+01, 4.49e+01, 1.09e-13, -4.49e+01
             , 0.00e+00, 3.60e+01, -3.60e+01, 8.81e+01, 1.76e+02, 8.81e+01, 2.21e+00, 1.74e+02, 1.24e+02
             , 8.81e+01, 5.24e+01, 1.33e+02, 8.81e+01, 4.36e+01, -8.54e+01, -1.71e+02, 2.14e+00, -8.54e+01
             , -1.73e+02, -5.02e+01, -8.54e+01, -1.20e+02, -4.16e+01, -8.54e+01, -1.29e+02, -0.00e+00, 7.50e-01
             , -0.00e+00, 1.00e-04],
         [5.32e+01, 8.13e+00, 8.07e+00, 5.41e+01, 5.32e+01, 7.94e+00, 1.35e+02, 6.17e+01, 6.12e+01
             , 1.35e+02, 1.35e+02, 6.02e+01, 8.34e+00, 4.14e-01, 1.23e+00, 0.00e+00, -8.34e+00, 2.02e+00
             , 6.65e+01, 3.85e+01, -1.88e+01, 6.21e+01, 3.19e+01, -2.54e+01, 2.09e+01, 1.03e+00, -2.40e+01
             , 0.00e+00, 1.62e+01, -1.98e+01, 5.79e+01, 8.81e+01, 6.46e+01, 6.67e+00, 9.25e+01, 4.19e+01
             , 2.58e+01, 6.14e+00, 4.65e+01, 2.68e+01, 1.97e+00, -2.80e+01, -8.54e+01, -4.44e+00, -3.46e+01
             , -9.19e+01, -5.05e+01, -6.64e+01, -8.57e+01, -4.60e+01, -6.53e+01, -8.98e+01, 1.25e-01, 3.75e-01
             , -0.00e+00, 1.00e-04],
         [4.49e+01, -1.17e+01, -1.16e+01, 4.52e+01, 4.49e+01, -1.15e+01, 0.00e+00, 1.29e+01, 1.28e+01
             , 0.00e+00, 0.00e+00, 1.26e+01, 0.00e+00, 1.08e+00, 3.20e+00, 0.00e+00, 3.55e-14, 5.25e+00
             , 1.69e+01, 1.80e+01, 1.90e+01, 1.36e+01, 1.24e+01, 1.14e+01, 4.64e+00, -1.46e+00, -4.21e+00
             , 0.00e+00, 5.63e+00, -2.48e+00, 1.06e+00, 2.21e+00, 6.67e+00, 7.73e+00, 5.52e+00, -5.63e+00
             , -1.12e+01, -1.37e+01, -6.62e+00, -1.27e+01, -1.54e+01, 1.14e+00, 2.14e+00, -3.29e+00, -4.44e+00
             , -5.43e+00, -1.10e+01, -1.65e+01, -1.89e+01, -1.20e+01, -1.79e+01, -2.06e+01, 1.25e-01, -0.00e+00
             , -0.00e+00, 1.00e-04],
         [4.49e+01, -1.17e+01, -1.16e+01, 4.52e+01, 4.49e+01, -1.15e+01, 2.69e+02, 1.10e+02, 1.10e+02
             , 2.71e+02, 2.69e+02, 1.08e+02, 2.13e-14, 1.08e+00, 3.20e+00, 0.00e+00, 1.42e-14, 5.25e+00
             , 1.04e+02, 1.80e+01, -6.85e+01, 9.89e+01, 1.24e+01, -7.39e+01, 4.07e+01, -1.46e+00, -4.02e+01
             , 0.00e+00, 3.36e+01, -3.04e+01, 8.69e+01, 1.74e+02, 9.25e+01, 5.52e+00, 1.79e+02, 1.08e+02
             , 7.47e+01, 4.45e+01, 1.15e+02, 7.32e+01, 3.48e+01, -8.64e+01, -1.73e+02, -5.43e+00, -9.19e+01
             , -1.78e+02, -7.13e+01, -1.04e+02, -1.34e+02, -6.43e+01, -1.05e+02, -1.43e+02, 1.25e-01, 7.50e-01
             , -0.00e+00, 1.00e-04],
         [-9.64e+01, -1.56e+02, -1.55e+02, -9.83e+01, -9.95e+01, -1.53e+02, 1.82e+02, 1.96e+02, 1.97e+02
             , 1.84e+02, 1.83e+02, 1.95e+02, -2.63e+00, 2.76e+01, 1.53e+01, 1.21e+01, 2.65e+01, 2.62e+00
             , -1.25e+02, -1.91e+02, -2.38e+02, -1.10e+02, -1.79e+02, -2.32e+02, 3.97e+01, -1.95e+01, -9.05e+01
             , 0.00e+00, 5.41e+01, -6.95e+01, 5.42e+01, 1.24e+02, 4.19e+01, -5.63e+00, 1.08e+02, 2.88e+02
             , 2.34e+02, 1.64e+02, 2.74e+02, 2.15e+02, 1.43e+02, -6.58e+01, -1.13e+02, 1.60e+01, -5.34e+01
             , -1.07e+02, 1.79e+02, 1.25e+02, 5.62e+01, 1.65e+02, 1.06e+02, 3.57e+01, -8.75e-01, 7.50e-01
             , 9.38e-02, 1.00e+01],
         [-1.16e+02, -1.64e+02, -1.64e+02, -1.18e+02, -1.19e+02, -1.61e+02, 1.35e+02, 6.17e+01, 6.12e+01
             , 1.35e+02, 1.35e+02, 6.02e+01, 6.01e-01, 3.59e+01, 1.53e+01, 1.41e+01, 2.72e+01, -5.72e+00
             , -1.63e+02, -2.12e+02, -2.48e+02, -1.50e+02, -1.97e+02, -2.37e+02, -1.77e+01, -2.05e+01, -6.25e+01
             , 0.00e+00, -4.29e+00, -4.03e+01, 4.05e+01, 8.81e+01, 2.58e+01, -1.12e+01, 7.47e+01, 2.34e+02
             , 2.39e+02, 1.98e+02, 2.21e+02, 2.18e+02, 1.76e+02, -4.85e+01, -8.54e+01, 1.36e+01, -3.36e+01
             , -7.39e+01, 1.58e+02, 1.63e+02, 1.23e+02, 1.45e+02, 1.43e+02, 1.02e+02, -8.75e-01, 3.75e-01
             , 9.38e-02, 1.00e+01],
         [-9.64e+01, -1.56e+02, -1.55e+02, -9.83e+01, -9.95e+01, -1.53e+02, 8.71e+01, -7.30e+01, -7.43e+01
             , 8.71e+01, 8.59e+01, -7.45e+01, -2.63e+00, 2.76e+01, 1.53e+01, 1.21e+01, 2.65e+01, 2.62e+00
             , -1.52e+02, -1.91e+02, -2.11e+02, -1.45e+02, -1.79e+02, -1.97e+02, -4.56e+01, -1.95e+01, -5.17e+00
             , 0.00e+00, -3.34e+01, 1.80e+01, 1.84e+01, 5.24e+01, 6.14e+00, -1.37e+01, 4.45e+01, 1.64e+02
             , 1.98e+02, 2.17e+02, 1.52e+02, 1.79e+02, 1.93e+02, -3.79e+01, -5.74e+01, 7.93e+00, -2.55e+01
             , -4.35e+01, 1.19e+02, 1.53e+02, 1.72e+02, 1.07e+02, 1.34e+02, 1.49e+02, -8.75e-01, -0.00e+00
             , 9.38e-02, 1.00e+01],
         [-1.08e+02, -1.11e+02, -1.10e+02, -1.10e+02, -1.11e+02, -1.08e+02, 1.95e+02, 1.96e+02, 1.97e+02
             , 1.97e+02, 1.96e+02, 1.95e+02, 2.62e+00, 2.76e+01, 1.53e+01, 1.53e+01, 2.76e+01, 2.62e+00
             , -1.05e+02, -1.73e+02, -2.26e+02, -8.79e+01, -1.60e+02, -2.19e+02, 4.63e+01, -1.39e+01, -8.61e+01
             , 0.00e+00, 5.31e+01, -6.83e+01, 6.02e+01, 1.33e+02, 4.65e+01, -6.62e+00, 1.15e+02, 2.74e+02
             , 2.21e+02, 1.52e+02, 2.67e+02, 2.07e+02, 1.34e+02, -6.83e+01, -1.21e+02, 1.78e+01, -5.44e+01
             , -1.15e+02, 1.57e+02, 1.05e+02, 3.69e+01, 1.51e+02, 9.15e+01, 1.96e+01, -7.50e-01, 7.50e-01
             , 9.38e-02, 1.00e+01],
         [-1.29e+02, -1.03e+02, -1.01e+02, -1.32e+02, -1.33e+02, -9.95e+01, 1.35e+02, 6.17e+01, 6.12e+01
             , 1.35e+02, 1.35e+02, 6.02e+01, 6.74e+00, 3.59e+01, 1.53e+01, 1.78e+01, 2.84e+01, -5.72e+00
             , -1.44e+02, -1.91e+02, -2.30e+02, -1.30e+02, -1.74e+02, -2.17e+02, -1.11e+01, -1.29e+01, -5.60e+01
             , 0.00e+00, -3.30e+00, -3.93e+01, 4.33e+01, 8.81e+01, 2.68e+01, -1.27e+01, 7.32e+01, 2.15e+02
             , 2.18e+02, 1.79e+02, 2.07e+02, 2.06e+02, 1.62e+02, -4.60e+01, -8.54e+01, 1.50e+01, -2.94e+01
             , -7.25e+01, 1.40e+02, 1.44e+02, 1.05e+02, 1.33e+02, 1.32e+02, 8.91e+01, -7.50e-01, 3.75e-01
             , 9.38e-02, 1.00e+01],
         [-1.08e+02, -1.11e+02, -1.10e+02, -1.10e+02, -1.11e+02, -1.08e+02, 7.45e+01, -7.30e+01, -7.43e+01
             , 7.43e+01, 7.30e+01, -7.45e+01, 2.62e+00, 2.76e+01, 1.53e+01, 1.53e+01, 2.76e+01, 2.62e+00
             , -1.40e+02, -1.73e+02, -1.91e+02, -1.32e+02, -1.60e+02, -1.75e+02, -4.12e+01, -1.39e+01, 1.40e+00
             , 0.00e+00, -3.23e+01, 1.70e+01, 1.57e+01, 4.36e+01, 1.97e+00, -1.54e+01, 3.48e+01, 1.43e+02
             , 1.76e+02, 1.93e+02, 1.34e+02, 1.62e+02, 1.78e+02, -3.23e+01, -4.93e+01, 8.93e+00, -1.84e+01
             , -3.37e+01, 1.08e+02, 1.41e+02, 1.58e+02, 9.94e+01, 1.28e+02, 1.43e+02, -7.50e-01, -0.00e+00
             , 9.38e-02, 1.00e+01],
         [7.27e+00, 1.99e+01, 1.99e+01, 8.34e+00, 8.95e+00, 1.97e+01, -1.32e+02, -4.88e+01, -4.88e+01
             , -1.35e+02, -1.35e+02, -4.84e+01, 1.45e+01, 6.63e-01, -6.63e-01, 8.34e+00, 0.00e+00, -1.97e+00
             , -3.71e+01, 1.98e+01, 4.88e+01, -3.62e+01, 1.89e+01, 4.77e+01, -1.98e+01, 2.48e+00, 1.62e+01
             , 0.00e+00, -1.73e+01, 1.06e+01, -2.90e+01, -8.54e+01, -2.80e+01, 1.14e+00, -8.64e+01, -6.58e+01
             , -4.85e+01, -3.79e+01, -6.83e+01, -4.60e+01, -3.23e+01, 5.90e+01, 8.81e+01, 1.06e+00, 5.79e+01
             , 8.69e+01, 2.09e+01, 3.81e+01, 4.86e+01, 1.84e+01, 4.05e+01, 5.42e+01, -0.00e+00, -3.75e-01
             , -0.00e+00, 1.00e-04],
         [-1.78e-14, -8.53e-14, -1.42e-13, -2.13e-14, -1.42e-14, 0.00e+00, -2.64e+02, -9.75e+01, -9.75e+01
             , -2.69e+02, -2.71e+02, -9.67e+01, -3.55e-14, -2.66e-15, 4.44e-15, -2.13e-14, 0.00e+00, 0.00e+00
             , -8.59e+01, -7.93e-14, 8.59e+01, -8.39e+01, -5.95e-14, 8.39e+01, -3.60e+01, 0.00e+00, 3.60e+01
             , 0.00e+00, -2.79e+01, 2.79e+01, -8.54e+01, -1.71e+02, -8.54e+01, 2.14e+00, -1.73e+02, -1.13e+02
             , -8.54e+01, -5.74e+01, -1.21e+02, -8.54e+01, -4.93e+01, 8.81e+01, 1.76e+02, 2.21e+00, 8.81e+01
             , 1.74e+02, 6.04e+01, 8.81e+01, 1.16e+02, 5.24e+01, 8.81e+01, 1.24e+02, -0.00e+00, -7.50e-01
             , -0.00e+00, 1.00e-04],
         [-4.41e+01, 1.17e+01, 1.17e+01, -4.49e+01, -4.52e+01, 1.16e+01, 0.00e+00, 1.29e+01, 1.29e+01
             , 0.00e+00, 0.00e+00, 1.28e+01, 0.00e+00, 1.08e+00, -1.08e+00, 0.00e+00, 0.00e+00, -3.20e+00
             , -1.88e+01, -1.80e+01, -1.68e+01, -1.16e+01, -1.25e+01, -1.36e+01, 4.21e+00, 1.46e+00, -4.64e+00
             , 0.00e+00, 2.48e+00, -5.63e+00, 1.14e+00, 2.14e+00, -4.44e+00, -3.29e+00, -5.43e+00, 1.60e+01
             , 1.36e+01, 7.93e+00, 1.78e+01, 1.50e+01, 8.93e+00, 1.06e+00, 2.21e+00, 7.73e+00, 6.67e+00
             , 5.52e+00, 2.14e+01, 1.90e+01, 1.34e+01, 2.31e+01, 2.04e+01, 1.44e+01, -1.25e-01, -0.00e+00
             , -0.00e+00, 1.00e-04],
         [-5.14e+01, 3.38e+01, 3.38e+01, -5.32e+01, -5.41e+01, 3.35e+01, -1.32e+02, -4.88e+01, -4.88e+01
             , -1.35e+02, -1.35e+02, -4.84e+01, 1.45e+01, 1.92e+00, -1.92e+00, 8.34e+00, 0.00e+00, -5.72e+00
             , -5.75e+01, -1.56e+00, 2.84e+01, -5.12e+01, 4.79e+00, 3.27e+01, -2.08e+01, 4.21e+00, 1.52e+01
             , 0.00e+00, -1.98e+01, 8.11e+00, -2.80e+01, -8.54e+01, -3.46e+01, -4.44e+00, -9.19e+01, -5.34e+01
             , -3.36e+01, -2.55e+01, -5.44e+01, -2.94e+01, -1.84e+01, 5.79e+01, 8.81e+01, 6.67e+00, 6.46e+01
             , 9.25e+01, 3.88e+01, 5.84e+01, 6.65e+01, 3.78e+01, 6.26e+01, 7.35e+01, -1.25e-01, -3.75e-01
             , -0.00e+00, 1.00e-04],
         [-4.41e+01, 1.17e+01, 1.17e+01, -4.49e+01, -4.52e+01, 1.16e+01, -2.64e+02, -1.10e+02, -1.10e+02
             , -2.69e+02, -2.71e+02, -1.10e+02, -3.55e-14, 1.08e+00, -1.08e+00, 1.42e-14, 0.00e+00, -3.20e+00
             , -1.03e+02, -1.80e+01, 6.70e+01, -9.75e+01, -1.25e+01, 7.23e+01, -4.07e+01, 1.46e+00, 4.02e+01
             , 0.00e+00, -3.36e+01, 3.04e+01, -8.64e+01, -1.73e+02, -9.19e+01, -5.43e+00, -1.78e+02, -1.07e+02
             , -7.39e+01, -4.35e+01, -1.15e+02, -7.25e+01, -3.37e+01, 8.69e+01, 1.74e+02, 5.52e+00, 9.25e+01
             , 1.79e+02, 7.16e+01, 1.05e+02, 1.35e+02, 6.45e+01, 1.06e+02, 1.45e+02, -1.25e-01, -7.50e-01
             , -0.00e+00, 1.00e-04],
         [-1.38e+02, -1.44e+02, -1.44e+02, -1.41e+02, -1.43e+02, -1.43e+02, -8.59e+01, 8.35e+01, 8.59e+01
             , -8.71e+01, -8.71e+01, 8.71e+01, -1.70e+01, 4.00e+01, 2.65e+01, -2.63e+00, 1.21e+01, 1.21e+01
             , -2.26e+02, -2.06e+02, -1.67e+02, -2.06e+02, -1.89e+02, -1.56e+02, -1.56e+00, -1.80e+01, -4.94e+01
             , 4.96e-15, 1.98e+01, -3.82e+01, -3.29e+01, -5.02e+01, -5.05e+01, -1.10e+01, -7.13e+01, 1.79e+02
             , 1.58e+02, 1.19e+02, 1.57e+02, 1.40e+02, 1.08e+02, 2.09e+01, 6.04e+01, 2.14e+01, 3.88e+01
             , 7.16e+01, 2.52e+02, 2.31e+02, 1.91e+02, 2.30e+02, 2.13e+02, 1.80e+02, -1.00e+00, -0.00e+00
             , 9.38e-02, 1.00e+01],
         [-1.57e+02, -1.51e+02, -1.53e+02, -1.61e+02, -1.63e+02, -1.52e+02, -1.32e+02, -4.88e+01, -4.88e+01
             , -1.35e+02, -1.35e+02, -4.84e+01, -1.26e+01, 5.46e+01, 3.48e+01, 6.01e-01, 1.41e+01, 1.21e+01
             , -2.63e+02, -2.26e+02, -1.77e+02, -2.45e+02, -2.07e+02, -1.61e+02, -5.75e+01, -1.88e+01, -2.15e+01
             , 4.96e-15, -3.71e+01, -9.16e+00, -4.63e+01, -8.54e+01, -6.64e+01, -1.65e+01, -1.04e+02, 1.25e+02
             , 1.63e+02, 1.53e+02, 1.05e+02, 1.44e+02, 1.41e+02, 3.81e+01, 8.81e+01, 1.90e+01, 5.84e+01
             , 1.05e+02, 2.31e+02, 2.69e+02, 2.58e+02, 2.10e+02, 2.50e+02, 2.46e+02, -1.00e+00, -3.75e-01
             , 9.38e-02, 1.00e+01],
         [-1.38e+02, -1.44e+02, -1.44e+02, -1.41e+02, -1.43e+02, -1.43e+02, -1.79e+02, -1.81e+02, -1.83e+02
             , -1.82e+02, -1.84e+02, -1.84e+02, -1.70e+01, 4.00e+01, 2.65e+01, -2.63e+00, 1.21e+01, 1.21e+01
             , -2.53e+02, -2.06e+02, -1.40e+02, -2.40e+02, -1.89e+02, -1.22e+02, -8.54e+01, -1.80e+01, 3.45e+01
             , 4.96e-15, -6.61e+01, 4.77e+01, -6.80e+01, -1.20e+02, -8.57e+01, -1.89e+01, -1.34e+02, 5.62e+01
             , 1.23e+02, 1.72e+02, 3.69e+01, 1.05e+02, 1.58e+02, 4.86e+01, 1.16e+02, 1.34e+01, 6.65e+01
             , 1.35e+02, 1.91e+02, 2.58e+02, 3.07e+02, 1.72e+02, 2.41e+02, 2.94e+02, -1.00e+00, -7.50e-01
             , 9.38e-02, 1.00e+01],
         [-1.49e+02, -9.99e+01, -9.95e+01, -1.53e+02, -1.55e+02, -9.83e+01, -7.36e+01, 8.35e+01, 8.59e+01
             , -7.45e+01, -7.43e+01, 8.71e+01, -9.87e+00, 4.00e+01, 2.65e+01, 2.62e+00, 1.53e+01, 1.21e+01
             , -2.07e+02, -1.89e+02, -1.55e+02, -1.85e+02, -1.70e+02, -1.43e+02, 4.79e+00, -1.25e+01, -4.50e+01
             , 4.96e-15, 1.89e+01, -3.71e+01, -2.70e+01, -4.16e+01, -4.60e+01, -1.20e+01, -6.43e+01, 1.65e+02
             , 1.45e+02, 1.07e+02, 1.51e+02, 1.33e+02, 9.94e+01, 1.84e+01, 5.24e+01, 2.31e+01, 3.78e+01
             , 6.45e+01, 2.30e+02, 2.10e+02, 1.72e+02, 2.17e+02, 1.98e+02, 1.64e+02, -8.75e-01, -0.00e+00
             , 9.38e-02, 1.00e+01],
         [-1.70e+02, -9.26e+01, -9.11e+01, -1.74e+02, -1.77e+02, -8.94e+01, -1.32e+02, -4.88e+01, -4.88e+01
             , -1.35e+02, -1.35e+02, -4.84e+01, -4.26e+00, 5.46e+01, 3.48e+01, 6.74e+00, 1.78e+01, 1.21e+01
             , -2.45e+02, -2.06e+02, -1.59e+02, -2.25e+02, -1.85e+02, -1.41e+02, -5.12e+01, -1.16e+01, -1.51e+01
             , 4.96e-15, -3.62e+01, -8.28e+00, -4.36e+01, -8.54e+01, -6.53e+01, -1.79e+01, -1.05e+02, 1.06e+02
             , 1.43e+02, 1.34e+02, 9.15e+01, 1.32e+02, 1.28e+02, 4.05e+01, 8.81e+01, 2.04e+01, 6.26e+01
             , 1.06e+02, 2.13e+02, 2.50e+02, 2.41e+02, 1.98e+02, 2.39e+02, 2.34e+02, -8.75e-01, -3.75e-01
             , 9.38e-02, 1.00e+01],
         [-1.49e+02, -9.99e+01, -9.95e+01, -1.53e+02, -1.55e+02, -9.83e+01, -1.91e+02, -1.81e+02, -1.83e+02
             , -1.95e+02, -1.97e+02, -1.84e+02, -9.87e+00, 4.00e+01, 2.65e+01, 2.62e+00, 1.53e+01, 1.21e+01
             , -2.41e+02, -1.89e+02, -1.21e+02, -2.27e+02, -1.70e+02, -1.01e+02, -8.11e+01, -1.25e+01, 4.08e+01
             , 4.96e-15, -6.50e+01, 4.68e+01, -7.07e+01, -1.29e+02, -8.98e+01, -2.06e+01, -1.43e+02, 3.57e+01
             , 1.02e+02, 1.49e+02, 1.96e+01, 8.91e+01, 1.43e+02, 5.42e+01, 1.24e+02, 1.44e+01, 7.35e+01
             , 1.45e+02, 1.80e+02, 2.46e+02, 2.94e+02, 1.64e+02, 2.34e+02, 2.88e+02, -8.75e-01, -7.50e-01
             , 9.38e-02, 1.00e+01],
         [1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 1.00e+00, 1.00e+00, 1.00e+00, 8.75e-01, 8.75e-01, 8.75e-01, 1.25e-01, 1.25e-01, 1.25e-01
             , -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, 1.25e-01, 1.25e-01, 1.25e-01, -8.75e-01
             , -8.75e-01, -8.75e-01, -7.50e-01, -7.50e-01, -7.50e-01, -0.00e+00, -0.00e+00, -1.25e-01, -1.25e-01
             , -1.25e-01, -1.00e+00, -1.00e+00, -1.00e+00, -8.75e-01, -8.75e-01, -8.75e-01, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00],
         [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 1.00e+00, 1.00e+00
             , 1.00e+00, 1.00e+00, 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 3.75e-01, -0.00e+00, -3.75e-01, 3.75e-01, -0.00e+00, -3.75e-01, 3.75e-01, -0.00e+00, -3.75e-01
             , -0.00e+00, 3.75e-01, -3.75e-01, 3.75e-01, 7.50e-01, 3.75e-01, -0.00e+00, 7.50e-01, 7.50e-01
             , 3.75e-01, -0.00e+00, 7.50e-01, 3.75e-01, -0.00e+00, -3.75e-01, -7.50e-01, -0.00e+00, -3.75e-01
             , -7.50e-01, -0.00e+00, -3.75e-01, -7.50e-01, -0.00e+00, -3.75e-01, -7.50e-01, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00],
         [0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00, 0.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00, 1.00e+00
             , -9.38e-02, -9.38e-02, -9.38e-02, -9.38e-02, -9.38e-02, -9.38e-02, -0.00e+00, -0.00e+00, -0.00e+00
             , -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00, 9.38e-02
             , 9.38e-02, 9.38e-02, 9.38e-02, 9.38e-02, 9.38e-02, -0.00e+00, -0.00e+00, -0.00e+00, -0.00e+00
             , -0.00e+00, 9.38e-02, 9.38e-02, 9.38e-02, 9.38e-02, 9.38e-02, 9.38e-02, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00],
         [1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04
             , 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04
             , -1.00e+01, -1.00e+01, -1.00e+01, -1.00e+01, -1.00e+01, -1.00e+01, 1.00e-04, 1.00e-04, 1.00e-04
             , 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e+01
             , 1.00e+01, 1.00e+01, 1.00e+01, 1.00e+01, 1.00e+01, 1.00e-04, 1.00e-04, 1.00e-04, 1.00e-04
             , 1.00e-04, 1.00e+01, 1.00e+01, 1.00e+01, 1.00e+01, 1.00e+01, 1.00e+01, 0.00e+00, 0.00e+00
             , 0.00e+00, 0.00e+00], ],

    )
    pass

    5 + 5

    return foo
