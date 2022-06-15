import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.interp_output import InterpOutput
from test import helper_functions_pyvista
from test.conftest import TEST_SPEED

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
    n_uni_eq = options.n_uni_eq  # * Number of equations. This should be how many faults are active
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
        n_uni_eq = options.n_uni_eq  # * Number of equations. This should be how many faults are active
        
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
    selector = (foo.sel_ui * (foo.sel_vj + 1)).sum(-1) # * This has to be a list

    # endregion

    fault_matrix = selector * bar

    pass


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_fault_kernel(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    # TODO: Grab second scalar and create fault kernel
    output: InterpOutput = solutions.octrees_output[0].outputs_centers[1]

    all_values = output.scalar_fields._values_block
    sp_val = all_values[0, -2:]

    if PLOT or True:
        helper_functions_pyvista.plot_pyvista(solutions.octrees_output, dc_meshes=solutions.dc_meshes)
