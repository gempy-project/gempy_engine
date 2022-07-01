import numpy as np
import pytest

from gempy_engine.API.interp_single._interp_scalar_field import _input_preprocess
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.grid import Grid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, StackRelationType, TensorsStructure, StacksStructure
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import DualContouringMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.modules import kernel_constructor
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level, ValueType
from test import helper_functions_pyvista
from test.conftest import TEST_SPEED, pykeops_enabled
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
    n_dim    = 1                           
    n_uni_eq = options.n_uni_eq           # * Number of equations. This should be how many graben_data are active
    n_faults = 1                           
    z        = np.zeros((ori_size, n_dim))
    z2       = np.zeros((n_uni_eq, n_dim))
    z3       = np.ones((n_faults,  n_dim))
    # Degree 1
    return np.vstack((z, matrix_val, z2, z3))


def test_creating_several_faults_kernel_with_dummy_data(simple_model_2):
    from gempy_engine.modules.data_preprocess._input_preparation import orientations_preprocess

    matrix_val   = simple_model_2[0]
    orientations = simple_model_2[1]
    options      = simple_model_2[2]

    input_data_descriptor: InputDataDescriptor =                 simple_model_2[3]

    ori_internals = orientations_preprocess(orientations)

    # ! For each fault we are going to need to compute everything again
    sp_points_fault_values = np.array([[0, 0, 1, 1, 0, 0, 1],
                                       [0, 1, 1, 1, 1, 0, 1],
                                       [0, 0, 1, 0, 0, 1, 1]]).T

    # region Preprocess
    # * This dadata has to go to SolverInput

    partitions_bool    = input_data_descriptor.tensors_structure.partitions_bool                  
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

    matrix_val        = rest_matrix_val.reshape(-1,             n_faults)                                
    fault_vector_rest = foo(matrix_val,                         options,   ori_internals)
    fault_vector_ref  = foo(ref_matrix_val_repeated.reshape(-1, n_faults), options,       ori_internals)

    fault_vector_ref_i  = fault_vector_ref[:,     None, :]
    fault_vector_rest_i = fault_vector_rest[:,    None, :]
    fault_vector_ref_j  = fault_vector_ref[None,  :,    :]
    fault_vector_rest_j = fault_vector_rest[None, :,    :]


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


def test_one_fault_model_pykeops(one_fault_model, n_oct_levels=3):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    i = 1
    structure.stack_structure.stack_number = i
    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(
        interpolation_input, structure.stack_structure)

    tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(structure, i)
    
    xyz_lvl0, ori_internal, sp_internal, fault_internal = _input_preprocess(tensor_struct_i, interpolation_input_i)
    solver_input = SolverInput(
        sp_internal=sp_internal,
        ori_internal=ori_internal,
        fault_internal=fault_internal,
        options=options.kernel_options)

    A_matrix = yield_covariance(solver_input)
    array_to_cache = A_matrix
    
    if pykeops_enabled is False:
        cache_array = np.save("cached_array", array_to_cache)
    cached_array = np.load("cached_array.npy")
    foo = A_matrix.sum(0).T - cached_array.sum(0)
    print(cached_array)


def test_one_fault_model(one_fault_model,  n_oct_levels=7):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    options.dual_contouring = False
    options.dual_contouring_masking_options = DualContouringMaskingOptions.RAW

    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    # TODO: Grab second scalar and create fault kernel
    outputs: list[OctreeLevel] = solutions.octrees_output

    array_to_cache = outputs[-1].outputs_centers[1].exported_fields.debug

    if pykeops_enabled is False:
        cache_array = np.save("cached_array", array_to_cache)
    cached_array = np.load("cached_array.npy")

    if False:  # * This is in case we need to compare the covariance matrices
        last_cov = outputs[-1].outputs_centers.exported_fields.debug
        gempy_v2_cov = covariance_for_one_fault_model_from_gempy_v2()
        diff = last_cov - gempy_v2_cov

    if False:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

    if PLOT or False:
        grid = interpolation_input.grid.regular_grid

        plot_block(outputs[0].values_block, grid)
        plot_block(outputs[0].squeezed_mask_array, grid)
        print(outputs[0].squeezed_mask_array)
        plot_block(outputs[1].values_block, grid)
        plot_block(outputs[2].values_block, grid)

    if False:
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

    if False:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes
        )


def plot_scalar_and_input_2d(foo, interpolation_input, outputs:list[OctreeLevel], structure: StacksStructure):
    structure.stack_number = foo

    regular_grid_scalar = get_regular_grid_value_for_level(outputs, value_type=ValueType.scalar, scalar_n=foo)
    grid: Grid = outputs[-1].grid_centers
    
    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(interpolation_input, structure)
    plot_2d_scalar_y_direction(interpolation_input_i, regular_grid_scalar, grid.regular_grid)


def plot_block_and_input_2d(stack_number, interpolation_input, outputs: list[OctreeLevel], structure: StacksStructure):
    from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level

    regular_grid_scalar = get_regular_grid_value_for_level(outputs).astype("int8")
    grid: Grid = outputs[-1].grid_centers

    structure.stack_number = stack_number
    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(interpolation_input, structure)
    plot_block(regular_grid_scalar, grid.regular_grid, interpolation_input_i)


def covariance_for_one_fault_model_from_gempy_v2():
    one_fault_covariance = np.load("one_fault_test_data.npy")
    return one_fault_covariance
