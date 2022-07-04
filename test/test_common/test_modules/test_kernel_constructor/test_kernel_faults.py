import numpy as np

from gempy_engine.API.interp_single._interp_scalar_field import _input_preprocess
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.grid import Grid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor, StackRelationType, TensorsStructure, StacksStructure
from gempy_engine.core.data.internal_structs import SolverInput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import DualContouringMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.modules.kernel_constructor.kernel_constructor_interface import yield_covariance
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level, ValueType
from test import helper_functions_pyvista
from test.conftest import TEST_SPEED, pykeops_enabled
from test.helper_functions import plot_block, plot_2d_scalar_y_direction

PLOT = False


# noinspection PyUnreachableCode
def test_graben_fault_model(graben_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = graben_fault_model

    options.compute_scalar_gradient = False
    options.dual_contouring = False

    options.number_octree_levels = 1
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output
    if True:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)


# noinspection PyUnreachableCode
def test_graben_fault_model_thickness(graben_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions
    
    interpolation_input, structure, options = graben_fault_model

    options.compute_scalar_gradient = False
    options.dual_contouring = True

    fault_data: FaultsData = FaultsData.from_user_input(thickness=.2)
    fault_data2: FaultsData = FaultsData.from_user_input(thickness=.2)
    structure.stack_structure.faults_input_data = [fault_data, fault_data2, None]

    options.number_octree_levels = 4
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output
    
    if True:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)
        
    if True:
        # plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)
        # plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)

    if True:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes
        )


# noinspection PyUnreachableCode
def test_graben_fault_model_offset(graben_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions
    
    interpolation_input, structure, options = graben_fault_model

    options.compute_scalar_gradient = False
    options.dual_contouring = False

    fault_data: FaultsData = FaultsData.from_user_input(thickness=None, offset=50)
    structure.stack_structure.faults_input_data = [fault_data, None, None]

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output

    if True:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

    if True:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.values_block)
    
    if True:
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)

def test_one_fault_model_pykeops(one_fault_model):
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


# noinspection PyUnreachableCode
def test_one_fault_model(one_fault_model,  n_oct_levels=8):
    """
    300 MB 4 octree levels and no gradient
    
    """
    
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model
    
    options.compute_scalar_gradient = False
    options.dual_contouring = False
    options.dual_contouring_masking_options = DualContouringMaskingOptions.DISJOINT

    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)

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

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        
    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.mask_component)

    if False:
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)

    if False:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes
        )


def test_one_fault_model_thickness(one_fault_model, n_oct_levels=3):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = one_fault_model

    fault_data: FaultsData = FaultsData.from_user_input(thickness=.5)
    structure.stack_structure.faults_input_data = [fault_data, None, None]
    options.dual_contouring = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.DISJOINT

    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    # TODO: Grab second scalar and create fault kernel
    outputs: list[OctreeLevel] = solutions.octrees_output

    if False:
        plot_block_and_input_2d(0, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(1, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, ValueType.squeeze_mask)
        
    if True:
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


def plot_block_and_input_2d(stack_number, interpolation_input, outputs: list[OctreeLevel], structure: StacksStructure,
                            value_type=ValueType.ids):
    from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level

    regular_grid_scalar = get_regular_grid_value_for_level(outputs, value_type=value_type, scalar_n=stack_number)
    grid: Grid = outputs[-1].grid_centers

    structure.stack_number = stack_number
    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(interpolation_input, structure)
    plot_block(regular_grid_scalar, grid.regular_grid, interpolation_input_i)


def covariance_for_one_fault_model_from_gempy_v2():
    one_fault_covariance = np.load("one_fault_test_data.npy")
    return one_fault_covariance
