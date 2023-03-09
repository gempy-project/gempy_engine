from gempy_engine import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import DualContouringMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.modules.octrees_topology.octrees_topology_interface import ValueType

from test import helper_functions_pyvista
from test.conftest import plot_pyvista
from test.helper_functions import plot_scalar_and_input_2d, plot_block_and_input_2d


def test_graben_fault_model(graben_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = graben_fault_model

    options.compute_scalar_gradient = False
    options.dual_contouring = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.RAW

    options.number_octree_levels = 3
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output
    if False:
        plot_scalar_and_input_2d(0, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(1, interpolation_input, outputs, structure.stack_structure)
        plot_scalar_and_input_2d(2, interpolation_input, outputs, structure.stack_structure)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)

    if plot_pyvista or False:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes
        )


def test_graben_fault_model_thickness(graben_fault_model, n_octree_levels=3):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = graben_fault_model

    options.compute_scalar_gradient = False
    options.dual_contouring = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.RAW

    fault_data: FaultsData = FaultsData.from_user_input(thickness=.2)
    fault_data2: FaultsData = FaultsData.from_user_input(thickness=.2)
    structure.stack_structure.faults_input_data = [fault_data, fault_data2, None]

    options.number_octree_levels = n_octree_levels
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

    if plot_pyvista or False:
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes
        )


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
