from gempy_engine import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import MeshExtractionMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.output.blocks_value_type import ValueType

from gempy_engine.plugins.plotting import helper_functions_pyvista
from tests.conftest import plot_pyvista
from gempy_engine.plugins.plotting.helper_functions import plot_scalar_and_input_2d, plot_block_and_input_2d
from tests.test_common.test_api.test_faults.test_one_fault import _plot_stack_raw


def test_graben_fault_model(graben_fault_model):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = graben_fault_model

    options.compute_scalar_gradient = False
    options.evaluation_options.dual_contouring = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    options.evaluation_options.dual_conturing_fancy = True
    options.debug=True

    options.evaluation_options.number_octree_levels = 5
    solutions: Solutions = compute_model(interpolation_input, options, structure)

    outputs: list[OctreeLevel] = solutions.octrees_output
    if plot_2d := False:
        _plot_stack_raw(interpolation_input, outputs, structure)
        plot_block_and_input_2d(2, interpolation_input, outputs, structure.stack_structure, value_type=ValueType.ids)

    if plot_pyvista or False:
        dc_data = solutions.dc_meshes[-1].dc_data
        centers = dc_data.xyz_on_centers
        valid_voxels = dc_data.valid_voxels.reshape(dc_data.n_surfaces_to_export, -1)
      
        #  selected_centers = centers[dc_data.valid_voxels]

        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes,
            show_edges=True,
        #  v_just_points =selected_centers,
            a=dc_data.bias_center_mass,
            b=dc_data.bias_normals,
            
            xyz_on_edge=dc_data.xyz_on_centers,
            plot_label=False
        )


def test_graben_fault_model_thickness(graben_fault_model, n_octree_levels=3):
    interpolation_input: InterpolationInput
    structure: InputDataDescriptor
    options: InterpolationOptions

    interpolation_input, structure, options = graben_fault_model

    options.compute_scalar_gradient = False
    options.evaluation_options.dual_contouring = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW

    fault_data: FaultsData = FaultsData.from_user_input(thickness=.2)
    fault_data2: FaultsData = FaultsData.from_user_input(thickness=.2)
    structure.stack_structure.faults_input_data = [fault_data, fault_data2, None]

    options.evaluation_options.number_octree_levels = n_octree_levels
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
    options.evaluation_options.dual_contouring = False

    fault_data: FaultsData = FaultsData.from_user_input(thickness=None)
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
