import copy
from typing import List

import pytest

from gempy_engine.API.interp_single._multi_scalar_field_manager import _interpolate_stack, interpolate_all_fields
from gempy_engine.API.interp_single._octree_generation import _generate_corners
from gempy_engine.API.interp_single.interp_features import interpolate_n_octree_levels
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid 
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level
from gempy_engine.plugins.plotting import helper_functions_pyvista
from gempy_engine.plugins.plotting.helper_functions import plot_block
from tests.conftest import plot_pyvista, TEST_SPEED

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
except ImportError:
    plot_pyvista = False


plot_pyvista = False

def test_extract_input_subsets(unconformity_complex):
    interpolation_input, options, input_descriptor = unconformity_complex
    stack_structure = input_descriptor.stack_structure
    for i in range(stack_structure.n_stacks):
        stack_structure.stack_number = i
        tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(input_descriptor, i)
        interpolation_input_i = InterpolationInput.from_interpolation_input_subset(interpolation_input, stack_structure)
        if i == 1:
            assert interpolation_input_i.surface_points.sp_coords.shape[0] == 2
        if i == 0:
            assert interpolation_input_i.surface_points.sp_coords.shape[0] == 3
        print("Iteration {}".format(i))
        print(tensor_struct_i)
        print(interpolation_input_i)


def test_compute_several_scalar_fields(unconformity_complex):
    """Plot each individual scalar field"""
    # TODO:
    interpolation_input, options, structure = unconformity_complex
    outputs: List[ScalarFieldOutput] = _interpolate_stack(structure, interpolation_input, options)

    if plot_pyvista or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].values_block, grid)
        plot_block(outputs[1].values_block, grid)
        plot_block(outputs[2].values_block, grid)


def test_compute_mask_components_all_erode(unconformity_complex):
    """Plot each individual mask compontent"""
    # TODO:
    interpolation_input, options, structure = unconformity_complex
    outputs: List[ScalarFieldOutput] = _interpolate_stack(structure, interpolation_input, options)

    if plot_pyvista or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].mask_components_erode, grid)
        plot_block(outputs[1].mask_components_erode, grid)
        plot_block(outputs[2].mask_components_erode, grid)


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
# noinspection PyUnreachableCode
def test_mask_arrays(unconformity_complex):
    interpolation_input, options, structure = copy.deepcopy(unconformity_complex)
    outputs: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)
    grid = interpolation_input.grid.octree_grid

    grid_0_centers = interpolation_input.grid

    grid_0_corners = EngineGrid.from_xyz_coords(
        xyz_coords=_generate_corners(regular_grid=grid_0_centers.octree_grid)
    )
    interpolation_input.set_temp_grid(grid_0_corners)

    output_0_corners: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)  # TODO: This is unnecessary for the last level except for Dual contouring

    if plot_pyvista or False:
        plot_block(outputs[0].combined_scalar_field.squeezed_mask_array, grid)
        plot_block(outputs[1].combined_scalar_field.squeezed_mask_array, grid)
        plot_block(outputs[2].combined_scalar_field.squeezed_mask_array, grid)

    m1 = output_0_corners[0].squeezed_mask_array.reshape((1, -1, 8))
    m2 = output_0_corners[1].squeezed_mask_array.reshape((1, -1, 8))
    m3  = output_0_corners[2].squeezed_mask_array.reshape((1, -1, 8))
    mask_1 = BackendTensor.t.to_numpy(BackendTensor.t.sum(m1, -1, bool)[0])
    mask_2 = BackendTensor.t.to_numpy(BackendTensor.t.sum(m2, -1, bool)[0])
    mask_3 = BackendTensor.t.to_numpy(BackendTensor.t.sum(m3, -1, bool)[0])

    if True:
        plot_block(mask_1, grid)
        plot_block(mask_2, grid)
        plot_block(mask_3, grid)

    if True:
        mask_1_f = mask_1
        mask_2_f = (mask_1_f ^ mask_2) * mask_2
        mask_3_f = (mask_2_f ^ mask_3) * mask_3

        plot_block(mask_1_f, grid)
        plot_block(mask_2_f, grid)
        plot_block(mask_3_f, grid)


def test_final_block(unconformity_complex):
    interpolation_input, options, structure = unconformity_complex
    outputs: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)

    if plot_pyvista or True:
        grid = interpolation_input.grid.octree_grid
        plot_block(
            BackendTensor.t.to_numpy(outputs[0].final_block),
            grid)
        plot_block(
            BackendTensor.t.to_numpy(outputs[1].final_block),
            grid)
        plot_block(
            BackendTensor.t.to_numpy(outputs[2].final_block),
            grid)


def test_final_exported_fields(unconformity_complex):
    interpolation_input, options, structure = unconformity_complex
    outputs: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)

    if plot_pyvista or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(BackendTensor.t.to_numpy(outputs[0].final_exported_fields._scalar_field), grid)
        plot_block(BackendTensor.t.to_numpy(outputs[1].final_exported_fields._scalar_field), grid)
        plot_block(BackendTensor.t.to_numpy(outputs[2].final_exported_fields._scalar_field), grid)


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_plot_corners(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)
    vertices = solutions.octrees_output[-1].grid_centers.corners_grid.values
    if plot_pyvista or False:
        helper_functions_pyvista.plot_pyvista(solutions.octrees_output, v_just_points=vertices)


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_final_block_octrees(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    solution: Solutions = Solutions(
        octrees_output=interpolate_n_octree_levels(interpolation_input, options, structure)
    )

    final_block = solution.octrees_output[0].output_centers.final_block
    final_block2 = get_regular_grid_value_for_level(solution.octrees_output, 1).astype("int8")

    if plot_2d := False or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(final_block, grid)

        grid2 = solution.octrees_output[1].grid_centers.regular_grid
        plot_block(final_block2, grid2)
