import copy
from typing import List

import pytest
from matplotlib import pyplot as plt

from gempy_engine.API.interp_single._multi_scalar_field_manager import _interpolate_stack, interpolate_all_fields
from gempy_engine.API.model.model_api import _interpolate, compute_model
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.core.data.interp_output import InterpOutput
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.input_data_descriptor import TensorsStructure
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.options import DualContouringMaskingOptions
from gempy_engine.modules.octrees_topology.octrees_topology_interface import get_regular_grid_value_for_level
from test import helper_functions_pyvista
from test.conftest import plot_pyvista, TEST_SPEED
from test.helper_functions import plot_block

plot_pyvista = False
try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from test.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
except ImportError:
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
        plot_block(outputs[0].mask_components.mask_lith, grid)
        plot_block(outputs[1].mask_components.mask_lith, grid)
        plot_block(outputs[2].mask_components.mask_lith, grid)


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
# noinspection PyUnreachableCode
def test_mask_arrays(unconformity_complex):
    interpolation_input, options, structure = copy.deepcopy(unconformity_complex)
    outputs: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)
    grid = interpolation_input.grid.regular_grid

    grid_0_centers = interpolation_input.grid
    from gempy_engine.modules.octrees_topology._octree_common import _generate_corners
    from gempy_engine.core.data.grid import Grid
    grid_0_corners = Grid(_generate_corners(grid_0_centers.values, grid_0_centers.dxdydz))
    interpolation_input.grid = grid_0_corners

    output_0_corners: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)  # TODO: This is unnecessary for the last level except for Dual contouring

    if plot_pyvista or False:
        plot_block(outputs[0].combined_scalar_field.squeezed_mask_array, grid)
        plot_block(outputs[1].combined_scalar_field.squeezed_mask_array, grid)
        plot_block(outputs[2].combined_scalar_field.squeezed_mask_array, grid)

    mask_1 = output_0_corners[0].squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
    mask_2 = output_0_corners[1].combined_scalar_field.squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]
    mask_3 = output_0_corners[2].combined_scalar_field.squeezed_mask_array.reshape((1, -1, 8)).sum(-1, bool)[0]

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
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].final_block, grid)
        plot_block(outputs[1].final_block, grid)
        plot_block(outputs[2].final_block, grid)


def test_final_exported_fields(unconformity_complex):
    interpolation_input, options, structure = unconformity_complex
    outputs: List[InterpOutput] = interpolate_all_fields(interpolation_input, options, structure)

    if plot_pyvista or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(outputs[0].final_exported_fields._scalar_field, grid)
        plot_block(outputs[1].final_exported_fields._scalar_field, grid)
        plot_block(outputs[2].final_exported_fields._scalar_field, grid)


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_plot_corners(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    solutions: Solutions = compute_model(interpolation_input, options, structure)
    output_corners: InterpOutput = solutions.octrees_output[-1].outputs_corners[-1]

    vertices = output_corners.grid.values
    if plot_pyvista or False:
        helper_functions_pyvista.plot_pyvista(solutions.octrees_output, v_just_points=vertices)


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.DISJOINT

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        dc_data = solutions.dc_meshes[0].dc_data  # * Scalar field where to show gradients
        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals

        helper_functions_pyvista.plot_pyvista(solutions.octrees_output,
                                              dc_meshes=solutions.dc_meshes,
                                              # xyz_on_edge=intersection_xyz, gradients=gradients, # * Uncomment for more detailed plots
                                              # a=center_mass, b=normals
                                              )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields_intersect(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.INTERSECT

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        dc_data = solutions.dc_meshes[1].dc_data  # * Scalar field where to show gradients
        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals

        helper_functions_pyvista.plot_pyvista(solutions.octrees_output,
                                              dc_meshes=solutions.dc_meshes,
                                              # xyz_on_edge=intersection_xyz, gradients=gradients, # * Uncomment for more detailed plots
                                              # a=center_mass, b=normals
                                              )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields_intersect_raw(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.RAW

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        dc_data = solutions.dc_meshes[1].dc_data  # * Scalar field where to show gradients
        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals

        helper_functions_pyvista.plot_pyvista(solutions.octrees_output,
                                              dc_meshes=solutions.dc_meshes,
                                              # xyz_on_edge=intersection_xyz, gradients=gradients, # * Uncomment for more detailed plots
                                              # a=center_mass, b=normals
                                              )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields_mask(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.debug_water_tight = True

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        dc_data = solutions.dc_meshes[0].dc_data  # * Scalar field where to show gradients
        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals

        helper_functions_pyvista.plot_pyvista(octree_list=solutions.octrees_output,
                                              dc_meshes=solutions.dc_meshes,
                                              # xyz_on_edge=intersection_xyz, gradients=gradients,
                                              # a=center_mass, b=normals,
                                              # vertices=solutions.dc_meshes[0].vertices, delaunay_3d=False
                                              )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_dependent_fields(unconformity_complex, n_oct_levels=2):
    # * Dependent_dual_contouring seems a bad idea

    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.dependent_dual_contouring = True

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        dc_data = solutions.dc_meshes[0].dc_data
        intersection_xyz = dc_data.xyz_on_edge
        gradients = dc_data.gradients

        center_mass = dc_data.bias_center_mass
        normals = dc_data.bias_normals

        helper_functions_pyvista.plot_pyvista(solutions.octrees_output, dc_meshes=solutions.dc_meshes,
                                              xyz_on_edge=intersection_xyz, gradients=gradients,
                                              a=center_mass, b=normals
                                              )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_final_block_octrees(unconformity_complex, n_oct_levels=2):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    solution: Solutions = _interpolate(interpolation_input, options, structure)
    final_block = solution.octrees_output[0].output_centers.final_block
    final_block2 = get_regular_grid_value_for_level(solution.octrees_output, 1).astype("int8")

    if plot_pyvista or False:
        grid = interpolation_input.grid.regular_grid
        plot_block(final_block, grid)

        grid2 = solution.octrees_output[1].grid_centers.regular_grid
        plot_block(final_block2, grid2)
