import pytest

from gempy_engine import compute_model
from gempy_engine.core.data.options import DualContouringMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from test import helper_functions_pyvista
from test.conftest import TEST_SPEED
from test.test_common.test_integrations.test_multi_fields import plot_pyvista


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
    options.compute_scalar_gradient = True
    
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
    options.compute_scalar_gradient = True

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
