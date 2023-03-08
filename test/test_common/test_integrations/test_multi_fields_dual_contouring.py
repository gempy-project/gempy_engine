import pytest

from gempy_engine import compute_model
from gempy_engine.core.data.grid import RegularGrid, Grid
from gempy_engine.core.data.options import DualContouringMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from test import helper_functions_pyvista
from test.conftest import TEST_SPEED, plot_pyvista
from test.test_common.test_integrations.test_multi_fields import plot_pyvista


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields(unconformity_complex, n_oct_levels=2):
    _run_model_for_dual_contouring_option(
        dual_contouring_option=DualContouringMaskingOptions.DISJOINT,
        n_oct_levels=n_oct_levels,
        model=unconformity_complex
    )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields_intersect(unconformity_complex, n_oct_levels=2):
    _run_model_for_dual_contouring_option(
        dual_contouring_option=DualContouringMaskingOptions.INTERSECT,
        n_oct_levels=n_oct_levels,
        model=unconformity_complex
    )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields_intersect_raw(unconformity_complex, n_oct_levels=2):
    _run_model_for_dual_contouring_option(
        dual_contouring_option=DualContouringMaskingOptions.RAW,
        n_oct_levels=n_oct_levels,
        model=unconformity_complex
    )
    
    
@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields_intersect_raw_fancy_triangulation(
        unconformity_complex, n_oct_levels=2):
    
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.dual_contouring_masking_options = DualContouringMaskingOptions.RAW
    
    options.dual_contouring_fancy = True
    regular_grid = RegularGrid(extent=[0, 10., 0, 2., 0, 5.], regular_grid_shape=[2, 2, 2])
    grid = Grid(regular_grid.values, regular_grid=regular_grid)
    interpolation_input.grid = grid

    solutions: Solutions = compute_model(interpolation_input, options, structure)
    if plot_pyvista or False:
        dc_data = solutions.dc_meshes[0].dc_data  # * Scalar field where to show gradients
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes,
            # xyz_on_edge=dc_data.xyz_on_edge,
            # gradients=dc_data.gradients,  # * Uncomment for more detailed plots
            # a=dc_data.bias_center_mass,
            # b=dc_data.bias_normals
        )


def _run_model_for_dual_contouring_option(dual_contouring_option, n_oct_levels, model):
    interpolation_input, options, structure = model
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.dual_contouring_masking_options = dual_contouring_option
    solutions: Solutions = compute_model(interpolation_input, options, structure)
    if plot_pyvista or False:
        dc_data = solutions.dc_meshes[0].dc_data  # * Scalar field where to show gradients
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes,
            # xyz_on_edge=dc_data.xyz_on_edge,
            # gradients=dc_data.gradients,  # * Uncomment for more detailed plots
            # a=dc_data.bias_center_mass,
            # b=dc_data.bias_normals
        )


@pytest.mark.skipif(True, reason="This is just experimental. Run only manually.")
def test_dual_contouring_multiple_independent_fields_mask_experimental_water_tight(unconformity_complex, n_oct_levels=2):
    """This is just experimental"""
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.debug_water_tight = True # ! This is what makes the difference
    options.compute_scalar_gradient = True

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        dc_data = solutions.dc_meshes[0].dc_data  # * Scalar field where to show gradients
        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes,
            # xyz_on_edge=dc_data.xyz_on_edge,
            # gradients=dc_data.gradients,  # * Uncomment for more detailed plots
            # a=dc_data.bias_center_mass,
            # b=dc_data.bias_normals
        )