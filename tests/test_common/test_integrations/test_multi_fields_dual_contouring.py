import pytest

from gempy_engine import compute_model
from gempy_engine.core.data.engine_grid  import RegularGrid, EngineGrid
from gempy_engine.core.data.options import MeshExtractionMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.plugins.plotting import helper_functions_pyvista
from tests.conftest import TEST_SPEED
from tests.fixtures.simple_models import unconformity_complex_factory
from tests.test_common.test_integrations.test_multi_fields import plot_pyvista


# @pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
@pytest.mark.skipif(True, reason="Disjoint is not working for now")
def test_dual_contouring_multiple_independent_fields(n_oct_levels=2):
    _run_model_for_dual_contouring_option(
        dual_contouring_option=MeshExtractionMaskingOptions.DISJOINT,
        n_oct_levels=n_oct_levels,
        model=unconformity_complex_factory()
    )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields_intersect(n_oct_levels=2):
    _run_model_for_dual_contouring_option(
        dual_contouring_option=MeshExtractionMaskingOptions.INTERSECT,
        n_oct_levels=n_oct_levels,
        model=unconformity_complex_factory()
    )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields_intersect_raw(n_oct_levels=2):
    _run_model_for_dual_contouring_option(
        dual_contouring_option=MeshExtractionMaskingOptions.RAW,
        n_oct_levels=n_oct_levels,
        model=unconformity_complex_factory()
    )


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields_intersect_RAW_fancy_triangulation(
        unconformity_complex, n_oct_levels=5):
    
    _run_model_for_FANCY_dual_contouring(
        dual_contouring_option=MeshExtractionMaskingOptions.RAW,
        n_oct_levels=n_oct_levels,
        unconformity_complex=unconformity_complex
    )


# @pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
@pytest.mark.skipif(True, reason="Disjoint is not working for now")
def test_dual_contouring_multiple_independent_fields_intersect_DISJOINT_fancy_triangulation(
        unconformity_complex, n_oct_levels=5):
    _run_model_for_FANCY_dual_contouring(
        dual_contouring_option=MeshExtractionMaskingOptions.DISJOINT,
        n_oct_levels=n_oct_levels,
        unconformity_complex=unconformity_complex
    )
    
    
@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_dual_contouring_multiple_independent_fields_intersect_INTERSECT_fancy_triangulation(
        unconformity_complex, n_oct_levels=5):
    _run_model_for_FANCY_dual_contouring(
        dual_contouring_option=MeshExtractionMaskingOptions.INTERSECT,
        n_oct_levels=n_oct_levels,
        unconformity_complex=unconformity_complex
    )


def _run_model_for_FANCY_dual_contouring(dual_contouring_option, n_oct_levels, unconformity_complex):
    interpolation_input, options, structure = unconformity_complex
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.evaluation_options.mesh_extraction_masking_options = dual_contouring_option
    options.evaluation_options.mesh_extraction_fancy = True
    regular_grid = RegularGrid(orthogonal_extent=[0, 10., 0, 2., 0, 5.], regular_grid_shape=[2, 2, 2])
    grid = EngineGrid(octree_grid=regular_grid)
    interpolation_input.set_temp_grid(grid)
    solutions: Solutions = compute_model(interpolation_input, options, structure)
    if plot_pyvista or False:
        dc_data = solutions.dc_meshes[3].dc_data  # * Scalar field where to show gradients
        valid_voxels_per_surface = dc_data.valid_voxels.reshape((dc_data.n_surfaces_to_export, -1))

        valid_voxels = dc_data.xyz_on_centers[valid_voxels_per_surface[1]]

        helper_functions_pyvista.plot_pyvista(
            solutions.octrees_output,
            dc_meshes=solutions.dc_meshes,
            #v_just_points=valid_voxels
            # xyz_on_edge=dc_data.xyz_on_edge,
            # gradients=dc_data.gradients,  # * Uncomment for more detailed plots
            # a=dc_data.bias_center_mass,
            # b=dc_data.bias_normals
        )


def _run_model_for_dual_contouring_option(dual_contouring_option, n_oct_levels, model):
    interpolation_input, options, structure = model
    options.number_octree_levels = n_oct_levels
    options.debug = True
    options.evaluation_options.mesh_extraction_masking_options = dual_contouring_option
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
    options.debug_water_tight = True  # ! This is what makes the difference
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
