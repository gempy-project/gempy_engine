import pytest

from gempy_engine.API.model.model_api import compute_model
from ...conftest import plot_pyvista, TEST_SPEED

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
except ImportError:
    plot_pyvista = False


def test_interpolate_model(simple_model_interpolation_input, n_oct_levels=3):

    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir()
    """Kernel function Cubic"""
    interpolation_input, options, structure = simple_model_interpolation_input
    print(interpolation_input)

    options.number_octree_levels = n_oct_levels
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels -1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        p.show()


@pytest.mark.skipif(TEST_SPEED.value <= 1, reason="Global test speed below this test value.")
def test_interpolate_model_no_octtree(simple_model_3_layers_high_res, n_oct_levels=2):

    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir()
    interpolation_input, options, structure = simple_model_3_layers_high_res
    print(interpolation_input)

    options.number_octree_levels = n_oct_levels
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        # pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        p.show()


def test_interpolate_model_several_surfaces(simple_model_3_layers, n_oct_levels=3):

    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir()
    interpolation_input, options, structure = simple_model_3_layers
    print(interpolation_input)

    options.number_octree_levels = n_oct_levels
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        # pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        plot_dc_meshes(p, solutions.dc_meshes[0])

        plot_points(p, interpolation_input.surface_points.sp_coords, True)
        plot_vector(p, interpolation_input.orientations.dip_positions,
                    interpolation_input.orientations.dip_gradients)
        p.show()


def test_interpolate_model_unconformity(unconformity, n_oct_levels=4):
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir()

    interpolation_input, options, structure = unconformity
    print(interpolation_input)

    options.number_octree_levels = n_oct_levels
    solutions = compute_model(interpolation_input, options, structure)
    
    assert solutions.dc_meshes[2].vertices.shape[0] == 0

    if plot_pyvista or False:
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)

        plot_points(p, interpolation_input.surface_points.sp_coords, True)
        plot_vector(p, interpolation_input.orientations.dip_positions,
                    interpolation_input.orientations.dip_gradients)

        plot_dc_meshes(p, solutions.dc_meshes[0])
        plot_dc_meshes(p, solutions.dc_meshes[1])
        
        p.show()
