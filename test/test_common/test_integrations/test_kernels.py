import copy

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from test.conftest import plot_pyvista
from test.fixtures.simple_models import simple_model_interpolation_input_factory


def test_interpolate_model_cubic(simple_grid_3d_octree, n_oct_levels=3):
    interpolation_input, options, structure = simple_model_interpolation_input_factory(simple_grid_3d_octree)
    options.kernel_options.kernel_function = AvailableKernelFunctions.cubic

    options.number_octree_levels = n_oct_levels
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or True:
        import pyvista as pv
        from test.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels -1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        p.show()


def test_interpolate_model_exponential(simple_grid_3d_octree, n_oct_levels=3):
    interpolation_input, options, structure = simple_model_interpolation_input_factory(simple_grid_3d_octree)
    options.kernel_options.kernel_function = AvailableKernelFunctions.exponential

    options.number_octree_levels = n_oct_levels
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or True:
        import pyvista as pv
        from test.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels -1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        plot_points(p, interpolation_input.surface_points.sp_coords)
        p.show()
