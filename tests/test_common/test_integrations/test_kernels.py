from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions
from tests.conftest import plot_pyvista
from tests.fixtures.simple_models import simple_model_interpolation_input_factory


def test_interpolate_model_cubic(n_oct_levels=3):
    interpolation_input, options, structure = simple_model_interpolation_input_factory()
    options = InterpolationOptions.from_args(
        range=options.range,
        c_o=options.c_o,
        uni_degree=0,
        number_dimensions=3,
        kernel_function=AvailableKernelFunctions.cubic,
        number_octree_levels=n_oct_levels,
    )

    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        import pyvista as pv
        from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        p.show()


def test_interpolate_model_exponential(n_oct_levels=3):
    interpolation_input, options, structure = simple_model_interpolation_input_factory()

    options = InterpolationOptions.from_args(
        range=options.range,
        c_o=options.c_o,
        uni_degree=0,
        number_dimensions=3,
        kernel_function=AvailableKernelFunctions.exponential,
        number_octree_levels=n_oct_levels,
    )

    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        import pyvista as pv
        from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        plot_points(p, interpolation_input.surface_points.sp_coords)
        p.show()
