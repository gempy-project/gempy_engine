from gempy_engine.integrations.interp_manager.interp_manager_api import interpolate_model
from test.helper_functions import plot_octree_pyvista


def test_interpolate_model(simple_model_interpolation_input, n_oct_levels = 4):
    interpolation_input, options, structure = simple_model_interpolation_input
    print(interpolation_input)

    options.number_octree_levels = n_oct_levels
    output = interpolate_model(interpolation_input, options ,structure)
    plot_octree_pyvista(output, n_oct_levels - 1)