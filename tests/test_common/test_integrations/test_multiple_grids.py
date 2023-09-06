import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.generic_grid import GenericGrid
from ...conftest import plot_pyvista, TEST_SPEED

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
except ImportError:
    plot_pyvista = False


def test_interpolate_model_on_multiple_grids(simple_model_interpolation_input, n_oct_levels=3):
    """Kernel function Cubic"""
    interpolation_input, options, structure = simple_model_interpolation_input
    print(interpolation_input)

    options.number_octree_levels = n_oct_levels
    
    # * Add the second grid 
    interpolation_input.grid.topography = GenericGrid(values=np.array([[1,2,3], [4,5,6]]))
    interpolation_input.grid.custom_grid = GenericGrid(values=np.array([[7,8,9], [10,11,12]]))
    
    
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels -1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        p.show()
