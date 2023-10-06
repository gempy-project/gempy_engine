
import pytest
import numpy as np

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.geophysics_input import GeophysicsInput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from ...conftest import plot_pyvista, TEST_SPEED

try:
    # noinspection PyUnresolvedReferences
    import pyvista as pv
    from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points, plot_vector
except ImportError:
    plot_pyvista = False
    
    
def test_gravity(simple_model_interpolation_input, n_oct_levels=3):
    """Kernel function Cubic"""
    interpolation_input: InterpolationInput
    interpolation_input, options, structure = simple_model_interpolation_input
    print(interpolation_input)

    options.number_octree_levels = n_oct_levels
    
    # * set up the geophysics grid
    
    # TODO: [ ] Calculate reasonable geophysics grid with tz
    
    # Here is spawning from the device
    geophysics_grid = GenericGrid(
        values=np.array([[0.5, 0.5, 0.75]]),
    )
    
    interpolation_input.grid.geophysics_grid = geophysics_grid
    # -----------------------------
    
    # * set up the geophysics input
    geophysics_input = GeophysicsInput(
        tz=np.array([0.0]),
        densities=np.array([1.3, 2]),
    )
    
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or True:
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels -1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        p.show()