
import pytest
import numpy as np

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.data.centered_grid import CenteredGrid
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.geophysics_input import GeophysicsInput
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.modules.geophysics.gravity_gradient import calculate_gravity_gradient
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
    geophysics_grid = CenteredGrid(
        centers=np.array([[0.25, 0.5, 0.75], [0.75, 0.5, 0.75]]),
        resolution=np.array([10, 10, 15]), 
        radius=np.array([1, 1, 1])  
    )
    
    interpolation_input.grid.geophysics_grid = geophysics_grid
    # -----------------------------
    
    gravity_gradient = calculate_gravity_gradient(geophysics_grid)
    # * set up the geophysics input
    geophysics_input = GeophysicsInput(
        tz=gravity_gradient,
        densities=np.array([1.3, 4]),
    )
    
    solutions = compute_model(interpolation_input, options, structure, geophysics_input=geophysics_input)

    # Expected: [-0.09757573171386501, -0.10282069913078262]
    np.testing.assert_array_almost_equal(
        solutions.gravity,
        np.array([-0.09757573171386501, -0.10282069913078262]),
        decimal=8
    )
    
    if plot_pyvista or False:
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels -1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        p.show()