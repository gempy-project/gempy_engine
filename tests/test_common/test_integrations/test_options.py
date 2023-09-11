import pytest

from gempy_engine.API.model.model_api import compute_model
import gempy_engine.config
from gempy_engine.core.data import InterpolationOptions
from tests.conftest import plot_pyvista
from tests.fixtures.simple_models import simple_model_interpolation_input_factory


@pytest.mark.skipif(
    condition=gempy_engine.config.DEFAULT_BACKEND == gempy_engine.config.AvailableBackends.numpy,
    reason="Only for Tensorflow and Pykeops"
)
def test_dtype_propagates_float32(simple_grid_3d_octree, n_oct_levels=3):
    """This test only makes sense for Tensorflow and Pykeops I think"""
    options: InterpolationOptions
    interpolation_input, options, structure = simple_model_interpolation_input_factory(simple_grid_3d_octree)
    BackendTensor.dtype = 'float32'

    options.number_octree_levels = n_oct_levels
    solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista or False:
        import pyvista as pv
        from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes
        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        p.show()
