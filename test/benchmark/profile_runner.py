"""
Most of the profiles required to trigger it directly from the command line. This module helps to run benchmark
models together with a profiler. 
"""
# ! Open settings in pycharm -> Terminal and add as Environment variable: PYTHONPATH=/WorkSSD/PythonProjects/gempy_engine
from typing import Tuple

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.solutions import Solutions

from test.fixtures.heavy_models import moureze_model_factory
from test.conftest import plot_pyvista

def profile_moureze_model():
    BackendTensor.change_backend(
        engine_backend=AvailableBackends.numpy,
        use_gpu=False,
        pykeops_enabled=False
    )

    _run_model(moureze_model_factory(pick_every=16))


def _run_model(model: Tuple[InterpolationInput, InterpolationOptions, InputDataDescriptor]):
    """Use benchmark_active=False to debug"""

    interpolation_input: InterpolationInput
    options: InterpolationOptions
    structure: InputDataDescriptor
    interpolation_input, options, structure = model
    n_oct_levels = options.number_octree_levels

    solutions: Solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista and False:
        import pyvista as pv
        from test.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points

        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        plot_points(p, interpolation_input.surface_points.sp_coords)
        p.show()


if __name__ == '__main__':
    profile_moureze_model()
    print("Done")