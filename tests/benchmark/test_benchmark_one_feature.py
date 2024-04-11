"""
This is the script used for benchmarking


"""
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends, LINE_PROFILER_ENABLED
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.optional_dependencies import require_tensorflow
from tests.conftest import plot_pyvista, use_gpu, REQUIREMENT_LEVEL, Requirements

# ! Make sure profiler is disabled!
pytestmark = pytest.mark.skipif(LINE_PROFILER_ENABLED and False, reason="Line profiler is enabled")


def test_one_feature_numpy(moureze_model, benchmark):
    options: InterpolationOptions = moureze_model[1]
    if options.number_octree_levels > 3:
        pytest.skip("Too many octree levels, too slow")
    
    if BackendTensor.engine_backend != AvailableBackends.numpy:
        pytest.skip("NumPy backend is not set")

    _run_model(benchmark, moureze_model, benchmark_active = True)


@pytest.mark.skipif(REQUIREMENT_LEVEL.value < Requirements.OPTIONAL.value, reason="This test needs higher requirements.")
class TestPyKeops:
    def test_one_feature_numpy_pykeops_CPU(self, moureze_model, benchmark):
        BackendTensor.change_backend(
            engine_backend=AvailableBackends.numpy,
            use_gpu=False,
            pykeops_enabled=True
        )
        _run_model(benchmark, moureze_model, benchmark_active = False)


    def test_one_feature_numpy_pykeops_GPU(self, moureze_model, benchmark):
        BackendTensor.change_backend(
            engine_backend=AvailableBackends.numpy,
            use_gpu=True,
            pykeops_enabled=True
        )
        _run_model(benchmark, moureze_model, benchmark_active=False)


def _run_model(benchmark, moureze_model, benchmark_active=True):
    """Use benchmark_active=False to debug"""

    interpolation_input: InterpolationInput
    options: InterpolationOptions
    structure: InputDataDescriptor
    interpolation_input, options, structure = moureze_model
    n_oct_levels = options.number_octree_levels

    if benchmark_active:
        solutions: Solutions = benchmark.pedantic(
            target=compute_model,
            args=(interpolation_input, options, structure)
        )
        benchmark.extra_info['n_oct_levels'] = n_oct_levels
        benchmark.extra_info['n_points'] = interpolation_input.surface_points.n_points
        benchmark.extra_info['n_orientations'] = interpolation_input.orientations.n_items
    else:
        solutions: Solutions = compute_model(interpolation_input, options, structure)

    if plot_pyvista and False:
        import pyvista as pv
        from gempy_engine.plugins.plotting.helper_functions_pyvista import plot_octree_pyvista, plot_dc_meshes, plot_points

        pv.global_theme.show_edges = True
        p = pv.Plotter()
        plot_octree_pyvista(p, solutions.octrees_output, n_oct_levels - 1)
        plot_dc_meshes(p, solutions.dc_meshes[0])
        plot_points(p, interpolation_input.surface_points.sp_coords)
        p.show()
