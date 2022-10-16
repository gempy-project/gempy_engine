import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends, LINE_PROFILER_ENABLED
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.solutions import Solutions
from test.conftest import plot_pyvista, use_gpu

import tensorflow as tf

# ! Make sure profiler is disable!
pytestmark = pytest.mark.skipif(LINE_PROFILER_ENABLED and False, reason="Line profiler is enabled")

def test_one_feature_numpy(moureze_model, benchmark):
    BackendTensor.change_backend(
        engine_backend=AvailableBackends.numpy,
        use_gpu=False,
        pykeops_enabled=False
    )
    _run_model(benchmark, moureze_model, True)
    
    
def test_one_feature_numpy_pykeops_CPU(moureze_model, benchmark):
    BackendTensor.change_backend(
        engine_backend=AvailableBackends.numpy,
        use_gpu=False,
        pykeops_enabled=True
    )
    _run_model(benchmark, moureze_model, True)
    
    
def test_one_feature_numpy_pykeops_GPU(moureze_model, benchmark):
    BackendTensor.change_backend(
        engine_backend=AvailableBackends.numpy,
        use_gpu=True,
        pykeops_enabled=True
    )
    _run_model(benchmark, moureze_model, True)
    

class TestTF:
    # ! The order seems to matter!
    def test_one_feature_tf_GPU(self, moureze_model, benchmark):
        if use_gpu is False:
            raise pytest.skip("conftest.py is set to not use GPU")
        
        BackendTensor.change_backend(
            engine_backend=AvailableBackends.tensorflow,
            use_gpu=True,  
            pykeops_enabled=False
        )
        
        with tf.device('/GPU:0'):
            _run_model(benchmark, moureze_model)    
        
    
    def test_one_feature_tf_CPU(self, moureze_model, benchmark):
        BackendTensor.change_backend(
            engine_backend=AvailableBackends.tensorflow,
            use_gpu=True, # ! This has to be true because once is set to False it will affect the whole run
            pykeops_enabled=False
        )
        
        with tf.device('/CPU:0'):
            _run_model(benchmark, moureze_model)
    
    
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
    else:
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
