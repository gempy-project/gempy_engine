import pytest

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.config import AvailableBackends

from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess


@pytest.mark.skip("Run only manually")
class TestBackEnds:
    def test_optional_dependencies(self):
        import gempy_engine.config
    
        print(gempy_engine.config.is_numpy_installed)
        print(gempy_engine.config.is_tensorflow_installed)
        print(gempy_engine.config.is_pykeops_installed)
    
    def test_change_backend_on_the_fly(self):
    
        np_module = BackendTensor.tfnp
        BackendTensor.change_backend(AvailableBackends.tensorflow)
        tf_module = BackendTensor.tfnp
    
        pass
    
    def test_backends_are_running(self, simple_model_2):
        surface_points = simple_model_2[0]
        tensors_structure = simple_model_2[3]
    
        # Run Default numpy-cpu
        s = surface_points_preprocess(surface_points, tensors_structure.tensors_structure)
        print(s)
    
        # Run TF-Default
        BackendTensor.change_backend(AvailableBackends.tensorflow)
        s = surface_points_preprocess(surface_points, tensors_structure.tensors_structure)
        print(s)
    
    
    def test_data_class_hash(self, simple_model_2):
        import numpy as np
        sp_coords = np.array([[4, 0],
                              [0, 0],
                              [2, 0],
                              [3, 0],
                              [3, 3],
                              [0, 2],
                              [2, 2]])
        nugget_effect_scalar = 0
        from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
    
        surface_points = SurfacePoints(sp_coords, nugget_effect_scalar)
    
        print(surface_points.__hash__())
    
        f = {"a": np.ones(6)}
        print(f.__hash__)
    
    
    # We should trigger them manually
    def test_tf_function(self, simple_model_2):
        surface_points = simple_model_2[0]
        tensors_structure = simple_model_2[3]
    
        BackendTensor.change_backend(AvailableBackends.tensorflow)
    
        import tensorflow as tf
        @tf.function(experimental_compile=False)
        def xla_(surface_points, tensors_structure):
            s = surface_points_preprocess(surface_points, tensors_structure.number_of_points_per_surface)
            return s.ref_surface_points
    
        s = xla_(surface_points, tensors_structure)
    
        print(s)
