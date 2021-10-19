from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.config import AvailableBackends

from gempy_engine.modules.data_preprocess._input_preparation import surface_points_preprocess

from ..fixtures.simple_models import simple_model_2

def test_optional_dependencies():
    import gempy_engine.config

    print(gempy_engine.config.is_numpy_installed)
    print(gempy_engine.config.is_tensorflow_installed)
    print(gempy_engine.config.is_pykeops_installed)


def test_data_class_hash(simple_model_2):
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
