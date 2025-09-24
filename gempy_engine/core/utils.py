from typing import Any
import numpy as np
import gempy_engine.config
from ..core.backend_tensor import BackendTensor


def cast_type_inplace(data_instance: Any, requires_grad:bool = False):
    """Converts all numpy arrays to the global dtype"""    
    for key, val in data_instance.__dict__.items():
        if type(val) != np.ndarray: continue
        match BackendTensor.engine_backend:
            case (gempy_engine.config.AvailableBackends.numpy):
                data_instance.__dict__[key] = val.astype(BackendTensor.dtype)
            case (gempy_engine.config.AvailableBackends.PYTORCH):
                # tensor = BackendTensor.t.from_numpy(val.astype(BackendTensor.dtype))
                # if (BackendTensor.use_gpu):
                #     tensor = tensor.cuda()

                tensor = BackendTensor.tfnp.array(val, dtype=BackendTensor.dtype_obj)
                tensor.requires_grad = requires_grad
                data_instance.__dict__[key] = tensor
        


def gempy_profiler_decorator(func):
    """Decorator to profile a function"""
    if gempy_engine.config.LINE_PROFILER_ENABLED:
        try:
            from line_profiler_pycharm import profile
            return profile(func)
        except ImportError:
            return func
    else:
        return func


def _check_and_convert_list_to_array(field):
    if type(field) == list:
        field = np.array(field)
    return field
