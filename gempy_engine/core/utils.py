from typing import Any
import numpy as np
import gempy_engine.config


def cast_type_inplace(data_instance: Any):
    """Converts all numpy arrays to the global dtype"""    
    for key, val in data_instance.__dict__.items():
        if type(val) != np.ndarray: continue
        data_instance.__dict__[key] = val.astype(gempy_engine.config.TENSOR_DTYPE)
