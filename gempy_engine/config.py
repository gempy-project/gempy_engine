from importlib.util import find_spec
from enum import Enum, auto, Flag


class AvailableBackends(Flag):
    numpy = auto()
    tensorflow = auto()
    PYTORCH = auto()
    
    # Legacy
    aesara = auto()
    legacy = auto()


# ! Careful what we commit here!
DEBUG_MODE = True
OPTIMIZE_MEMORY = True
DEFAULT_BACKEND = AvailableBackends.PYTORCH
DEFAULT_PYKEOPS = False
DEFAULT_TENSOR_DTYPE = 'float64'
LINE_PROFILER_ENABLED = False
SET_RAW_ARRAYS_IN_SOLUTION = True
COMPUTE_GRADIENTS = True

is_numpy_installed = find_spec("numpy") is not None
is_tensorflow_installed = find_spec("tensorflow") is not None
is_pytorch_installed = find_spec("pytorch")
is_pykeops_installed = find_spec("pykeops") is not None



