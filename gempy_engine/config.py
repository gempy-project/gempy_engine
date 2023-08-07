from importlib.util import find_spec
from enum import Enum, auto, Flag


class AvailableBackends(Flag):
    numpy = auto()
    tensorflow = auto()
    # Legacy
    aesara = auto()
    legacy = auto()


DEBUG_MODE = True
OPTIMIZE_MEMORY = True
DEFAULT_BACKEND = AvailableBackends.numpy
TENSOR_DTYPE = 'float64'
LINE_PROFILER_ENABLED = False

is_numpy_installed = find_spec("numpy") is not None
is_tensorflow_installed = find_spec("tensorflow") is not None
is_pykeops_installed = find_spec("pykeops") is not None



