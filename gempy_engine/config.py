from importlib.util import find_spec
from enum import Enum, auto


class AvailableBackends(Enum):
    numpy = auto()
    numpyPykeopsCPU = auto()
    numpyPykeopsGPU = auto()
    tensorflowCPU = auto()
    tensorflowGPU = auto()

DEBUG_MODE = True
DEFAULT_BACKEND = AvailableBackends.numpyPykeopsGPU
DEFAULT_DTYPE = "float32" # This is only used if backend is CPU

is_numpy_installed = find_spec("numpy") is not None
is_tensorflow_installed = find_spec("tensorflow") is not None
is_jax_installed = find_spec("jax") is not None
is_pykeops_installed = find_spec("pykeops") is not None



