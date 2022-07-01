from importlib.util import find_spec
from enum import Enum, auto


class AvailableBackends(Enum):
    numpy = auto()
    tensorflow = auto()
    jax = auto()

DEBUG_MODE = True
OPTIMIZE_MEMORY = True
DEFAULT_BACKEND = AvailableBackends.numpy

is_numpy_installed = find_spec("numpy") is not None
is_tensorflow_installed = find_spec("tensorflow") is not None
is_jax_installed = find_spec("jax") is not None
is_pykeops_installed = find_spec("pykeops") is not None



