from importlib.util import find_spec
from enum import Enum, auto
from typing import Union, List, Any


class AvailableBackends(Enum):
    numpy = auto()
    tensorflow = auto()
    jax = auto()


# Choose the backend:
engine_backend = AvailableBackends.jax

is_numpy_installed = find_spec("numpy") is not None
is_tensorflow_installed = find_spec("tensorflow") is not None
is_jax_installed = find_spec("jax") is not None
is_pykeops_installed = find_spec("pykeops") is not None


class BackendConf():
    engine_backend: AvailableBackends
    pykeops_enabled: bool
    tensor_types: List
    tfnp: Any # Pycharm will infer the type. It is the best I got so far

    @classmethod
    def change_backend(cls, engine_backend: AvailableBackends, pykeops_enabled:bool = False):

        if pykeops_enabled and is_pykeops_installed and is_numpy_installed:
            cls.pykeops_enabled = True
            cls.engine_backend = AvailableBackends.numpy

            import numpy as tfnp
            tfnp.reduce_sum = tfnp.sum
            tfnp.concat = tfnp.concatenate
            tfnp.constant = tfnp.array

            cls.engine_backend = engine_backend
            cls.tfnp = tfnp
            cls.tensor_types = Union[tfnp.ndarray]  # tens

        else:
            cls.pykeops_enabled = False

            if engine_backend is AvailableBackends.jax and is_jax_installed:
                import jax.numpy as tfnp
                tfnp.reduce_sum = tfnp.sum
                tfnp.concat = tfnp.concatenate
                tfnp.constant = tfnp.array

                cls.engine_backend = engine_backend
                cls.tfnp = tfnp
                cls.tensor_types = Union[tfnp.ndarray]  # tensor Types with respect the backend:

            elif engine_backend is AvailableBackends.tensorflow and is_tensorflow_installed:
                import tensorflow as  tfnp

                cls.engine_backend = engine_backend
                cls.tfnp = tfnp
                cls.tensor_types = Union[tfnp.ndarray, tfnp.Tensor, tfnp.Variable]  # tensor Types with respect the backend:

            elif engine_backend is AvailableBackends.numpy and is_numpy_installed:

                import numpy as tfnp
                tfnp.reduce_sum = tfnp.sum
                tfnp.concat = tfnp.concatenate
                tfnp.constant = tfnp.array

                cls.engine_backend = engine_backend
                cls.tfnp = tfnp
                cls.tensor_types = Union[tfnp.ndarray]  # tensor Types with respect the backend

            else:
                raise AttributeError(f"Engine Backend: {engine_backend} cannot be used because the correspondent library"
                                     f"is not installed:")

BackendConf.change_backend(AvailableBackends.numpy)

