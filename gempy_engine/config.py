import warnings
from importlib.util import find_spec
from enum import Enum, auto
from typing import Union, List, Any
import numpy as np

class AvailableBackends(Enum):
    numpy = auto()
    tensorflow = auto()
    jax = auto()

DEBUG_MODE = True
DEFAULT_BACKEND = AvailableBackends.numpy


# Choose the backend:
engine_backend = AvailableBackends.jax

is_numpy_installed = find_spec("numpy") is not None
is_tensorflow_installed = find_spec("tensorflow") is not None
is_jax_installed = find_spec("jax") is not None
is_pykeops_installed = find_spec("pykeops") is not None


# TODO: Move this to its own script and to gempy_engine.core
class BackendTensor():
    engine_backend: AvailableBackends

    pykeops_enabled: bool
    use_gpu: bool = True

    tensor_types: Union
    tensor_backend_pointer: dict = dict() # Pycharm will infer the type. It is the best I got so far
    tfnp: Any # Alias for the tensor backend pointer
    _: Any # Alias for the tensor backend pointer
    t: Any # Alias for the tensor backend pointer

    @classmethod
    def change_backend(cls, engine_backend: AvailableBackends, pykeops_enabled:bool = False, use_gpu: bool = True):

        cls.use_gpu = use_gpu

        print(f"Setting Backend To: {engine_backend}")

        if pykeops_enabled and is_pykeops_installed and is_numpy_installed:

            cls.pykeops_enabled = True
            cls.engine_backend = engine_backend

            import numpy as tfnp
            tfnp.reduce_sum = tfnp.sum
            tfnp.concat = tfnp.concatenate
            tfnp.constant = tfnp.array

            cls.engine_backend = engine_backend
            cls.tensor_backend_pointer['active_backend'] = tfnp
            cls.tensor_types = Union[tfnp.ndarray]  # tens

        else:
            cls.pykeops_enabled = False

            if engine_backend is AvailableBackends.jax and is_jax_installed:
                import jax.numpy as tfnp
                tfnp.reduce_sum = tfnp.sum
                tfnp.concat = tfnp.concatenate
                tfnp.constant = tfnp.array

                cls._set_active_backend_pointers(engine_backend, tfnp)
                cls.tensor_types = Union[tfnp.ndarray]  # tensor Types with respect the backend:

            elif engine_backend is AvailableBackends.tensorflow and is_tensorflow_installed:
                import tensorflow as tf

                if cls.use_gpu is False:
                    physical_devices = tf.config.list_physical_devices('GPU')
                    tf.config.set_visible_devices(physical_devices[1:], 'GPU')
                    logical_devices = tf.config.list_logical_devices('GPU')

                if DEBUG_MODE:
                    # To find out which devices your operations and tensors are assigned to
                    tf.debugging.set_log_device_placement(True) # TODO: Remove in production

                cls._set_active_backend_pointers(engine_backend, tf)
                cls.tensor_types = Union[tf.Tensor, tf.Variable]  # tensor Types with respect the backend:

                import logging
                tf.get_logger().setLevel(logging.ERROR)
                logging.getLogger("tensorflow").setLevel(logging.ERROR)


            elif engine_backend is AvailableBackends.numpy and is_numpy_installed:
                import numpy as tfnp
                tfnp.reduce_sum = tfnp.sum
                tfnp.concat = tfnp.concatenate
                tfnp.constant = tfnp.array

                cls._set_active_backend_pointers(engine_backend, tfnp)
                cls.tensor_types = Union[tfnp.ndarray]  # tensor Types with respect the backend

            else:
                raise AttributeError(f"Engine Backend: {engine_backend} cannot be used because the correspondent library"
                                     f"is not installed:")

    @classmethod
    def _set_active_backend_pointers(cls, engine_backend, tfnp):
        cls.engine_backend = engine_backend
        cls.tensor_backend_pointer['active_backend'] = tfnp
        # Add any alias here
        cls.tfnp = cls.tensor_backend_pointer['active_backend']
        cls._ = cls.tensor_backend_pointer['active_backend']
        cls.t = cls.tensor_backend_pointer['active_backend']

    @classmethod
    def describe_conf(cls):
        print(f"\n Using {cls.engine_backend} backend. \n")
        print(f"\n Using gpu: {cls.use_gpu}. \n")
        print(f"\n Using pykeops: {cls.pykeops_enabled}. \n")



BackendTensor.change_backend(DEFAULT_BACKEND)

