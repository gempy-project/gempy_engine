from typing import Union, Any

import numpy

from ..config import is_pykeops_installed, is_numpy_installed, is_tensorflow_installed, DEBUG_MODE, \
    DEFAULT_BACKEND, AvailableBackends


class BackendTensor:
    engine_backend: AvailableBackends

    pykeops_enabled: bool
    use_gpu: bool = True

    tensor_types: Union
    tensor_backend_pointer: dict = dict()  # Pycharm will infer the type. It is the best I got so far
    tfnp: numpy  # Alias for the tensor backend pointer
    _: Any  # Alias for the tensor backend pointer
    t: numpy  # Alias for the tensor backend pointer

    @classmethod
    def change_backend(cls, engine_backend: AvailableBackends, pykeops_enabled: bool = False, use_gpu: bool = True):
        match engine_backend:
            case (engine_backend.numpy):
                if is_numpy_installed is False:
                    raise AttributeError(f"Engine Backend: {engine_backend} cannot be used because the correspondent library is not installed: numpy")
            
                # * Import a copy of numpy as tfnp
                from importlib.util import find_spec, module_from_spec
                spec = find_spec('numpy')
                tfnp = module_from_spec(spec)
                spec.loader.exec_module(tfnp)
                
                # ? DEP: Now we are using numpy as default
                tfnp.reduce_sum = tfnp.sum
                tfnp.concat = tfnp.concatenate
                tfnp.constant = tfnp.array

                cls._set_active_backend_pointers(engine_backend, tfnp)
                cls.tensor_types = Union[tfnp.ndarray]  # tensor Types with respect the backend

                match (pykeops_enabled, use_gpu):
                    case (True, True):
                        cls.pykeops_enabled = is_pykeops_installed
                        cls.use_gpu = True
                        cls._wrap_pykeops_functions()
                    case (True, False):
                        cls.pykeops_enabled = is_pykeops_installed
                        cls.use_gpu = False
                        cls._wrap_pykeops_functions()
                    case (False, _):
                        cls.pykeops_enabled = False
                        cls.use_gpu = False
            case (engine_backend.tensorflow):
                if is_tensorflow_installed is False:
                    raise AttributeError(f"Engine Backend: {engine_backend} cannot be used because the correspondent library is not installed: tensorflow")
                
                import tensorflow as tf
                from tensorflow.python.ops.numpy_ops import np_config
                experimental_numpy_api = tf.experimental.numpy
                cls._set_active_backend_pointers(engine_backend, experimental_numpy_api)  # * Here is where we set the tensorflow-numpy backend
                cls.tensor_types = Union[tf.Tensor, tf.Variable]  # tensor Types with respect the backend:
                np_config.enable_numpy_behavior()

                physical_devices = tf.config.list_physical_devices('GPU')
                
                if DEBUG_MODE:
                    
                    import logging
                    tf.get_logger().setLevel(logging.ERROR)
                    logging.getLogger("tensorflow").setLevel(logging.ERROR)
                    tf.debugging.set_log_device_placement(True) # To find out which devices your operations and tensors are assigned to

                match (pykeops_enabled, use_gpu):
                    case (False, True):
                        tf.config.experimental.set_memory_growth(physical_devices[0], True)
                        cls.use_gpu = True
                    case (False, False):
                        tf.config.set_visible_devices(physical_devices[1:], 'GPU')
                    case (True, _):
                        raise NotImplementedError("Pykeops is not compatible with Tensorflow yet")
            case (_):
                raise AttributeError(f"Engine Backend: {engine_backend} cannot be used because the correspondent library"
                                     f"is not installed:")
        
        # cls._wrap_backend_functions()

    @classmethod
    def _set_active_backend_pointers(cls, engine_backend, tfnp):
        cls.engine_backend = engine_backend
        cls.tensor_backend_pointer['active_backend'] = tfnp
        # Add any alias here
        cls.tfnp = cls.tensor_backend_pointer['active_backend']
        cls._ = cls.tensor_backend_pointer['active_backend']
        cls.t = cls.tensor_backend_pointer['active_backend']

        print(f"Setting Backend To: {engine_backend}")

    @classmethod
    def describe_conf(cls):
        print(f"\n Using {cls.engine_backend} backend. \n")
        print(f"\n Using gpu: {cls.use_gpu}. \n")
        print(f"\n Using pykeops: {cls.pykeops_enabled}. \n")

    @classmethod
    def _wrap_pykeops_functions(cls):
        
        # ! This is rewriting the whole numpy function
        cls.tfnp.sum = lambda tensor, axis, keepdims=False: tensor.sum(axis=axis)
        cls.tfnp.sqrt = lambda tensor: tensor.sqrt()

BackendTensor.change_backend(DEFAULT_BACKEND)
