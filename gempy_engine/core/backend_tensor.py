from typing import Union, Any

from ..config import is_pykeops_installed, is_numpy_installed, is_jax_installed, is_tensorflow_installed, DEBUG_MODE, \
    DEFAULT_BACKEND, AvailableBackends, DEFAULT_DTYPE


class BackendTensor():
    engine_backend: AvailableBackends

    pykeops_enabled: bool
    use_gpu: bool = True
    default_dtype = "float32"
    euclidean_distances_in_interpolation = True

    tensor_types: Union
    tensor_backend_pointer: dict = dict() # Pycharm will infer the type. It is the best I got so far
    tfnp: Any # Alias for the tensor backend pointer
    _: Any # Alias for the tensor backend pointer
    t: Any # Alias for the tensor backend pointer

    @classmethod
    def change_backend(cls, engine_backend: AvailableBackends):
        cls.engine_backend = engine_backend
        print(f"Setting Backend To: {engine_backend}")

        if engine_backend == AvailableBackends.numpy  and is_numpy_installed:
            cls.use_gpu = False
            cls.pykeops_enabled = False
            cls.default_dtype = DEFAULT_DTYPE
            tfnp = cls.setup_numpy_backend()

        elif engine_backend == AvailableBackends.numpyPykeosCPU:
            cls.use_gpu = False
            cls.pykeops_enabled = True
            cls.default_dtype = DEFAULT_DTYPE
            tfnp = cls.setup_numpy_backend()
        elif engine_backend == AvailableBackends.numpyPykeopsGPU:
            cls.use_gpu = True
            cls.pykeops_enabled = True
            cls.default_dtype = "float32"
            tfnp = cls.setup_numpy_backend()
        elif engine_backend == AvailableBackends.tensorflowCPU:
            cls.use_gpu = False
            cls.pykeops_enabled = False
            cls.default_dtype = DEFAULT_DTYPE

            tfnp = cls.setup_tensorflow_backend()
            physical_devices = tfnp.config.list_physical_devices('GPU')
            tfnp.config.set_visible_devices(physical_devices[1:], 'GPU')

        elif engine_backend == AvailableBackends.tensorflowGPU:
            cls.use_gpu = True
            cls.pykeops_enabled = False
            cls.default_dtype = "float32"

            tfnp = cls.setup_tensorflow_backend()
            physical_devices = tfnp.config.list_physical_devices('GPU')
            tfnp.config.experimental.set_memory_growth(physical_devices[0], True)

        else:
            raise AttributeError(f"Engine Backend: {engine_backend} cannot be used because the correspondent library"
                                 f"is not installed:")


        cls._set_active_backend_pointers(engine_backend, tfnp)
        cls.tensor_types = Union[tfnp.ndarray]  # tensor Types with respect the backend


    @classmethod
    def setup_tensorflow_backend(cls):
        import tensorflow as tfnp
        tfnp.sum = tfnp.reduce_sum
        if DEBUG_MODE:
            # To find out which devices your operations and tensors are assigned to
            tfnp.debugging.set_log_device_placement(True)
            import logging
            tfnp.get_logger().setLevel(logging.ERROR)
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
        return tfnp

    @classmethod
    def setup_numpy_backend(cls):
        import numpy as tfnp
        tfnp.reduce_sum = tfnp.sum
        tfnp.concat = tfnp.concatenate
        tfnp.constant = tfnp.array
        return tfnp

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

    @classmethod
    def get_backend(cls):
        if cls.use_gpu:
            return "GPU"
        else:
            return "CPU"


BackendTensor.change_backend(DEFAULT_BACKEND)