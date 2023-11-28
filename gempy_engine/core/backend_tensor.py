from typing import Union, Any, Optional
import warnings

import numpy

from gempy_engine.config import (is_pykeops_installed, is_numpy_installed, is_tensorflow_installed,
                                 is_pytorch_installed,
                                 DEBUG_MODE, DEFAULT_BACKEND, AvailableBackends, DEFAULT_PYKEOPS, DEFAULT_TENSOR_DTYPE)

if is_pykeops_installed:
    import pykeops.numpy

if is_pytorch_installed:
    import torch


class BackendTensor:
    engine_backend: AvailableBackends

    pykeops_enabled: bool
    use_gpu: bool = True
    dtype: str = DEFAULT_TENSOR_DTYPE
    dtype_obj: Union[str, "torch.dtype"] = DEFAULT_TENSOR_DTYPE

    tensor_types: Union
    tensor_backend_pointer: dict = dict()  # Pycharm will infer the type. It is the best I got so far
    tfnp: numpy  # Alias for the tensor backend pointer
    _: Any  # Alias for the tensor backend pointer
    t: numpy  # Alias for the tensor backend pointer

    
    @classmethod
    def get_backend_string(cls) -> str:
        match (cls.use_gpu, cls.pykeops_enabled):
            case (True, True):
                return "GPU"
            case (False, True):
                return "CPU"
            case (False, _):
                return "CPU"

    @classmethod
    def change_backend_gempy(cls, engine_backend: AvailableBackends, use_gpu: bool = True, dtype: Optional[str] = None):
        cls._change_backend(engine_backend, pykeops_enabled=DEFAULT_PYKEOPS, use_gpu=use_gpu, dtype=dtype)

    @classmethod
    def _change_backend(cls, engine_backend: AvailableBackends, pykeops_enabled: bool = False, use_gpu: bool = True, dtype: Optional[str] = None):
        cls.dtype = DEFAULT_TENSOR_DTYPE if dtype is None else dtype
        cls.dtype_obj = cls.dtype
        match engine_backend:
            case (engine_backend.numpy):
                if is_numpy_installed is False:
                    raise AttributeError(
                        f"Engine Backend: {engine_backend} cannot be used because the correspondent library is not installed: numpy")

                # * Import a copy of numpy as tfnp
                from importlib.util import find_spec, module_from_spec

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    spec = find_spec('numpy')
                    tfnp = module_from_spec(spec)
                    spec.loader.exec_module(tfnp)

                cls._set_active_backend_pointers(engine_backend, tfnp)
                cls.tensor_types = Union[tfnp.ndarray]  # tensor Types with respect the backend

                cls._wrap_numpy_functions()

                match (pykeops_enabled, is_pykeops_installed, use_gpu):
                    case (True, True, True):
                        cls.pykeops_enabled = True
                        cls.use_gpu = True
                        cls._wrap_pykeops_functions()
                    case (True, True, False):
                        cls.pykeops_enabled = True
                        cls.use_gpu = False
                        cls._wrap_pykeops_functions()
                    case (True, False, _):
                        raise AttributeError(
                            f"Engine Backend: {engine_backend} cannot be used because the correspondent library is not installed: pykeops")
                    case (False, _, _):
                        cls.pykeops_enabled = False
                        cls.use_gpu = False
            case (engine_backend.tensorflow):
                if is_tensorflow_installed is False:
                    raise AttributeError(
                        f"Engine Backend: {engine_backend} cannot be used because the correspondent library is not installed: tensorflow")

                import tensorflow as tf
                experimental_numpy_api = tf.experimental.numpy
                cls._set_active_backend_pointers(engine_backend, experimental_numpy_api)  # * Here is where we set the tensorflow-numpy backend
                cls.tensor_types = Union[tf.Tensor, tf.Variable]  # tensor Types with respect the backend:

                from tensorflow.python.ops.numpy_ops import np_config
                np_config.enable_numpy_behavior(prefer_float32=True)

                physical_devices_gpu = tf.config.list_physical_devices('GPU')
                physical_devices_cpu = tf.config.list_physical_devices('CPU')

                tf.config.experimental.set_memory_growth(physical_devices_gpu[0], True)  # * This cannot be modified on run time
                tf.config.set_soft_device_placement(True)  # * This seems to allow changing the device on run time

                if DEBUG_MODE:
                    import logging
                    tf.get_logger().setLevel(logging.ERROR)
                    logging.getLogger("tensorflow").setLevel(logging.ERROR)
                    tf.debugging.set_log_device_placement(False)  # * To find out which devices your operations and tensors are assigned to

                match (pykeops_enabled, use_gpu):
                    # * device visibility can only be set once. In case of CPU and GPU visible, tf will use the GPU
                    # * The only thing I can do in here is to remove the GPU from the list of visible devices
                    case (False, True):
                        cls.use_gpu = True
                        cls.pykeops_enabled = False
                    case (False, False):
                        tf.config.set_visible_devices([], 'GPU')
                        cls.pykeops_enabled = False
                    case (True, _):
                        raise NotImplementedError("Pykeops is not compatible with Tensorflow yet")
            case (engine_backend.PYTORCH):
                if is_pytorch_installed is False:
                    raise AttributeError(
                        f"Engine Backend: {engine_backend} cannot be used because the correspondent library is not installed: pytorch")

                import torch as pytorch_copy
                cls._set_active_backend_pointers(engine_backend, pytorch_copy)  # * Here is where we set the tensorflow-numpy backend
                cls._wrap_pytorch_functions()
                cls.dtype_obj = pytorch_copy.float32 if cls.dtype == "float32" else pytorch_copy.float64
                cls.tensor_types = pytorch_copy.Tensor
            
                cls.pykeops_enabled = pykeops_enabled  # TODO: Make this compatible with pykeops
                if (pykeops_enabled):
                    import pykeops
                    cls._wrap_pykeops_functions()


            case (_):
                raise AttributeError(
                    f"Engine Backend: {engine_backend} cannot be used because the correspondent library"
                    f"is not installed:")

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
    def _wrap_pytorch_functions(cls):
        from torch import sum, repeat_interleave
        import torch
        
        def _sum(tensor, axis=None, dtype=None, keepdim=False):
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            
            return sum(tensor, axis, dtype=dtype, keepdim=keepdim)
        
        def _repeat(tensor, n_repeats, axis=None):
            return repeat_interleave(tensor, n_repeats, dim=axis)
        
        def _array(array_like, dtype=None):
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            if isinstance(array_like, torch.Tensor):
                return array_like
            else: 
                return torch.tensor(array_like, dtype=dtype)
        
        def _concatenate(tensors, axis=0, dtype=None):
            # Switch if tensor is numpy array or a torch tensor
            match type(tensors[0]):
                case numpy.ndarray:
                    return numpy.concatenate(tensors, axis=axis)
                case torch.Tensor:
                    return torch.cat(tensors, dim=axis)
        
        def _transpose(tensor, axes=None):
            return tensor.transpose(axes[0], axes[1])
        
        
        cls.tfnp.sum = _sum
        cls.tfnp.repeat = _repeat
        cls.tfnp.expand_dims = lambda tensor, axis: tensor
        cls.tfnp.invert = lambda tensor: ~tensor
        cls.tfnp.hstack = lambda tensors: torch.concat(tensors, dim=1)
        cls.tfnp.array = _array 
        cls.tfnp.to_numpy = lambda tensor: tensor.detach().numpy()
        cls.tfnp.min = lambda tensor, axis: tensor.min(axis=axis)[0]
        cls.tfnp.max = lambda tensor, axis: tensor.max(axis=axis)[0]
        cls.tfnp.rint = lambda tensor: tensor.round().type(torch.int32)
        cls.tfnp.vstack = lambda tensors: torch.cat(tensors, dim=0)
        cls.tfnp.copy = lambda tensor: tensor.clone()
        cls.tfnp.concatenate = _concatenate
        cls.tfnp.transpose = _transpose
        

    @classmethod
    def _wrap_pykeops_functions_(cls):
        def _exp(tensor):
            if type(tensor) == numpy.ndarray:
                return numpy.exp(tensor)
            elif type(tensor) == pykeops.numpy.LazyTensor:
                return tensor.exp()
            elif type(tensor) == pykeops.torch.LazyTensor:
                return tensor.exp()
            elif type(tensor) == torch.Tensor:
                return tensor.exp()
            

        def _sum(tensor, axis, dtype=None, keepdims=False):
            if type(tensor) == numpy.ndarray:
                return numpy.sum(tensor, axis=axis, keepdims=keepdims, dtype=dtype)
            elif type(tensor) == pykeops.numpy.LazyTensor:
                return tensor.sum(axis)
            elif type(tensor) == pykeops.torch.LazyTensor:
                return tensor.sum(axis)
            elif type(tensor) == torch.Tensor:
                return tensor.sum(axis, keepdims=keepdims, dtype=dtype)
            

        def _divide(tensor, other, dtype=None):
            if type(tensor) == numpy.ndarray:
                return numpy.divide(tensor, other, dtype=dtype)
            elif type(tensor) == pykeops.numpy.LazyTensor:
                return tensor / other
            elif type(tensor) == pykeops.torch.LazyTensor:
                return tensor / other
            elif type(tensor) == torch.Tensor:
                return tensor / other

        def _sqrt_fn(tensor):
            return tensor.sqrt()

        cls.tfnp.sqrt = _sqrt_fn
        cls.tfnp.sum = _sum
        cls.tfnp.exp = _exp
        cls.tfnp.divide = _divide

    @classmethod
    def _wrap_pykeops_functions(cls):
        torch_available = cls.engine_backend == AvailableBackends.PYTORCH
        
        def _exp(tensor):
            match tensor:
                case numpy.ndarray():
                    return numpy.exp(tensor)
                case pykeops.numpy.LazyTensor() | pykeops.torch.LazyTensor() if torch_available:
                    return tensor.exp()
                case torch.Tensor() if torch_available:
                    return tensor.exp()
                case _:
                    raise TypeError("Unsupported tensor type")

        def _sum(tensor, axis, dtype=None, keepdims=False):
            match tensor:
                case numpy.ndarray():
                    return numpy.sum(tensor, axis=axis, keepdims=keepdims, dtype=dtype)
                case pykeops.numpy.LazyTensor() | pykeops.torch.LazyTensor() if torch_available:
                    return tensor.sum(axis)
                case torch.Tensor() if torch_available:
                    return tensor.sum(axis, keepdims=keepdims, dtype=dtype)
                case _:
                    raise TypeError("Unsupported tensor type")

        def _divide(tensor, other, dtype=None):
            match tensor:
                case numpy.ndarray():
                    return numpy.divide(tensor, other, dtype=dtype)
                case pykeops.numpy.LazyTensor() | pykeops.torch.LazyTensor() if torch_available:
                    return tensor / other
                case torch.Tensor() if torch_available:
                    return tensor / other
                case _:
                    raise TypeError("Unsupported tensor type")

        def _sqrt_fn(tensor):
            return tensor.sqrt()

        cls.tfnp.sqrt = _sqrt_fn
        cls.tfnp.sum = _sum
        cls.tfnp.exp = _exp
        cls.tfnp.divide = _divide
        
    @classmethod
    def _wrap_numpy_functions(cls):
        cls.tfnp.cast = lambda tensor, dtype: tensor.astype(dtype)
        cls.tfnp.reduce_sum = cls.tfnp.sum
        cls.tfnp.concat = cls.tfnp.concatenate
        cls.tfnp.constant = cls.tfnp.array
        cls.tfnp.to_numpy = lambda tensor: tensor
        cls.tfnp.rint = lambda tensor: tensor.round().astype(numpy.int32)
    
    @classmethod
    def is_pykeops_enabled(cls):

        compatible_backend = BackendTensor.engine_backend == AvailableBackends.numpy or cls.engine_backend == AvailableBackends.PYTORCH
        if compatible_backend and BackendTensor.pykeops_enabled:
            return True
        else:
            return False

BackendTensor._change_backend(DEFAULT_BACKEND)
