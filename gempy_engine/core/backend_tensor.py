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
    
PYKEOPS= DEFAULT_PYKEOPS

# * Import a copy of numpy as tfnp
from importlib.util import find_spec, module_from_spec

class BackendTensor:
    engine_backend: AvailableBackends

    pykeops_enabled: bool = False
    use_pykeops: bool = False
    use_gpu: bool = True
    dtype: str = DEFAULT_TENSOR_DTYPE
    dtype_obj: Union[str, "torch.dtype"] = DEFAULT_TENSOR_DTYPE

    tensor_types: Union
    tensor_backend_pointer: dict = dict()  # Pycharm will infer the type. It is the best I got so far
    tfnp: numpy  # Alias for the tensor backend pointer
    _: Any  # Alias for the tensor backend pointer
    t: numpy  # Alias for the tensor backend pointer
    
    COMPUTE_GRADS: bool = False

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
    def change_backend_gempy(cls, engine_backend: AvailableBackends, use_gpu: bool = False,
                             dtype: Optional[str] = None, grads:bool = False):
        cls._change_backend(engine_backend, use_pykeops=PYKEOPS, use_gpu=use_gpu, dtype=dtype,
                            grads=grads)

    @classmethod
    def _change_backend(cls, engine_backend: AvailableBackends, use_pykeops: bool = False,
                        use_gpu: bool = False, dtype: Optional[str] = None, grads:bool = False):
        cls.dtype = DEFAULT_TENSOR_DTYPE if dtype is None else dtype
        cls.dtype_obj = cls.dtype
        match engine_backend:
            case (engine_backend.numpy):
                if is_numpy_installed is False:
                    raise AttributeError(
                        f"Engine Backend: {engine_backend} cannot be used because the correspondent library is not installed: numpy")


                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    spec = find_spec('numpy')
                    tfnp = module_from_spec(spec)
                    spec.loader.exec_module(tfnp)

                cls._set_active_backend_pointers(engine_backend, tfnp)
                cls.tensor_types = Union[tfnp.ndarray]  # tensor Types with respect the backend

                cls._wrap_numpy_functions()

                match (use_pykeops, is_pykeops_installed, use_gpu):
                    case (True, True, True):
                        cls.use_pykeops = True
                        cls.use_gpu = True
                        cls._wrap_pykeops_functions()
                    case (True, True, False):
                        cls.use_pykeops = True
                        cls.use_gpu = False
                        cls._wrap_pykeops_functions()
                    case (True, False, _):
                        raise AttributeError(
                            f"Engine Backend: {engine_backend} cannot be used because the correspondent library is not installed: pykeops")
                    case (False, _, _):
                        cls.use_pykeops = False
                        cls.use_gpu = False

            case (engine_backend.PYTORCH):
                if is_pytorch_installed is False:
                    raise AttributeError(
                        f"Engine Backend: {engine_backend} cannot be used because the correspondent library is not installed: pytorch")

                import torch as pytorch_copy
                cls._set_active_backend_pointers(engine_backend, pytorch_copy)  # * Here is where we set the tensorflow-numpy backend
                cls._wrap_pytorch_functions()
                cls.dtype_obj = pytorch_copy.float32 if cls.dtype == "float32" else pytorch_copy.float64
                cls.tensor_types = pytorch_copy.Tensor

                torch.set_num_threads(torch.get_num_threads())  # Use all available threads
                cls.COMPUTE_GRADS = grads  # Store the grads setting
                if grads is False:
                    cls._torch_no_grad_context = torch.no_grad()
                    cls._torch_no_grad_context.__enter__()
                else:
                    # If there was a previous context, exit it first
                    if hasattr(cls, '_torch_no_grad_context') and cls._torch_no_grad_context is not None:
                        try:
                            cls._torch_no_grad_context.__exit__(None, None, None)
                        except:
                            pass  # Context might already be exited
                    cls._torch_no_grad_context = None
                    torch.set_grad_enabled(True)
                    
                cls.use_pykeops = use_pykeops  # TODO: Make this compatible with pykeops
                if (use_pykeops):
                    import pykeops
                    cls._wrap_pykeops_functions()
                
                if (use_gpu):
                    cls.use_gpu = True
                    # cls.tensor_backend_pointer['active_backend'].set_default_device("cuda")
                    # Check if CUDA is available
                    if not pytorch_copy.cuda.is_available():
                        raise RuntimeError("GPU requested but CUDA is not available in PyTorch")
                    if False: # * (Miguel) this slows down the code a lot
                        # Check if CUDA device is available
                        if not pytorch_copy.cuda.device_count():
                            raise RuntimeError("GPU requested but no CUDA device is available in PyTorch")
                        # Set default device to CUDA
                        cls.device = pytorch_copy.device("cuda")
                        pytorch_copy.set_default_device("cuda")
                        print(f"GPU enabled. Using device: {cls.device}")
                        print(f"GPU device count: {pytorch_copy.cuda.device_count()}")
                        print(f"Current GPU device: {pytorch_copy.cuda.current_device()}")
                else:
                    cls.use_gpu = False
                    cls.device = pytorch_copy.device("cpu")
                    pytorch_copy.set_default_device("cpu")
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
        from torch import sum, repeat_interleave, isclose
        import torch

        def _sum(tensor, axis=None, dtype=None, keepdims=False):
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)

            return sum(tensor, axis, dtype=dtype)

        def _repeat(tensor, n_repeats, axis=None):
            return repeat_interleave(tensor, n_repeats, dim=axis)

        def _array(array_like, dtype=None):
            if array_like is None:
                return None
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            if isinstance(array_like, torch.Tensor):
                if dtype is None: return array_like
                else: return array_like.type(dtype)
            else:
                # Ensure numpy arrays are contiguous before converting to torch tensor
                if isinstance(array_like, numpy.ndarray):
                    if not array_like.flags.c_contiguous:
                        array_like = numpy.ascontiguousarray(array_like)

                return torch.tensor(array_like, dtype=dtype)

        def _concatenate(tensors, axis=0, dtype=None):
            # Switch if tensor is numpy array or a torch tensor
            match type(tensors[0]):
                case _ if any(isinstance(t, numpy.ndarray) for t in tensors):
                    return numpy.concatenate(tensors, axis=axis)
                case _ if isinstance(tensors[0], torch.Tensor):
                    return torch.cat(tensors, dim=axis)
            raise TypeError(f"Unsupported tensor type:  {type(tensors[0])}")

        def _transpose(tensor, axes=None):
            return tensor.transpose(axes[0], axes[1])
        

        def _packbits(tensor, axis=None, bitorder="big"):
            """
            Pack boolean values into uint8 bytes along the specified axis.
            For a (4, n) tensor with axis=0, this packs every 4 bits into nibbles,
            then pads to create full bytes.
            """
            # Convert to uint8 if boolean
            if tensor.dtype == torch.bool:
                tensor = tensor.to(torch.uint8)

            if axis == 0:
                # Pack along axis 0 (rows)
                n_rows, n_cols = tensor.shape
                n_output_rows = (n_rows + 7) // 8  # Round up to nearest byte boundary

                # Pad with zeros if we don't have multiples of 8 rows
                if n_rows % 8 != 0:
                    padding_rows = 8 - (n_rows % 8)
                    padding = torch.zeros(padding_rows, n_cols, dtype=torch.uint8, device=tensor.device)
                    tensor = torch.cat([tensor, padding], dim=0)

                # Reshape to group every 8 rows together: (n_output_rows, 8, n_cols)
                tensor_reshaped = tensor.view(n_output_rows, 8, n_cols)

                # Define bit positions (powers of 2)
                if bitorder == "little":
                    # Little endian: LSB first [1, 2, 4, 8, 16, 32, 64, 128]
                    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                                          dtype=torch.uint8, device=tensor.device).view(1, 8, 1)
                else:
                    # Big endian: MSB first [128, 64, 32, 16, 8, 4, 2, 1]
                    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                                          dtype=torch.uint8, device=tensor.device).view(1, 8, 1)

                # Pack bits: multiply each bit by its power and sum along the 8-bit dimension
                packed = (tensor_reshaped * powers).sum(dim=1)  # Shape: (n_output_rows, n_cols)

                return packed

            elif axis == 1:
                # Pack along axis 1 (columns) 
                n_rows, n_cols = tensor.shape
                n_output_cols = (n_cols + 7) // 8

                # Pad with zeros if needed
                if n_cols % 8 != 0:
                    padding_cols = 8 - (n_cols % 8)
                    padding = torch.zeros(n_rows, padding_cols, dtype=torch.uint8, device=tensor.device)
                    tensor = torch.cat([tensor, padding], dim=1)

                # Reshape: (n_rows, n_output_cols, 8)
                tensor_reshaped = tensor.view(n_rows, n_output_cols, 8)

                # Define bit positions
                if bitorder == "little":
                    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                                          dtype=torch.uint8, device=tensor.device).view(1, 1, 8)
                else:
                    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1],
                                          dtype=torch.uint8, device=tensor.device).view(1, 1, 8)

                packed = (tensor_reshaped * powers).sum(dim=2)  # Shape: (n_rows, n_output_cols)
                return packed

            else:
                raise NotImplementedError(f"packbits not implemented for axis={axis}")


        def _to_numpy(tensor):
            """Convert tensor to numpy array, handling GPU tensors properly"""
            if hasattr(tensor, 'device') and tensor.device.type == 'cuda':
                # Move to CPU first, then detach and convert to numpy
                return tensor.cpu().detach().numpy()
            elif hasattr(tensor, 'detach'):
                # CPU tensor, just detach and convert
                return tensor.detach().numpy()
            else:
                # Not a torch tensor, return as-is
                return tensor

        def _fill_diagonal(tensor, value):
            """Fill the diagonal of a 2D tensor with the given value"""
            if tensor.dim() != 2:
                raise ValueError("fill_diagonal only supports 2D tensors")
            diagonal_indices = torch.arange(min(tensor.size(0), tensor.size(1)))
            tensor[diagonal_indices, diagonal_indices] = value
            return tensor

        cls.tfnp.sum = _sum
        cls.tfnp.repeat = _repeat
        cls.tfnp.expand_dims = lambda tensor, axis: tensor
        cls.tfnp.invert = lambda tensor: ~tensor
        cls.tfnp.flip = lambda tensor, axis: tensor.flip(axis)
        cls.tfnp.hstack = lambda tensors: torch.concat(tensors, dim=1)
        cls.tfnp.array = _array
        cls.tfnp.to_numpy = _to_numpy
        cls.tfnp.min = lambda tensor, axis: tensor.min(axis=axis)[0]
        cls.tfnp.max = lambda tensor, axis: tensor.max(axis=axis)[0]
        cls.tfnp.rint = lambda tensor: tensor.round().type(torch.int32)
        cls.tfnp.vstack = lambda tensors: torch.cat(tensors, dim=0)
        cls.tfnp.copy = lambda tensor: tensor.clone()
        cls.tfnp.concatenate = _concatenate
        cls.tfnp.transpose = _transpose
        cls.tfnp.geomspace = lambda start, stop, step: torch.logspace(start, stop, step, base=10)
        cls.tfnp.abs = lambda tensor, dtype = None: tensor.abs().type(dtype) if dtype is not None else tensor.abs()
        cls.tfnp.tile = lambda tensor, repeats: tensor.repeat(repeats)
        cls.tfnp.ravel = lambda tensor: tensor.flatten()
        cls.tfnp.packbits = _packbits
        cls.tfnp.ascontiguousarray = lambda tensor: tensor.contiguous()
        cls.tfnp.fill_diagonal = _fill_diagonal
        cls.tfnp.isclose = lambda a, b, rtol=1e-05, atol=1e-08, equal_nan=False: isclose(
            a,
            torch.tensor(b, dtype=a.dtype, device=a.device),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan
        )

    @classmethod
    def _wrap_pykeops_functions(cls):
        torch_available = cls.engine_backend == AvailableBackends.PYTORCH

        def _exp(tensor):
            match tensor:
                case numpy.ndarray():
                    return numpy.exp(tensor)
                case pykeops.numpy.LazyTensor() | pykeops.torch.LazyTensor(): 
                    return tensor.exp()
                case torch.Tensor() if torch_available:
                    return tensor.exp()
                case _:
                    raise TypeError("Unsupported tensor type")


        @torch.jit.ignore
        def _sum(tensor, axis=None, dtype=None, keepdims=False):
            match tensor:
                case numpy.ndarray():
                    return numpy.sum(tensor, axis=axis, keepdims=keepdims, dtype=dtype)
                case pykeops.numpy.LazyTensor() | pykeops.torch.LazyTensor():
                    return tensor.sum(axis)
                case torch.Tensor() if torch_available:
                    if isinstance(dtype, str):
                        dtype = getattr(torch, dtype)
                    return tensor.sum(axis, keepdims=keepdims, dtype=dtype)
                case _:
                    raise TypeError("Unsupported tensor type")

        def _divide(tensor, other, dtype=None):
            match tensor:
                case numpy.ndarray():
                    return numpy.divide(tensor, other, dtype=dtype)
                case pykeops.numpy.LazyTensor() | pykeops.torch.LazyTensor():
                    return tensor / other
                case torch.Tensor() if torch_available:
                    return tensor / other
                case _:
                    raise TypeError("Unsupported tensor type")

        def _sqrt_fn(tensor):
            match tensor:
                case numpy.ndarray():
                    return numpy.sqrt(tensor)
                case pykeops.numpy.LazyTensor() | pykeops.torch.LazyTensor(): 
                    return tensor.sqrt()
                case torch.Tensor() if torch_available:
                    return tensor.sqrt()
                case _:
                    raise TypeError("Unsupported tensor type")

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
