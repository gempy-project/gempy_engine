import gc
from typing import Union, Any, Optional
import warnings

import numpy
import numpy as np

from gempy_engine.config import (is_pykeops_installed, is_numpy_installed, is_tensorflow_installed,
                                 is_pytorch_installed,
                                 DEBUG_MODE, DEFAULT_BACKEND, AvailableBackends, DEFAULT_PYKEOPS, DEFAULT_TENSOR_DTYPE)

if is_pykeops_installed:
    import pykeops.numpy

if is_pytorch_installed:
    import torch

PYKEOPS = DEFAULT_PYKEOPS

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
                             dtype: Optional[str] = None, grads: bool = False):
        cls._change_backend(engine_backend, use_pykeops=PYKEOPS, use_gpu=use_gpu, dtype=dtype,
                            grads=grads)
        
    @classmethod
    def change_backend_gempy(cls, engine_backend: AvailableBackends, use_gpu: bool = False,
                             dtype: Optional[str] = None, grads: bool = False):
        cls._change_backend(engine_backend, use_pykeops=PYKEOPS, use_gpu=use_gpu, dtype=dtype,
                            grads=grads)

    @classmethod
    def _change_backend(cls, engine_backend: AvailableBackends, use_pykeops: bool = False,
                        use_gpu: bool = False, dtype: Optional[str] = None, grads: bool = False):
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
                    # Test
                    if True:  # * (Miguel) this slows down the code a lot
                        # Check if CUDA device is available
                        if not pytorch_copy.cuda.device_count():
                            raise RuntimeError("GPU requested but no CUDA device is available in PyTorch")
                        # Set default device to CUDA
                        cls.device = pytorch_copy.device("cuda")
                        pytorch_copy.set_default_device("cuda")
                        torch.set_default_device("cuda")
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
        import torch
        _true_torch_zeros = torch._C._VariableFunctions.zeros
        _true_torch_ones = torch._C._VariableFunctions.ones
        _true_torch_eye = torch._C._VariableFunctions.eye
        _true_torch_sum = torch._C._VariableFunctions.sum
        _true_torch_repeat_interleave = torch._C._VariableFunctions.repeat_interleave
        _true_torch_isclose = torch._C._VariableFunctions.isclose

        def _sum(tensor, axis=None, dtype=None, keepdims=False):
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)

            return _true_torch_sum(tensor, axis, dtype=dtype)

        def _repeat(tensor, n_repeats, axis=None):
            return _true_torch_repeat_interleave(tensor, n_repeats, dim=axis)

        def _array(array_like, dtype=None):
            if array_like is None:
                return None

            # Resolve string dtypes safely
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)

            # 1. Fast Path: It's already a Tensor
            if isinstance(array_like, torch.Tensor):
                return array_like if dtype is None else array_like.to(dtype)

            # 2. Slow Path: NumPy / Lists (The "Dirty" Data)
            # Use bool(array_like) instead of len() > 0 for a cleaner empty-check
            is_list_of_arrays = (isinstance(array_like, (list, tuple)) and bool(array_like) and isinstance(array_like[0], numpy.ndarray))

            # Check for memory misalignment
            is_misaligned_array = (isinstance(array_like, numpy.ndarray) and any(s % array_like.itemsize != 0 for s in array_like.strides))

            # Apply the fix if either dirty condition is met
            if is_list_of_arrays or is_misaligned_array:
                array_like = numpy.ascontiguousarray(np.array(array_like))

            # 3. Final Conversion
            return torch.tensor(array_like, dtype=dtype, device=cls.device)

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

        def _packbits(binary_tensor, axis=None, bitorder=None):
            """
            Converts a binary tensor (shape: bits, N) to standard integers.
            Replaces packbits to avoid multi-byte splitting at high resolutions.
            """
            num_bits = binary_tensor.shape[0]

            # Create an array of [1, 2, 4, 8, 16...] depending on how many bits there are
            powers = (2 ** torch.arange(num_bits, device=binary_tensor.device, dtype=torch.int64))

            # Multiply the bits by their power and sum them up!
            # unsqueeze(1) ensures it broadcasts correctly against (bits, N)
            return (binary_tensor.to(torch.int64) * powers.unsqueeze(1)).sum(dim=0).unsqueeze(0)

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

        def _zeros(shape, dtype=None, device=None):
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            return _true_torch_zeros(shape, dtype=dtype, device=cls.device)

        def _ones(shape, dtype=None, device=None):
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            return _true_torch_ones(shape, dtype=dtype, device=cls.device)

        def _eye(n, dtype=None, device=None):
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype)
            return _true_torch_eye(n, dtype=dtype, device=cls.device)

        import torch

        def _torch_intersect1d_indices(codes_i: torch.Tensor, codes_j: torch.Tensor, max_memory_mb: float = 250.0, assume_unique: bool = True,
                                       return_indices: bool = False):
            """
            Finds the intersection of two 1D tensors and returns the sorted common values 
            and their corresponding indices, mimicking np.intersect1d(assume_unique=True).

            Dynamically routes to a fast broadcasting method or a memory-safe isin method 
            based on the expected VRAM footprint.
            """
            N = codes_i.numel()
            M = codes_j.numel()

            # Calculate the theoretical memory footprint of the boolean broadcast matrix
            # A boolean tensor in PyTorch takes 1 byte per element
            memory_footprint_mb = (N * M) / (1024 ** 2)

            if memory_footprint_mb <= max_memory_mb:
                # ==========================================
                # ROUTE 1: FAST BROADCASTING (Low Memory)
                # ==========================================
                # Creates an N x M boolean mask
                matches = codes_i.unsqueeze(1) == codes_j.unsqueeze(0)
                idx_i, idx_j = torch.where(matches)
                common = codes_i[idx_i]

                # Sort to perfectly match numpy's default behavior
                sort_idx = torch.argsort(common)
                return common[sort_idx], idx_i[sort_idx], idx_j[sort_idx]

            else:
                # ==========================================
                # ROUTE 2: MEMORY-SAFE ISIN (High Memory)
                # ==========================================
                # 1. Find elements of codes_i in codes_j
                mask_i = torch.isin(codes_i, codes_j)
                idx_i_unsorted = torch.nonzero(mask_i).squeeze(-1)
                common_unsorted = codes_i[idx_i_unsorted]

                # 2. Find elements of codes_j in codes_i
                mask_j = torch.isin(codes_j, codes_i)
                idx_j_unsorted = torch.nonzero(mask_j).squeeze(-1)

                # 3. Align and sort both index arrays based on the actual common values
                sort_i = torch.argsort(common_unsorted)
                sort_j = torch.argsort(codes_j[idx_j_unsorted])

                common_sorted = common_unsorted[sort_i]
                idx_i_sorted = idx_i_unsorted[sort_i]
                idx_j_sorted = idx_j_unsorted[sort_j]

                return common_sorted, idx_i_sorted, idx_j_sorted

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
        cls.tfnp.abs = lambda tensor, dtype=None: tensor.abs().type(dtype) if dtype is not None else tensor.abs()
        cls.tfnp.tile = lambda tensor, repeats: tensor.repeat(repeats)
        cls.tfnp.ravel = lambda tensor: tensor.flatten()
        cls.tfnp.packbits = _packbits
        cls.tfnp.ascontiguousarray = lambda tensor: tensor.contiguous()
        cls.tfnp.fill_diagonal = _fill_diagonal
        cls.tfnp.isclose = lambda a, b, rtol=1e-05, atol=1e-08, equal_nan=False: _true_torch_isclose(
            a,
            torch.tensor(b, dtype=a.dtype, device=a.device),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan
        )

        cls.tfnp.zeros = _zeros
        cls.tfnp.ones = _ones
        cls.tfnp.eye = _eye
        cls.tfnp.intersect1d = _torch_intersect1d_indices

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

        def _sum(tensor, axis=None, dtype=None, keepdims=False):
            if isinstance(tensor, numpy.ndarray):
                return numpy.sum(tensor, axis=axis, keepdims=keepdims, dtype=dtype)

            # Handle LazyTensors (KeOps)
            # We check for the attribute or common base if imports are tricky, 
            # but explicit isinstance is safest if they are already in scope.
            import pykeops
            if isinstance(tensor, (pykeops.numpy.LazyTensor, pykeops.torch.LazyTensor)):
                return tensor.sum(axis)

            if torch_available and isinstance(tensor, torch.Tensor):
                if isinstance(dtype, str):
                    dtype = getattr(torch, dtype)
                return tensor.sum(axis, keepdims=keepdims, dtype=dtype)

            raise TypeError(f"Unsupported tensor type: {type(tensor)}")

        def _divide(tensor, other, dtype=None):
            if isinstance(tensor, numpy.ndarray):
                return numpy.divide(tensor, other, dtype=dtype)

            import pykeops
            if isinstance(tensor, (pykeops.numpy.LazyTensor, pykeops.torch.LazyTensor)):
                return tensor / other

            if torch_available and isinstance(tensor, torch.Tensor):
                return tensor / other

            raise TypeError(f"Unsupported tensor type: {type(tensor)}")

        def _sqrt_fn(tensor):
            if isinstance(tensor, numpy.ndarray):
                return numpy.sqrt(tensor)

            import pykeops
            if isinstance(tensor, (pykeops.numpy.LazyTensor, pykeops.torch.LazyTensor)):
                return tensor.sqrt()

            if torch_available and isinstance(tensor, torch.Tensor):
                return tensor.sqrt()

            raise TypeError(f"Unsupported tensor type: {type(tensor)}")

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

    @classmethod
    def clear_gpu_memory(cls):
        if BackendTensor.use_gpu:
            # 1. Barrier: Wait for all GPU threads to finish
            torch.cuda.synchronize()

        # 2. GC: Safely destroy orphaned Python/C++ objects
        gc.collect()

        if BackendTensor.use_gpu:
            # 3. Release VRAM: Give the memory back to the OS
            torch.cuda.empty_cache()


BackendTensor._change_backend(DEFAULT_BACKEND)
