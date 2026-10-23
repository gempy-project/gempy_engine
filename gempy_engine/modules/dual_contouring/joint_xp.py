"""NumPy/Torch array operations for joint extraction: one code path on host or device.

``namespace(array)`` returns the operations of the array's library (and Torch
device). Functions mirror NumPy names and semantics for the subset joint
extraction uses; Torch is optional.
"""

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - Torch is an optional dependency
    torch = None

INT64_MAX = np.iinfo(np.int64).max


def is_tensor(a):
    return torch is not None and isinstance(a, torch.Tensor)


def namespace(*arrays):
    """Torch operations on the first tensor's device if any argument is a tensor, else NumPy."""
    for a in arrays:
        if is_tensor(a):
            return TorchOps(a.device)
    return NUMPY


def host(a):
    """Host NumPy copy-free view (tensors are detached and moved)."""
    return a.detach().cpu().numpy() if is_tensor(a) else np.asarray(a)


class NumpyOps:
    torch = False
    int32, int64, float64, bool = np.int32, np.int64, np.float64, np.bool_

    def asarray(self, a, dtype=None):
        return np.asarray(host(a), dtype=dtype)

    def astype(self, a, dtype):
        return a.astype(dtype)

    def copy(self, a):
        return a.copy()

    def zeros(self, shape, dtype=np.float64):
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=np.float64):
        return np.ones(shape, dtype=dtype)

    def empty(self, shape, dtype=np.float64):
        return np.empty(shape, dtype=dtype)

    def full(self, shape, value, dtype=np.float64):
        return np.full(shape, value, dtype=dtype)

    def arange(self, n, dtype=np.int64):
        return np.arange(n, dtype=dtype)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def where(self, condition, a, b):
        return np.where(condition, a, b)

    def flatnonzero(self, a):
        return np.flatnonzero(a)

    def nonzero(self, a):
        return np.nonzero(a)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def max(self, a, axis=None):
        return a.max(axis=axis)

    def min(self, a, axis=None):
        return a.min(axis=axis)

    def abs(self, a):
        return np.abs(a)

    def sign(self, a):
        return np.sign(a)

    def isnan(self, a):
        return np.isnan(a)

    def isfinite(self, a):
        return np.isfinite(a)

    def cumsum(self, a, axis=0):
        return np.cumsum(a, axis=axis)

    def flip(self, a, axis):
        return np.flip(a, axis=axis)

    def roll(self, a, shift, axis):
        return np.roll(a, shift, axis=axis)

    def repeat(self, a, repeats, axis=0):
        return np.repeat(a, repeats, axis=axis)

    def take_along_axis(self, a, indices, axis):
        return np.take_along_axis(a, indices, axis=axis)

    def sort(self, a, axis=-1):
        return np.sort(a, axis=axis)

    def argsort(self, a):
        """Stable ascending order of a 1-D array."""
        return np.argsort(a, kind='stable')

    def searchsorted(self, sorted_values, values):
        return np.searchsorted(sorted_values, values)

    def unique(self, a, return_index=False, return_inverse=False, return_counts=False):
        """Sorted unique values of a 1-D array, with NumPy's optional first index, inverse and counts."""
        result = np.unique(a, return_index=return_index, return_inverse=return_inverse,
                           return_counts=return_counts)
        if isinstance(result, tuple) and return_inverse:
            position = 1 + return_index
            result = result[:position] + (result[position].reshape(-1),) + result[position + 1:]
        return result

    def cross(self, a, b):
        return np.cross(a, b)

    def norm(self, a, axis=-1):
        return np.linalg.norm(a, axis=axis)

    def dot_rows(self, a, b):
        return np.einsum('ij,ij->i', a, b)


class TorchOps(NumpyOps):
    torch = True

    def __init__(self, device):
        self.device = device
        self.int32, self.int64, self.float64, self.bool = torch.int32, torch.int64, torch.float64, torch.bool

    def _dtype(self, dtype):
        mapping = {np.int32: torch.int32, np.int64: torch.int64, np.float64: torch.float64, np.bool_: torch.bool,
                   int: torch.int64, float: torch.float64, bool: torch.bool}
        return mapping.get(dtype, dtype)

    def asarray(self, a, dtype=None):
        if is_tensor(a):
            return a.to(device=self.device, dtype=self._dtype(dtype) if dtype is not None else a.dtype)
        a = np.asarray(a)
        # Torch cannot view negative-stride arrays (e.g. ``[::-1]`` slices).
        return torch.as_tensor(np.ascontiguousarray(a) if a.ndim else a, dtype=self._dtype(dtype),
                               device=self.device)

    def astype(self, a, dtype):
        return a.to(self._dtype(dtype))

    def copy(self, a):
        return a.clone()

    def zeros(self, shape, dtype=np.float64):
        return torch.zeros(shape, dtype=self._dtype(dtype), device=self.device)

    def ones(self, shape, dtype=np.float64):
        return torch.ones(shape, dtype=self._dtype(dtype), device=self.device)

    def empty(self, shape, dtype=np.float64):
        return torch.empty(shape, dtype=self._dtype(dtype), device=self.device)

    def full(self, shape, value, dtype=np.float64):
        return torch.full(shape if isinstance(shape, tuple) else (shape,), value, dtype=self._dtype(dtype),
                          device=self.device)

    def arange(self, n, dtype=np.int64):
        return torch.arange(n, dtype=self._dtype(dtype), device=self.device)

    def concatenate(self, arrays, axis=0):
        return torch.cat([self.asarray(a) for a in arrays], dim=axis)

    def stack(self, arrays, axis=0):
        return torch.stack([self.asarray(a) for a in arrays], dim=axis)

    def _operand(self, value, like):
        return value if is_tensor(value) else torch.as_tensor(value, dtype=like.dtype, device=self.device)

    def where(self, condition, a, b):
        like = a if is_tensor(a) else b if is_tensor(b) else None
        if like is None:
            return torch.where(condition, torch.as_tensor(a, device=self.device), torch.as_tensor(b, device=self.device))
        return torch.where(condition, self._operand(a, like), self._operand(b, like))

    def flatnonzero(self, a):
        return torch.nonzero(a.reshape(-1)).reshape(-1)

    def nonzero(self, a):
        return torch.nonzero(a, as_tuple=True)

    def maximum(self, a, b):
        like = a if is_tensor(a) else b
        return torch.maximum(self._operand(a, like), self._operand(b, like))

    def minimum(self, a, b):
        like = a if is_tensor(a) else b
        return torch.minimum(self._operand(a, like), self._operand(b, like))

    def max(self, a, axis=None):
        return a.max() if axis is None else a.max(dim=axis).values

    def min(self, a, axis=None):
        return a.min() if axis is None else a.min(dim=axis).values

    def abs(self, a):
        return torch.abs(a)

    def sign(self, a):
        return torch.sign(a)

    def isnan(self, a):
        return torch.isnan(a)

    def isfinite(self, a):
        return torch.isfinite(a)

    def cumsum(self, a, axis=0):
        return torch.cumsum(a, dim=axis)

    def flip(self, a, axis):
        return torch.flip(a, dims=(axis,))

    def roll(self, a, shift, axis):
        return torch.roll(a, shift, dims=axis)

    def repeat(self, a, repeats, axis=0):
        if not is_tensor(repeats) and not isinstance(repeats, int):
            repeats = torch.as_tensor(repeats, device=self.device)
        return torch.repeat_interleave(a, repeats, dim=axis)

    def take_along_axis(self, a, indices, axis):
        return torch.take_along_dim(a, indices, dim=axis)

    def sort(self, a, axis=-1):
        return torch.sort(a, dim=axis, stable=True).values

    def argsort(self, a):
        return torch.argsort(a, stable=True)

    def searchsorted(self, sorted_values, values):
        return torch.searchsorted(sorted_values, values)

    def unique(self, a, return_index=False, return_inverse=False, return_counts=False):
        values, inverse, counts = torch.unique(a, sorted=True, return_inverse=True, return_counts=True)
        result = [values]
        if return_index:
            first = torch.full((len(values),), len(a), dtype=torch.int64, device=self.device)
            first.scatter_reduce_(0, inverse, torch.arange(len(a), device=self.device), reduce='amin')
            result.append(first)
        if return_inverse:
            result.append(inverse)
        if return_counts:
            result.append(counts)
        return tuple(result) if len(result) > 1 else values

    def cross(self, a, b):
        return torch.linalg.cross(a, b)

    def norm(self, a, axis=-1):
        return torch.linalg.norm(a, dim=axis)

    def dot_rows(self, a, b):
        return (a*b).sum(dim=1)


NUMPY = NumpyOps()


def extraction_namespace():
    """Array operations for joint extraction, chosen by ``GEMPY_JOINT_DEVICE``.

    ``auto`` (default): Torch on the GPU when the engine runs PyTorch there,
    else NumPy. ``numpy``: always NumPy. ``torch``: Torch on the engine's
    device (GPU if the engine uses one, else CPU).
    """
    import os
    from ...config import AvailableBackends
    from ...core.backend_tensor import BackendTensor as BT
    choice = os.environ.get('GEMPY_JOINT_DEVICE', 'auto').strip().lower()
    if choice not in ('auto', 'numpy', 'torch'):
        raise ValueError(f'invalid GEMPY_JOINT_DEVICE {choice!r}: expected auto, numpy or torch')
    on_gpu = (torch is not None and BT.engine_backend is AvailableBackends.PYTORCH and BT.use_gpu
              and torch.cuda.is_available())
    if choice == 'numpy' or (choice == 'auto' and not on_gpu):
        return NUMPY
    if torch is None:
        raise ValueError('GEMPY_JOINT_DEVICE=torch requires PyTorch')
    return TorchOps(torch.device('cuda') if on_gpu else torch.device('cpu'))
