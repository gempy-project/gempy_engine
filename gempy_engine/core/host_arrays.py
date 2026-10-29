"""Small constant arrays built on the host and uploaded in one non-blocking copy.

Writing Python scalars into device tensors element by element (``z[i, j] = 1``) costs one host
to device copy, and one synchronization, per write. Index patterns that only depend on sizes
are built with NumPy instead and uploaded at once.
"""
import numpy as np

from .backend_tensor import BackendTensor
from ..config import AvailableBackends


def upload(array: np.ndarray):
    """``array`` as a backend tensor with the same dtype; on CUDA through pinned memory, non-blocking."""
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH:
        return array
    import torch
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    device = BackendTensor.device
    if getattr(device, "type", device) == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor
