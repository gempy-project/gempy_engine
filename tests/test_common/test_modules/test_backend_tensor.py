import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor


def test_numpy_backend_resets_gradient_flag():
    pytest.importorskip('torch')
    BackendTensor._change_backend(AvailableBackends.PYTORCH, grads=True)
    assert BackendTensor.COMPUTE_GRADS
    BackendTensor._change_backend(AvailableBackends.numpy)
    assert not BackendTensor.COMPUTE_GRADS
