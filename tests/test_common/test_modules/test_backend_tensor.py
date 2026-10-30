from types import SimpleNamespace

import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor


def test_numpy_backend_resets_gradient_flag():
    pytest.importorskip('torch')
    BackendTensor._change_backend(AvailableBackends.PYTORCH, grads=True)
    assert BackendTensor.COMPUTE_GRADS
    BackendTensor._change_backend(AvailableBackends.numpy)
    assert not BackendTensor.COMPUTE_GRADS


@pytest.mark.parametrize("reserved_share, released", [(0.25, False), (0.26, True)])
def test_gpu_cache_is_released_only_above_the_kept_share(monkeypatch, reserved_share, released):
    torch = pytest.importorskip("torch")
    calls = []
    total = SimpleNamespace(total_memory=1000)
    monkeypatch.setattr(BackendTensor, "use_gpu", True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: total)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: int(reserved_share * total.total_memory))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("released"))

    BackendTensor.clear_gpu_memory()

    assert bool(calls) is released
