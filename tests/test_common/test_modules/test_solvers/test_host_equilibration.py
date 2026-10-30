"""Ruiz equilibration on the host (small CUDA systems) agrees with the torch iterations."""
import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.modules.solver import linear_system_transform


@pytest.fixture(autouse=True)
def torch_backend(restore_backend):
    pytest.importorskip("torch")
    BackendTensor._change_backend(AvailableBackends.PYTORCH)


def _system(rng, n_rows, rhs_shape, dtype):
    import torch
    a = rng.normal(size=(n_rows, n_rows)) * 10.0 ** rng.integers(-3, 4, size=(n_rows, 1))
    a = a @ a.T
    a[2], a[:, 2] = 0, 0  # a zero row stays unscaled
    return torch.as_tensor(a, dtype=dtype), torch.as_tensor(rng.normal(size=rhs_shape), dtype=dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("n_rows, rhs_shape", [(6, (6,)), (40, (40, 1)), (7, (7, 3))])
def test_host_equilibration_matches_torch(monkeypatch, dtype, n_rows, rhs_shape):
    import torch
    matrix, rhs = _system(np.random.default_rng(n_rows), n_rows, rhs_shape, getattr(torch, dtype))
    host = linear_system_transform._host_equilibration(matrix, rhs, 10, 1e-2)
    monkeypatch.setattr(linear_system_transform, "_equilibrate_on_host", lambda *args: False)
    expected = linear_system_transform.equilibrate_symmetric_system(matrix, rhs, 10, 1e-2)

    assert (host.iterations, host.converged) == (expected.iterations, expected.converged)
    for a, b in ((host.matrix, expected.matrix), (host.rhs, expected.rhs), (host.factors, expected.factors)):
        assert a.shape == b.shape and a.dtype == b.dtype
        # torch's vectorized CPU sqrt is not correctly rounded, NumPy's and CUDA's are: a few ulps here
        assert torch.allclose(a, b, rtol=1e-5 if dtype == "float32" else 1e-12, atol=0)


def test_host_path_only_for_cuda_systems_without_autograd():
    import torch
    matrix, rhs = _system(np.random.default_rng(0), 6, (6,), torch.float32)
    assert not linear_system_transform._equilibrate_on_host(matrix, rhs)
    if torch.cuda.is_available():
        assert linear_system_transform._equilibrate_on_host(matrix.cuda(), rhs.cuda())
        assert not linear_system_transform._equilibrate_on_host(matrix.cuda().requires_grad_(), rhs.cuda())
