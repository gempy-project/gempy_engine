"""Host synchronizations of a GPU interpolation (CUDA only; run on Modal with
``scripts/benchmark_modal_model.py --pytest``).

Each sync stalls the host until the device drains, so their number is part of the performance
budget. The remaining ones come from data-dependent sizes (octree refinement, unique, nonzero) and
first-level checks. If a change legitimately moves the count, update the budget.
"""
import warnings

import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor

from .test_compute_scope import _fault_micro_model

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Measured on an A10 at b2434d1a: 258 syncs per compute. ~15% headroom.
SYNC_BUDGET = 300


@pytest.fixture(autouse=True)
def gpu_pykeops(restore_backend, monkeypatch):
    pytest.importorskip("pykeops")
    monkeypatch.setenv("GEMPY_FLAT_STACKS", "True")
    BackendTensor._change_backend(AvailableBackends.PYTORCH, use_pykeops=True, use_gpu=True)


def _syncs():
    compute_model(*_fault_micro_model())  # warm-up: KeOps compilation, CUDA context
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            compute_model(*_fault_micro_model())
            torch.cuda.synchronize()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return [w for w in caught if "synchronizing CUDA operation" in str(w.message)]


def test_interpolation_syncs_within_budget():
    syncs = _syncs()
    print(f"CUDA syncs per compute: {len(syncs)}")
    assert len(syncs) <= SYNC_BUDGET
