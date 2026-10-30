"""Launch-overhead budget of an interpolation: torch ops and KeOps reductions per compute.

Small models are dominated by per-op launch cost, not arithmetic, so the number of torch ops and
KeOps calls is the performance budget. Counts do not depend on the device, so this runs on CPU.
If a change legitimately moves a count, update the budget with the new measured value.
"""
import collections
import sys

import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor

from .test_compute_scope import N_LEVELS, _fault_micro_model

# Measured on the model below (2 stacks, micro contacts, 3 octree levels) in a clean environment
# (no engine .env): 4,359 ops, ~1,000 of them the solutions' regular-grid expansion; 7,597 before the
# per-compute stack cache, the KeOps launch changes and the compiled evaluation, 4,624 before the
# host-index surface-point split and the shared assemblies. ~15% headroom.
TORCH_OPS_BUDGET = 5000


@pytest.fixture(autouse=True)
def flat_pykeops(restore_backend, monkeypatch):
    pytest.importorskip("pykeops")
    pytest.importorskip("torch")
    monkeypatch.setenv("GEMPY_FLAT_STACKS", "True")
    BackendTensor._change_backend(AvailableBackends.PYTORCH, use_pykeops=True)


def _count(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode
    from pykeops.common.lazy_tensor import GenericLazyTensor

    reductions = []
    original = GenericLazyTensor.fixvariables

    def fixvariables(self):
        reductions.append(sum(type(v) is list for v in self.variables))
        return original(self)

    monkeypatch.setattr(GenericLazyTensor, "fixvariables", fixvariables)

    class Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.ops = collections.Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.ops[str(func.overloadpacket)] += 1
            return func(*args, **(kwargs or {}))

    compute_model(*_fault_micro_model())  # warm-up: KeOps compilation, lazy imports
    reductions.clear()
    with Count() as counter:
        compute_model(*_fault_micro_model())
    return counter.ops, reductions


def test_keops_formulas_are_built_once_per_stack(monkeypatch):
    _, reductions = _count(monkeypatch)
    # Each stack's evaluation is built once per compute and called at every level: the micro stack's
    # (field and gradient in one reduction) and the fault stack's stacked (fused) one. The micro
    # correction is still built per level.
    assert len(reductions) == 2 + N_LEVELS
    assert sum(reductions) == 0  # float constants travel in one packed parameter per reduction


def test_torch_ops_within_budget(monkeypatch):
    ops, _ = _count(monkeypatch)
    total = sum(ops.values())
    print(f"torch ops per compute: {total}; top: {ops.most_common(8)}", file=sys.stderr)
    assert total <= TORCH_OPS_BUDGET
