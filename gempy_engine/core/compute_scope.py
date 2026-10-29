"""Per-compute cache of the stack data that does not change between octree levels.

Within one ``compute_model`` call a stack's preprocessed surface points and orientations, its weights,
the data side of its evaluation kernels and its micro fit are the same at every octree level and in
every mesh-extraction query; only the evaluation grid changes. ``compute_scope()`` keeps them for the
duration of the call.

Entries are keyed by host facts (the input object, the stack index and the hash of the kernel
options, which covers their settings but not the condition numbers the solver stores as tensors),
never by tensor contents, so a lookup costs no device work. Outside a scope nothing is cached.
Cached tensors are shared between levels and must not be modified in place.

``GEMPY_CHECK_COMPUTE_CACHE=True`` recomputes every cached value on a hit and raises if it differs
(tests and debugging only: the comparison synchronizes).
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

CHECK_COMPUTE_CACHE = os.getenv("GEMPY_CHECK_COMPUTE_CACHE", "False").lower() in ("true", "1", "t", "y", "yes")


@dataclass
class StackCache:
    sp_internal: Any = None
    ori_internal: Any = None
    fault_values_ref: Any = None
    fault_values_rest: Any = None
    weights: Any = None
    weights_host: Any = None
    debug: Optional[dict] = None
    data_side: dict = field(default_factory=dict)
    micro_fitted: bool = False
    micro: Any = None  # (MicroCorrection | None, MicroPointResults | None)
    fingerprint_key: Any = None
    fingerprint: Optional[str] = None


class _Scope:
    def __init__(self):
        self._entries = {}

    def stack(self, owner, stack_index: int, options_key) -> StackCache:
        key = (id(owner), stack_index, options_key)
        entry = self._entries.get(key)
        if entry is None:
            # Holding the owner keeps its id from being reused by another object during the scope.
            entry = self._entries[key] = (owner, StackCache())
        return entry[1]


_active: ContextVar[Optional[_Scope]] = ContextVar("gempy_compute_scope", default=None)


@contextmanager
def compute_scope():
    token = _active.set(_Scope())
    try:
        yield
    finally:
        _active.reset(token)


def stack_cache(owner, stack_index: int, options_key=None) -> Optional[StackCache]:
    """The cache of stack ``stack_index`` of ``owner`` in the active scope, or None outside one."""
    scope = _active.get()
    return None if scope is None else scope.stack(owner, stack_index, options_key)


def checking() -> bool:
    """Whether cache hits are recomputed and compared (``GEMPY_CHECK_COMPUTE_CACHE``)."""
    return CHECK_COMPUTE_CACHE


def stack_cache_of(solver_input) -> Optional[StackCache]:
    """The cache attached to a solver or evaluator input, if any."""
    return getattr(getattr(solver_input, "solver_input", solver_input), "stack_cache", None)


def assert_same(cached, fresh, what: str, rtol: float = 0.0):
    """Raise if a cached value differs from its recomputation (GEMPY_CHECK_COMPUTE_CACHE only).

    ``rtol`` > 0 allows rounding-level differences, for values that are recomputed from a
    different evaluation at every level (fault values at the data points).
    """
    import numpy as np
    from .backend_tensor import BackendTensor

    def same(a, b):
        if a is None or b is None:
            return a is b
        a, b = BackendTensor.t.to_numpy(a), BackendTensor.t.to_numpy(b)
        if rtol:
            return a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=rtol, equal_nan=True)
        return np.array_equal(a, b, equal_nan=True)

    if not same(cached, fresh):
        raise AssertionError(f"compute cache: cached {what} differs from its recomputation")
