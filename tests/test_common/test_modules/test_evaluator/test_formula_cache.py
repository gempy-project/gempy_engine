"""Traced evaluation formulas are reused across computes of the same structure (compiled_evaluator._formula)."""
import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core import compute_scope
from gempy_engine.core.backend_tensor import BackendTensor

from ..test_compute_scope import _fault_micro_model, _fields


@pytest.fixture(autouse=True)
def flat_pykeops(restore_backend, monkeypatch):
    pytest.importorskip("pykeops")
    pytest.importorskip("torch")
    from gempy_engine.modules.evaluator import compiled_evaluator
    monkeypatch.setenv("GEMPY_FLAT_STACKS", "True")
    monkeypatch.setenv("GEMPY_FORMULA_CACHE", "True")
    BackendTensor._change_backend(AvailableBackends.PYTORCH, use_pykeops=True)
    compiled_evaluator.clear_formula_cache()
    yield
    compiled_evaluator.clear_formula_cache()


@pytest.fixture
def traces(monkeypatch):
    """Per formula request: True when it was traced, False when reused."""
    from gempy_engine.modules.evaluator import compiled_evaluator
    record, original = [], compiled_evaluator._formula

    def counting(structure, trace, backend):
        traced = []
        result = original(structure, lambda: traced.append(1) or trace(), backend)
        record.append(bool(traced))
        return result

    monkeypatch.setattr(compiled_evaluator, "_formula", counting)
    return record


def _moved(model):
    interpolation_input, options, descriptor = model
    interpolation_input.surface_points.sp_coords[5, 2] += 0.03  # a strata point: same structure, new data
    return interpolation_input, options, descriptor


def _compute(model):
    return _fields(compute_model(*model))


def test_same_structure_reuses_the_formulas_with_new_data(traces, monkeypatch):
    first = _compute(_fault_micro_model())
    assert traces and all(traces)
    traces.clear()
    repeated = _compute(_fault_micro_model())
    moved = _compute(_moved(_fault_micro_model()))
    assert traces and not any(traces)  # nothing traced again

    from gempy_engine.modules.evaluator import compiled_evaluator
    monkeypatch.setenv("GEMPY_FORMULA_CACHE", "False")
    compiled_evaluator.clear_formula_cache()
    cold_moved = _compute(_moved(_fault_micro_model()))
    for a, b in zip(first, repeated):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(moved, cold_moved):  # the reused formula with new data equals a new trace
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(first, moved))


def test_other_kernel_settings_trace_again(traces):
    _compute(_fault_micro_model())
    traces.clear()
    interpolation_input, options, descriptor = _fault_micro_model()
    options.kernel_options.range = 3.5
    _compute((interpolation_input, options, descriptor))
    assert traces and all(traces)


def test_cache_hits_match_a_new_trace(monkeypatch):
    _compute(_fault_micro_model())
    monkeypatch.setattr(compute_scope, "CHECK_COMPUTE_CACHE", True)
    _compute(_moved(_fault_micro_model()))  # raises AssertionError if a kept formula differs from a new trace
