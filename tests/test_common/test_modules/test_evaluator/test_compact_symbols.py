"""Renumbering a traced formula's symbolic variables equals tracing it again with only the used ones."""
import numpy as np
import pytest

from gempy_engine.modules.evaluator.compiled_evaluator import _compact_symbols


def test_compact_symbols_equals_a_trace_without_the_unused_placeholders():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pykeops")
    from pykeops.torch import LazyTensor

    def trace(i_x, i_w):
        x, w = LazyTensor((i_x, 3, 0)), LazyTensor((i_w, 1, 1))
        y = LazyTensor(torch.as_tensor(np.random.default_rng(0).random((5, 1, 3)), dtype=torch.float32))
        return ((x - y) ** 2).sum(-1) * w

    compact, kept = _compact_symbols(trace(1, 3))  # placeholders 0 and 2 never entered the formula
    fresh = trace(0, 1)
    assert kept == [1, 3]
    assert compact.formula == fresh.formula.replace(f"Var({id(fresh.variables[0])},", f"Var({id(compact.variables[0])},")
    assert sorted(compact.symbolic_variables) == sorted(fresh.symbolic_variables)

    points = torch.as_tensor(np.random.default_rng(1).random((4, 3)), dtype=torch.float32)
    weights = torch.ones((5, 1), dtype=torch.float32)
    expected = fresh.sum(axis=0, call=False)(points, weights)
    assert torch.equal(compact.sum(axis=0, call=False)(points, weights), expected)

    unchanged, kept = _compact_symbols(fresh)
    assert unchanged is fresh and kept == [0, 1]
