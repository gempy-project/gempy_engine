"""The stacked scalar-field combination with torch ops equals the per-stack loops bit for bit (_combine_stacked)."""
from types import SimpleNamespace

import numpy as np
import pytest

from gempy_engine.API.interp_single._masking_ops import (_combine_scalar_fields, _combine_stacked, _faults_mask,
                                                         _lithology_mask, _stackable)
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.stack_relation_type import StackRelationType as R

GRID, N_SP = 701, 9


@pytest.fixture(autouse=True)
def torch_backend(restore_backend):
    torch = pytest.importorskip("torch")
    BackendTensor._change_backend(AvailableBackends.PYTORCH)
    default = torch.get_default_dtype()
    yield
    torch.set_default_dtype(default)


def bitwise_equal(a, b) -> bool:
    """Same dtype, shape and bits (signed zeros and NaN payloads included); None only equals None."""
    if a is None or b is None:
        return a is b
    import torch
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.is_floating_point:
        as_int = {2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
        return torch.equal(a.contiguous().view(as_int), b.contiguous().view(as_int))
    return torch.equal(a, b)


def _output(rng, relation, dtype, gradients, limit_dtype=None):
    import torch
    total = GRID + N_SP

    def field():
        values = torch.as_tensor(rng.normal(size=total) * 2, dtype=dtype)
        values[:7], values[7:14] = 0.0, -0.0
        return values

    exported = ExportedFields(field(), *([field(), field(), field()] if gradients else []))
    exported.set_structure_values(reference_sp_position=torch.arange(0, N_SP, 3), slice_feature=slice(None),
                                  grid_size=GRID, macro_reference_size=total)
    if limit_dtype is not None:
        exported.scalar_field_at_surface_points = exported.scalar_field_everywhere[GRID:GRID + 3].to(limit_dtype)
    values_block = torch.as_tensor(rng.normal(size=(1, total)) + 1, dtype=dtype)
    values_block[0, 20:30] = -0.0
    return ScalarFieldOutput(weights=None, grid=None, exported_fields=exported, stack_relation=relation,
                             values_block=values_block)


def _per_stack(outputs, relations, gradients):
    structure = SimpleNamespace(masking_descriptor=relations)
    return _combine_scalar_fields(outputs, _lithology_mask(outputs, structure), _faults_mask(outputs, relations),
                                  relations, structure, gradients)


def _assert_identical(stacked, expected):
    assert len(stacked) == len(expected)
    for s, e in zip(stacked, expected):
        for a, b in [(s.squeezed_mask_array, e.squeezed_mask_array), (s.final_block, e.final_block),
                     (s.faults_block, e.faults_block)] + [
                (getattr(s.final_exported_fields, name), getattr(e.final_exported_fields, name))
                for name in ("_scalar_field", "_gx_field", "_gy_field", "_gz_field")]:
            assert bitwise_equal(a, b)


RELATIONS = {
    "perth": [R.FAULT] * 8 + [R.ERODE],
    "mixed": [R.FAULT, R.FAULT, R.ERODE, R.BASEMENT],
    "erode": [R.ERODE, R.ERODE, R.ERODE],
    "false_last": [R.FAULT, R.ERODE, False],
}


@pytest.mark.parametrize("default_dtype", ["float32", "float64"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("gradients", [False, True])
@pytest.mark.parametrize("relations", list(RELATIONS))
def test_stacked_combination_equals_per_stack(default_dtype, dtype, gradients, relations):
    import torch
    torch.set_default_dtype(getattr(torch, default_dtype))
    rng = np.random.default_rng(len(relations) + gradients)
    relations = RELATIONS[relations]
    outputs = [_output(rng, relation, getattr(torch, dtype), gradients) for relation in relations]

    assert _stackable(outputs, relations, relations, gradients)
    _assert_identical(_combine_stacked(outputs, relations, relations, gradients),
                      _per_stack(outputs, relations, gradients))


@pytest.mark.parametrize("limit_dtype", ["float32", "float64"])
def test_erosion_limit_of_another_dtype(limit_dtype):
    """A limit of another dtype is a 0-dim operand in the loops: it takes the field's dtype."""
    import torch
    rng = np.random.default_rng(3)
    relations = RELATIONS["mixed"]
    outputs = [_output(rng, relation, torch.float32, False, getattr(torch, limit_dtype)) for relation in relations]
    _assert_identical(_combine_stacked(outputs, relations, relations, False), _per_stack(outputs, relations, False))


def test_not_stackable():
    import torch
    rng = np.random.default_rng(5)
    erode = [_output(rng, R.ERODE, torch.float32, False) for _ in range(3)]
    assert not _stackable(erode, [R.ERODE, R.ONLAP, R.ERODE], [R.ERODE] * 3, False)
    assert not _stackable(erode, [R.ERODE, R.NULL_SPACE, R.ERODE], [R.ERODE] * 3, False)
    assert not _stackable(erode[:1], [R.ERODE], [R.ERODE], False)

    mixed_dtypes = erode[:2] + [_output(rng, R.ERODE, torch.float64, False)]
    assert not _stackable(mixed_dtypes, [R.ERODE] * 3, [R.ERODE] * 3, False)

    mixed_gradients = erode[:2] + [_output(rng, R.ERODE, torch.float32, True)]
    assert not _stackable(mixed_gradients, [R.ERODE] * 3, [R.ERODE] * 3, True)
    assert _stackable(mixed_gradients, [R.ERODE] * 3, [R.ERODE] * 3, False)


def test_bitwise_equal_tells_signed_zeros_apart():
    import torch
    assert torch.equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not bitwise_equal(torch.tensor([1.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float64))
