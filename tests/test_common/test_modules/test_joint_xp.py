"""Joint extraction on Torch arrays gives the NumPy results exactly (CPU Torch stands in for the GPU)."""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from gempy_engine.API.dual_contouring import joint_topology as core
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.dual_contouring.joint_cell_branches import CORNERS
from gempy_engine.modules.dual_contouring.joint_xp import NUMPY, TorchOps, extraction_namespace, host

from .test_joint_contacts import merge_case
from .test_joint_fault_topology import bank_case
from .test_joint_topology import mixed_leaves, plane_fields

TORCH = TorchOps(torch.device('cpu'))


@pytest.fixture(autouse=True)
def cpu_float64():
    saved = dict(engine_backend=BT.engine_backend, use_gpu=BT.use_gpu,
                 use_pykeops=BT.use_pykeops, dtype=BT.dtype, grads=BT.COMPUTE_GRADS)
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype="float64", grads=False)
    yield
    BT._change_backend(**saved)


def test_adapter_operations_match_numpy():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 20, 200)
    tensor = torch.as_tensor(values)
    for np_part, torch_part in zip(NUMPY.unique(values, return_index=True, return_inverse=True, return_counts=True),
                                   TORCH.unique(tensor, return_index=True, return_inverse=True, return_counts=True)):
        np.testing.assert_array_equal(np_part, host(torch_part))
    np.testing.assert_array_equal(NUMPY.argsort(values), host(TORCH.argsort(tensor)))
    counts = rng.integers(0, 3, 200)
    np.testing.assert_array_equal(NUMPY.repeat(values, counts), host(TORCH.repeat(tensor, torch.as_tensor(counts))))
    grid = values.reshape(50, 4)
    order = rng.permuted(np.tile(np.arange(4), (50, 1)), axis=1)
    np.testing.assert_array_equal(NUMPY.take_along_axis(grid, order, axis=1),
                                  host(TORCH.take_along_axis(torch.as_tensor(grid), torch.as_tensor(order), axis=1)))
    np.testing.assert_array_equal(NUMPY.flatnonzero(grid > 9), host(TORCH.flatnonzero(torch.as_tensor(grid) > 9)))
    np.testing.assert_array_equal(NUMPY.where(grid > 9, grid, -1), host(TORCH.where(torch.as_tensor(grid) > 9,
                                                                                  torch.as_tensor(grid), -1)))
    np.testing.assert_array_equal(NUMPY.max(grid, axis=1), host(TORCH.max(torch.as_tensor(grid), axis=1)))
    np.testing.assert_array_equal(NUMPY.searchsorted(np.sort(values), values[:10]),
                                  host(TORCH.searchsorted(torch.sort(tensor).values, tensor[:10])))


def test_extraction_namespace_follows_environment(monkeypatch):
    monkeypatch.setenv('GEMPY_JOINT_DEVICE', 'numpy')
    assert extraction_namespace() is NUMPY
    monkeypatch.setenv('GEMPY_JOINT_DEVICE', 'torch')
    assert extraction_namespace().torch
    monkeypatch.setenv('GEMPY_JOINT_DEVICE', 'auto')
    assert extraction_namespace() is NUMPY  # No engine GPU here.
    monkeypatch.setenv('GEMPY_JOINT_DEVICE', 'cuda')
    with pytest.raises(ValueError, match='GEMPY_JOINT_DEVICE'):
        extraction_namespace()


def _on_torch(args):
    args = list(args)
    args[0], args[1], args[4] = (TORCH.asarray(args[i]) for i in (0, 1, 4))
    return args


def _assert_same(expected, actual):
    for key in ('vertices', 'seam_edges', 'junction_cells', 'leaf_origins', 'leaf_spans'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual['vertex_keys'] == expected['vertex_keys']
    for key in ('faces', 'affected_faces'):
        for a, b in zip(actual[key], expected[key]):
            np.testing.assert_array_equal(a, b)
    assert actual['diagnostics'] == expected['diagnostics']


@pytest.mark.parametrize('onlap', [False, True])
@pytest.mark.parametrize('permutation', [(0, 1, 2), (2, 0, 1)])
def test_torch_matches_numpy_on_mixed_leaf_contacts(onlap, permutation):
    origins, spans = mixed_leaves(permutation)
    corners = origins[:, None, :]+spans[:, None, None]*CORNERS
    samples = plane_fields(corners.reshape(-1, 3), permutation, onlap)[0].reshape(2, -1, 8)
    args = (origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples, [0, 1], [0, 0],
            [R.ONLAP if onlap else R.ERODE, R.BASEMENT], [[0.], [0.]])
    kwargs = dict(sample_fields=lambda points: plane_fields(points, permutation, onlap), include_reference=True)
    expected = core.extract_adaptive_topology(*args, **kwargs)
    actual = core.extract_adaptive_topology(*_on_torch(args), **kwargs)
    _assert_same(expected, actual)
    assert len(expected['seam_edges'])
    reference, torch_reference = expected['reference'], actual['reference']
    np.testing.assert_array_equal(torch_reference['vertices'], reference['vertices'])
    assert torch_reference['vertex_keys'] == reference['vertex_keys']
    for a, b in zip(torch_reference['faces'], reference['faces']):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('case', [dict(), dict(fine=True), dict(planes=(3.3, 5.7)),
                                  dict(erosion=(2.05, .5), fine=True)])
def test_torch_matches_numpy_with_fault_merge(case):
    args, kwargs = merge_case(**case)
    _assert_same(core.extract_adaptive_topology(*args, **kwargs),
                 core.extract_adaptive_topology(*_on_torch(args), **kwargs))


@pytest.mark.parametrize('bank', [0, 1])
def test_torch_matches_numpy_on_fault_banks(bank):
    args, kwargs = bank_case(bank)
    expected = core.extract_adaptive_topology(*args, **kwargs, defer_emission=True)
    actual = core.extract_adaptive_topology(*_on_torch(args), **kwargs, defer_emission=True)
    for result in (expected, actual):
        result['faces'], result['affected_faces'] = core.emit_adaptive_triangles(
            result.pop('_triangle_plan'), result.pop('_key_to_id'))
    _assert_same(expected, actual)
    assert actual['bank'] == expected['bank'] == bank
