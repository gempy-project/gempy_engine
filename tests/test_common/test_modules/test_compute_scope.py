"""The per-compute stack cache must not change results: same fields with and without it."""
import contextlib

import numpy as np
import pytest

import gempy_engine.API.model.model_api as model_api
import gempy_engine.core.compute_scope as compute_scope
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions, Orientations, SurfacePoints, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid, RegularGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.micro_points import MicroPoints
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure

N_LEVELS = 3


@pytest.fixture(autouse=True)
def flat_pykeops(restore_backend, monkeypatch):
    """The production setup. (With NumPy the engine deep-copies the input per level, so nothing is cached.)"""
    pytest.importorskip("pykeops")
    pytest.importorskip("torch")
    monkeypatch.setenv("GEMPY_FLAT_STACKS", "True")
    BackendTensor._change_backend(AvailableBackends.PYTORCH, use_pykeops=True)


def _fault_micro_model():
    """A fault cutting a stratigraphic stack with micro contacts, on a 3-level octree."""
    fault_sp = np.array([[0.5, 0.2, 0.2], [0.5, 0.8, 0.2], [0.5, 0.2, 0.8], [0.5, 0.8, 0.8]])
    strata_sp = np.array([[0.2, 0.2, 0.4], [0.8, 0.2, 0.4], [0.2, 0.8, 0.4], [0.8, 0.8, 0.4]])
    relations = np.array([[False, True], [False, False]])
    micro = MicroPoints(np.array([[0.56, 0.5, 0.54], [0.5, 0.5, 0.5], fault_sp[0]]),
                        np.repeat(np.eye(3)[None] * 3, 3, axis=0), np.zeros(3), np.array([1, 1, 1]))
    grid = EngineGrid.from_regular_grid(RegularGrid(orthogonal_extent=[0, 1, 0, 1, 0, 1], regular_grid_shape=[2, 2, 2]))
    ii = InterpolationInput(SurfacePoints(np.vstack((fault_sp, strata_sp))), Orientations(
        np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.4]]), np.array([[1., 0., 0.], [0., 0., 1.]])),
        grid, micro_points=micro)
    descriptor = InputDataDescriptor(TensorsStructure(np.array([4, 4])), StacksStructure(
        np.array([4, 4]), np.array([1, 1]), np.array([1, 1]), [StackRelationType.FAULT, StackRelationType.ERODE],
        faults_relations=relations, faults_input_data=[None, None]))
    options = InterpolationOptions.from_args(range=3., c_o=1., uni_degree=1, mesh_extraction=False)
    options.micro_options.enabled = True
    options.evaluation_options.number_octree_levels = N_LEVELS
    return ii, options, descriptor


def _fields(solutions):
    return [np.asarray(BackendTensor.t.to_numpy(output.scalar_fields.exported_fields.scalar_field_everywhere)).copy()
            for level in solutions.octrees_output for output in level.outputs]


def test_cached_compute_matches_uncached(monkeypatch):
    cached = _fields(compute_model(*_fault_micro_model()))
    monkeypatch.setattr(model_api, "compute_scope", contextlib.nullcontext)
    uncached = _fields(compute_model(*_fault_micro_model()))
    assert len(cached) == len(uncached) == 2 * N_LEVELS
    for a, b in zip(cached, uncached):
        np.testing.assert_array_equal(a, b)


def test_cache_hits_match_recomputation(monkeypatch):
    monkeypatch.setattr(compute_scope, "CHECK_COMPUTE_CACHE", True)
    compute_model(*_fault_micro_model())  # raises AssertionError on any mismatch


def test_weights_are_resolved_once_per_stack_and_compute(monkeypatch):
    """Later levels reuse the weights: no solve and no weight-cache lookup (which hashes the inputs)."""
    import gempy_engine.API.interp_single._stack_ops as stack_ops
    calls = []
    original = stack_ops.compute_weights
    monkeypatch.setattr(stack_ops, "compute_weights", lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))
    compute_model(*_fault_micro_model())
    assert len(calls) == 2
    calls.clear()
    monkeypatch.setattr(model_api, "compute_scope", contextlib.nullcontext)
    compute_model(*_fault_micro_model())
    assert len(calls) == 2 * N_LEVELS


def test_no_caching_outside_a_compute():
    assert compute_scope.stack_cache(object(), 0) is None
    with compute_scope.compute_scope():
        owner = object()
        assert compute_scope.stack_cache(owner, 0) is compute_scope.stack_cache(owner, 0)
        assert compute_scope.stack_cache(owner, 1) is not compute_scope.stack_cache(owner, 0)
    assert compute_scope.stack_cache(owner, 0) is None
