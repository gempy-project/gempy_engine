"""Stacked (FLAT) KeOps evaluation must keep every block's fault drift.

Regression: the scalar kernel marked fault-drift rows as the trailing rows of
the whole stacked covariance, so every faulted stack except the last one in a
stacked chunk lost its fault drift (Model 7 stack 1 was off by 13% of its range).
"""

import importlib
import os

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT


def _keops_gpu():
    torch = pytest.importorskip('torch')
    pytest.importorskip('pykeops')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    BT._change_backend(AvailableBackends.PYTORCH, use_gpu=True, use_pykeops=True, dtype='float64', grads=False)
    return torch


def _restore(torch):
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
    torch.set_default_device('cpu')


def _model7_scalar_fields(flat):
    from gempy_engine.API.model.model_api import compute_model
    from tests.fixtures.model7_combination import model7_combination_factory

    previous = os.environ.get('GEMPY_FLAT_STACKS')
    os.environ['GEMPY_FLAT_STACKS'] = 'True' if flat else 'False'
    try:
        inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=False)
        options.evaluation_options.compute_scalar_gradient = False
        solution = compute_model(inputs, options, descriptor)
    finally:
        if previous is None:
            os.environ.pop('GEMPY_FLAT_STACKS')
        else:
            os.environ['GEMPY_FLAT_STACKS'] = previous
    return [BT.t.to_numpy(output.exported_fields.scalar_field) for output in solution.octrees_output[0].outputs]


def test_flat_keops_scalar_fields_match_per_stack_evaluation():
    torch = _keops_gpu()
    try:
        reference = _model7_scalar_fields(flat=False)
        flat = _model7_scalar_fields(flat=True)
    finally:
        _restore(torch)
    for stack, (expected, actual) in enumerate(zip(reference, flat)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10, err_msg=f'stack {stack}')


def _prepared_drift_sampler():
    import gempy_engine.API.model.model_api as api
    from gempy_engine.API.dual_contouring.adaptive_fault_sampler import prepare_fault_drift_sampler
    from tests.fixtures.model7_combination import model7_combination_factory

    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=True)
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.mesh_extraction_overlap = 'none'
    prepared = {}

    def capture(data_descriptor=None, interpolation_input=None, options=None, octree_list=None):
        prepared['sampler'] = prepare_fault_drift_sampler(data_descriptor, interpolation_input, options, octree_list)
        prepared['extent'] = BT.t.to_numpy(octree_list[0].grid.octree_grid.orthogonal_extent).reshape(3, 2)
        return []

    original = api.dual_contouring_multi_scalar
    api.dual_contouring_multi_scalar = capture
    try:
        api.compute_model(inputs, options, descriptor)
    finally:
        api.dual_contouring_multi_scalar = original
    return prepared['sampler'], prepared['extent']


@pytest.mark.parametrize('backend', ['numpy', 'keops_gpu'])
def test_drift_sampler_batches_match_individual_queries(backend):
    torch = None
    if backend == 'keops_gpu':
        torch = _keops_gpu()
    else:
        BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
    try:
        sampler, extent = _prepared_drift_sampler()
        rng = np.random.default_rng(3)
        points = [rng.uniform(extent[:, 0], extent[:, 1], size=(size, 3)) for size in (40, 7, 23)]
        requests = [(points[0], (1, 2), 'scalar'), (points[1], (0, 2), 'gradient'),
                    (points[2], (2, 0, 1), 'scalar'), (points[0], (1,), 'gradient')]
        answers = sampler['query_batch'](requests)
        for (pts, stacks, kind), answer in zip(requests, answers):
            values, gradients = sampler['query'](pts, stacks=stacks, gradients=True)
            np.testing.assert_allclose(answer, values if kind == 'scalar' else gradients, rtol=0, atol=1e-12)
        with pytest.raises(ValueError, match='invalid_query_kind'):
            sampler['query_batch']([(points[0], (1,), 'hessian')])
    finally:
        if torch is not None:
            _restore(torch)
