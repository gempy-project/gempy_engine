"""Fixed-weight production-drift sampler and the shared weight snapshot, without meshing."""

import copy
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gempy_engine.API.dual_contouring import fault_drift_sampler as sampler_api
from gempy_engine.API.dual_contouring import fixed_weight_snapshots as snapshot_api
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.options import InterpolationOptions
from tests.fixtures.model7_combination import model7_combination_factory


@pytest.fixture(autouse=True)
def restore_backend():
    saved = dict(engine_backend=BT.engine_backend, use_gpu=BT.use_gpu,
                 use_pykeops=BT.use_pykeops, dtype=BT.dtype, grads=BT.COMPUTE_GRADS)
    yield
    BT._change_backend(**saved)


@pytest.fixture
def numpy_backend():
    BT._change_backend(engine_backend=AvailableBackends.numpy, use_gpu=False,
                       use_pykeops=False, dtype='float64', grads=False)


def _keops_gpu():
    torch = pytest.importorskip('torch')
    pytest.importorskip('pykeops')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    BT._change_backend(AvailableBackends.PYTORCH, use_gpu=True, use_pykeops=True, dtype='float64', grads=False)
    return torch


def _prepare(monkeypatch, *, levels=2, surface_levels=2, relation=None, inside=None, finite=False):
    api = importlib.import_module('gempy_engine.API.model.model_api')
    inputs, options, descriptor = model7_combination_factory(
        number_octree_levels=levels, mesh_extraction=True,
        custom_grid=np.array([[x, 500, z] for x in (100, 600, 1200, 1800, 2400)
                              for z in (50, 300, 600, 950)]))
    if finite:
        descriptor.stack_structure.faults_input_data = [_finite_fault_data(inputs, descriptor), None, None]
    options.cache_mode = InterpolationOptions.CacheMode.IN_MEMORY_CACHE
    options.evaluation_options.compute_scalar_gradient = True
    options.evaluation_options.number_octree_levels_surface = surface_levels
    options.evaluation_options.mesh_extraction_overlap = 'none'
    if relation is not None:
        descriptor.stack_structure.faults_relations = relation
    before = copy.deepcopy(inputs)
    prepared = {}

    def extraction(data_descriptor, interpolation_input, options, octree_list):
        prepared['drift_sampler'] = sampler_api.prepare_fault_drift_sampler(
            data_descriptor, interpolation_input, options, octree_list)
        prepared['levels'] = copy.deepcopy(octree_list)
        prepared['input_grid_size'] = interpolation_input.grid.len_all_grids
        if inside is not None:
            inside(data_descriptor, interpolation_input, options, octree_list)
        return []

    with monkeypatch.context() as patch:
        patch.setattr(api, 'dual_contouring_multi_scalar', extraction)
        solution = api.compute_model(inputs, options, descriptor)
    np.testing.assert_array_equal(inputs.surface_points.sp_coords, before.surface_points.sp_coords)
    np.testing.assert_array_equal(inputs.orientations.dip_positions, before.orientations.dip_positions)
    np.testing.assert_array_equal(inputs.orientations.dip_gradients, before.orientations.dip_gradients)
    return prepared, solution


def test_cache_mismatch_still_uses_production_weights(monkeypatch, numpy_backend):
    def inside(descriptor, inputs, options, levels):
        changed = copy.deepcopy(inputs)
        changed.surface_points.sp_coords[3, 0] += .01
        sampler = sampler_api.prepare_fault_drift_sampler(descriptor, changed, options, levels)
        assert None in sampler['diagnostics']['cache_fingerprints']
    _prepare(monkeypatch, inside=inside)


@pytest.mark.parametrize('mutation', ['replacement', 'inplace'])
def test_same_fingerprint_corrupt_cache_payload_rejected(monkeypatch, numpy_backend, mutation):
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache

    def inside(descriptor, inputs, options, levels):
        cached = WeightCache.load_weights(f'{options.cache_model_name}.1', False)
        fingerprint = cached['hash']
        original = cached['weights'].copy()
        output = levels[-1].outputs[1]
        actual = output.weights.copy()
        scalar = output.exported_fields._scalar_field.copy()
        assert not output.weights.flags.writeable
        assert not np.shares_memory(output.weights, cached['weights'])
        sampler = sampler_api.prepare_fault_drift_sampler(descriptor, inputs, options, levels)
        points = output.grid.values[:5]
        expected = sampler['query'](points)
        if mutation == 'replacement':
            cached['weights'] = original + 1
        else:
            cached['weights'] += 1
        assert cached['hash'] == fingerprint
        np.testing.assert_array_equal(output.weights, actual)
        np.testing.assert_array_equal(output.exported_fields._scalar_field, scalar)

        def forbidden(*args, **kwargs):
            raise AssertionError('corrupt cache must be rejected without evaluating or solving')

        scalar_api = importlib.import_module('gempy_engine.API.interp_single._interp_scalar_field')
        try:
            with monkeypatch.context() as patch:
                patch.setattr(sampler_api, '_evaluate_sys_eq', forbidden)
                patch.setattr(scalar_api, '_solve_interpolation_result', forbidden)
                with pytest.raises(ValueError, match='cached weights differ from actual production solve for stack 1'):
                    sampler_api.prepare_fault_drift_sampler(descriptor, inputs, options, levels)
            for left, right in zip(sampler['query'](points), expected):
                np.testing.assert_array_equal(left, right)
        finally:
            cached['weights'] = original
    _prepare(monkeypatch, inside=inside)


@pytest.mark.parametrize('case', ['missing', 'cache_alias'])
def test_missing_or_aliased_production_weight_provenance_rejected(monkeypatch, numpy_backend, case):
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache

    def inside(descriptor, inputs, options, levels):
        changed = copy.deepcopy(levels)
        cached = WeightCache.load_weights(f'{options.cache_model_name}.1', False)
        changed[-1].outputs[1].scalar_fields.weights = None if case == 'missing' else cached['weights']
        with pytest.raises(ValueError, match='production weight provenance.*stack 1'):
            sampler_api.prepare_fault_drift_sampler(descriptor, inputs, options, changed)
    _prepare(monkeypatch, inside=inside)


def test_retained_torch_weights_preserve_autograd_and_do_not_alias(monkeypatch, numpy_backend):
    torch = pytest.importorskip('torch')
    BT._change_backend(engine_backend=AvailableBackends.PYTORCH, use_gpu=False,
                       use_pykeops=False, dtype='float64', grads=True)
    feature = importlib.import_module('gempy_engine.API.interp_single._interp_single_feature')
    weights = torch.tensor([1., 2.], dtype=torch.float64, requires_grad=True)
    fields = SimpleNamespace(set_structure_values=lambda **kwargs: None)
    output = SimpleNamespace(weights=None)
    inputs = SimpleNamespace(grid=object(), slice_feature=slice(None), macro_reference_size=0)
    inputs.grid = SimpleNamespace(len_all_grids=0)
    shape = SimpleNamespace(number_of_points_per_surface=[], reference_sp_position=[])
    solver = SimpleNamespace(xyz_to_interpolate=torch.empty((0, 3)), debug={})
    options = InterpolationOptions.init_octree_options(refinement=2)
    monkeypatch.setattr(feature, 'compute_weights', lambda *args: weights)
    monkeypatch.setattr(feature, '_evaluate_sys_eq', lambda *args, **kwargs: fields)
    monkeypatch.setattr(feature, 'micro_evaluation_options', lambda options, inputs: options)
    monkeypatch.setattr(feature, 'fit_micro_fields', lambda *args: None)
    monkeypatch.setattr(feature, '_segment', lambda *args: output)
    result = feature.interpolate_feature_with_cokrig(inputs, options, shape, solver)
    assert result.weights.data_ptr() != weights.data_ptr()
    assert result.weights.requires_grad
    result.weights.sum().backward()
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))
    with torch.no_grad():
        weights.add_(1)
    torch.testing.assert_close(result.weights, torch.tensor([1., 2.], dtype=torch.float64))


def test_torch_autograd_fault_backend_rejected_explicitly(numpy_backend):
    pytest.importorskip('torch')
    inputs, options, descriptor = model7_combination_factory()
    BT._change_backend(engine_backend=AvailableBackends.PYTORCH, use_gpu=False,
                       use_pykeops=False, dtype='float64', grads=True)
    try:
        with pytest.raises(NotImplementedError, match='unsupported_fault_backend'):
            sampler_api.prepare_fault_drift_sampler(descriptor, inputs, options, [])
    finally:
        BT._change_backend(engine_backend=AvailableBackends.numpy, use_gpu=False,
                           use_pykeops=False, dtype='float64', grads=False)


def test_production_drift_sampler_reproduces_production_fields(monkeypatch, numpy_backend):
    prepared, _ = _prepare(monkeypatch)
    sampler = prepared['drift_sampler']
    assert sampler['fault_stacks'] == (0,)
    assert sampler['affected_by'] == {0: (), 1: (0,), 2: (0,)}
    assert max(sampler['diagnostics']['drift_rebuild_max_error']) <= 1e-10

    def forbidden(*args, **kwargs):
        raise AssertionError('query must not solve or access the cache')

    scalar_api = importlib.import_module('gempy_engine.API.interp_single._interp_scalar_field')
    cache_api = importlib.import_module('gempy_engine.modules.weights_cache.weights_cache_interface')
    monkeypatch.setattr(snapshot_api, 'resolve_weight_cache', forbidden)
    monkeypatch.setattr(scalar_api, 'compute_weights', forbidden)
    monkeypatch.setattr(cache_api.WeightCache, 'load_weights', forbidden)
    # Every grid point of every level, including the mixed-drift ones next to the fault.
    for level in prepared['levels']:
        points = level.grid.values
        values, gradients = sampler['query'](points)
        assert values.shape == (3, len(points)) and gradients.shape == (3, len(points), 3)
        for index, output in enumerate(level.outputs):
            fields = output.exported_fields
            np.testing.assert_allclose(values[index], fields._scalar_field[:len(points)], rtol=1e-12, atol=1e-12)
            for axis, field in enumerate((fields._gx_field, fields._gy_field, fields._gz_field)):
                np.testing.assert_allclose(gradients[index, :, axis], field[:len(points)], rtol=1e-12, atol=1e-12)
    scalars, no_gradients = sampler['query'](points, stacks=(2, 1), gradients=False)
    assert no_gradients is None
    np.testing.assert_allclose(scalars, values[[2, 1]], rtol=1e-12, atol=1e-12)
    subset, subset_grad = sampler['query'](points, stacks=(2, 0))
    np.testing.assert_array_equal(subset, values[[2, 0]])
    np.testing.assert_array_equal(subset_grad, gradients[[2, 0]])
    for invalid in ((), (3,), (True,)):
        with pytest.raises(ValueError, match='invalid_query_stacks'):
            sampler['query'](points, stacks=invalid)
    with pytest.raises(ValueError, match='invalid_query_points'):
        sampler['query'](np.zeros(3))


def _finite_fault_data(inputs, descriptor, radius=.06):
    """Model7's vertical fault with a slip patch around its surface points, tips inside the domain."""
    from gempy_engine.core.data import FiniteFault
    from gempy_engine.core.data.kernel_classes.faults import FaultsData

    n_fault = descriptor.tensors_structure.number_of_points_per_surface[0]
    center = np.asarray(inputs.surface_points.sp_coords)[:n_fault].mean(axis=0)
    return FaultsData.from_user_input(None, FiniteFault(center=tuple(center), strike_radius=radius,
                                                        dip_radius=radius))


def test_finite_fault_drift_sampler_reproduces_production_fields(monkeypatch, numpy_backend):
    prepared, _ = _prepare(monkeypatch, finite=True)
    sampler = prepared['drift_sampler']
    diagnostics = sampler['diagnostics']
    assert diagnostics['finite_fault_stacks'] == (0,)
    assert max(diagnostics['drift_rebuild_max_error']) <= 1e-10
    monkeypatch.setattr(importlib.import_module('gempy_engine.API.interp_single._interp_scalar_field'),
                        'compute_weights', Mock(side_effect=AssertionError('query must not solve')))
    for level in prepared['levels']:
        points = level.grid.values
        values, gradients = sampler['query'](points)
        batched, = sampler['query_batch']([(points, (0, 1, 2), 'scalar')])
        np.testing.assert_allclose(batched, values, rtol=1e-12, atol=1e-12)
        for index, output in enumerate(level.outputs):
            fields = output.exported_fields
            np.testing.assert_allclose(values[index], fields._scalar_field[:len(points)], rtol=1e-12, atol=1e-12)
            for axis, field in enumerate((fields._gx_field, fields._gy_field, fields._gz_field)):
                np.testing.assert_allclose(gradients[index, :, axis], field[:len(points)], rtol=1e-12, atol=1e-12)


def test_finite_fault_side_queries_follow_the_taper(monkeypatch, numpy_backend):
    """Fixing the side offsets affected stacks only inside the slip patch."""
    prepared, _ = _prepare(monkeypatch, finite=True)
    sampler = prepared['drift_sampler']
    center = np.asarray(_finite_fault_data(*model7_combination_factory()[::2]).finite_fault.center)
    # Off the fault along its normal (x), then along strike (y) to inside and beyond the tip.
    for along_strike, offset in ((0., True), (.03, True), (.09, False)):
        points = center + np.array([[dx, along_strike, 0.] for dx in (-.05, -.02, .02, .05)])
        sides = np.ones((1, len(points)))
        positive = sampler['query_sides'](points, (1, 2), sides)
        negative = sampler['query_sides'](points, (1, 2), -sides)
        assert np.any(np.abs(positive - negative) > 1e-6) == offset, along_strike


def test_fault_banks_still_reject_finite_faults(monkeypatch, numpy_backend):
    from gempy_engine.API.dual_contouring.fault_bank_sampler import prepare_fault_bank_sampler

    def inside(*args):
        with pytest.raises(ValueError, match='unsupported_finite_fault'):
            prepare_fault_bank_sampler(*args)

    _prepare(monkeypatch, finite=True, inside=inside)


def _prepared_drift_sampler():
    import gempy_engine.API.model.model_api as api
    from tests.fixtures.model7_combination import model7_combination_factory

    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=True)
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.mesh_extraction_overlap = 'none'
    prepared = {}

    def capture(data_descriptor=None, interpolation_input=None, options=None, octree_list=None):
        prepared['sampler'] = sampler_api.prepare_fault_drift_sampler(data_descriptor, interpolation_input, options, octree_list)
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
            BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
            torch.set_default_device('cpu')
