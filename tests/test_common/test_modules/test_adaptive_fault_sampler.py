"""Production-cache preparation and fixed-solve bank queries, without meshing."""

import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from gempy_engine.API.dual_contouring import adaptive_fault_sampler as sampler_api
from gempy_engine.API.dual_contouring.model7_bank_experiment import capture_model7, query_bank
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from tests.fixtures.model7_combination import model7_combination_factory


@pytest.fixture
def numpy_backend():
    BT._change_backend(engine_backend=AvailableBackends.numpy, use_gpu=False,
                       use_pykeops=False, dtype='float64', grads=False)


def _prepare(monkeypatch, *, levels=2, surface_levels=2, relation=None, inside=None):
    api = importlib.import_module('gempy_engine.API.model.model_api')
    inputs, options, descriptor = model7_combination_factory(
        number_octree_levels=levels, mesh_extraction=True,
        custom_grid=np.array([[x, 500, z] for x in (100, 600, 1200, 1800, 2400)
                              for z in (50, 300, 600, 950)]))
    options.cache_mode = InterpolationOptions.CacheMode.IN_MEMORY_CACHE
    options.evaluation_options.compute_scalar_gradient = True
    options.evaluation_options.number_octree_levels_surface = surface_levels
    options.evaluation_options.mesh_extraction_overlap = 'none'
    if relation is not None:
        descriptor.stack_structure.faults_relations = relation
    before = copy.deepcopy(inputs)
    prepared = {}

    def extraction(data_descriptor, interpolation_input, options, octree_list):
        prepared['sampler'] = sampler_api.prepare_fault_bank_sampler(
            data_descriptor, interpolation_input, options, octree_list)
        prepared['root_sampler'] = sampler_api.prepare_fault_bank_sampler(
            data_descriptor, interpolation_input, options, octree_list[:1])
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


def test_production_parity_no_solve_no_cache_no_snapshot_mutation(monkeypatch, numpy_backend):
    prepared, _ = _prepare(monkeypatch)
    sampler = prepared['root_sampler']
    root = prepared['levels'][0]
    points = root.grid.values
    baseline_inputs, baseline_options, baseline_descriptor = model7_combination_factory(number_octree_levels=2)
    baseline_inputs.grid.custom_grid = copy.deepcopy(root.grid.custom_grid)
    baseline_options.evaluation_options.compute_scalar_gradient = True
    _, captures = capture_model7(baseline_inputs, baseline_options, baseline_descriptor)
    before = copy.deepcopy(sampler['snapshots'])
    for snapshot, capture in zip(sampler['snapshots'], captures):
        np.testing.assert_array_equal(snapshot[1], capture[1])
        for name in ('fault_values_everywhere', 'fault_values_on_sp', 'fault_values_ref', 'fault_values_rest'):
            np.testing.assert_array_equal(getattr(snapshot[0].fault_internal, name),
                                          getattr(capture[0].fault_internal, name))

    def forbidden(*args, **kwargs):
        raise AssertionError('query must not solve, access cache, or orchestrate interpolation')

    scalar_api = importlib.import_module('gempy_engine.API.interp_single._interp_scalar_field')
    feature_api = importlib.import_module('gempy_engine.API.interp_single.interp_features')
    cache_api = importlib.import_module('gempy_engine.modules.weights_cache.weights_cache_interface')
    monkeypatch.setattr(sampler_api, 'resolve_weight_cache', forbidden)
    monkeypatch.setattr(scalar_api, 'compute_weights', forbidden)
    monkeypatch.setattr(scalar_api, '_solve_interpolation_result', forbidden)
    monkeypatch.setattr(feature_api, 'interpolate_all_fields_no_octree', forbidden)
    monkeypatch.setattr(cache_api.WeightCache, 'load_weights', forbidden)
    banks = []
    for bank in (0, 1):
        values, gradients = sampler['query'](points, bank)
        assert values.shape == (3, len(points))
        assert gradients.shape == (3, len(points), 3)
        banks.append((values, gradients))
        for index, capture in enumerate(captures):
            expected, expected_grad = query_bank(capture, points, None if index == 0 else bank)
            np.testing.assert_allclose(values[index], expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(gradients[index], expected_grad, rtol=1e-12, atol=1e-12)
            drift = sampler['snapshots'][index][0].fault_internal
            if index:
                saturated = drift.fault_values_everywhere[0, :len(points)] == bank
                assert saturated.any()
                fields = root.outputs[index].exported_fields
                np.testing.assert_allclose(values[index, saturated], fields._scalar_field[:len(points)][saturated],
                                           rtol=1e-12, atol=1e-12)
                for axis, field in enumerate((fields._gx_field, fields._gy_field, fields._gz_field)):
                    np.testing.assert_allclose(gradients[index, saturated, axis], field[:len(points)][saturated],
                                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(banks[0][0][0], banks[1][0][0])
    assert np.max(np.abs(banks[0][0][1:] - banks[1][0][1:])) > .01
    for current, previous in zip(sampler['snapshots'], before):
        original, weights, _ = current
        np.testing.assert_array_equal(weights, previous[1])
        assert not weights.flags.writeable
        np.testing.assert_array_equal(original.xyz_to_interpolate, previous[0].xyz_to_interpolate)
        for name in ('fault_values_everywhere', 'fault_values_on_sp', 'fault_values_ref', 'fault_values_rest'):
            np.testing.assert_array_equal(getattr(original.fault_internal, name),
                                          getattr(previous[0].fault_internal, name))
    empty, empty_grad = sampler['query'](np.empty((0, 3)), 0)
    assert empty.shape == (3, 0) and empty_grad.shape == (3, 0, 3)
    # Even assigning new attributes on inspection copies cannot change queries.
    sampler['snapshots'][1][0].fault_internal.fault_values_ref = np.zeros((1, 1))
    actual, _ = sampler['query'](points, 0)
    np.testing.assert_array_equal(actual, banks[0][0])
    subset, _ = sampler['query'](points[:3], 0)
    np.testing.assert_allclose(subset, banks[0][0][:, :3], rtol=1e-12, atol=1e-12)


def test_directed_subset_and_last_surface_grid(monkeypatch, numpy_backend):
    matrix = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=bool)
    prepared, solution = _prepare(monkeypatch, levels=3, surface_levels=2, relation=matrix)
    sampler = prepared['sampler']
    assert sampler['fault_stack'] == 0
    assert sampler['affected_stacks'] == (2,)
    assert sampler['unaffected_stacks'] == (1,)
    last = prepared['levels'][-1]
    expected_xyz = last.outputs[0].grid.values
    assert sampler['diagnostics']['reference_grid_size'] == len(expected_xyz)
    for original, _, _ in sampler['snapshots']:
        np.testing.assert_array_equal(original.xyz_to_interpolate[:len(expected_xyz)], expected_xyz)
    assert len(solution.octrees_output) == 3 and len(prepared['levels']) == 2
    points = last.grid.values[::19]
    zero, zero_grad = sampler['query'](points, 0)
    one, one_grad = sampler['query'](points, 1)
    np.testing.assert_array_equal(zero[:2], one[:2])
    np.testing.assert_array_equal(zero_grad[:2], one_grad[:2])
    assert np.max(np.abs(zero[2] - one[2])) > .01
    for invalid in (None, -1, 2, [0]):
        with pytest.raises(ValueError, match='invalid_bank'):
            sampler['query'](points, invalid)
    for invalid in (np.zeros(3), np.zeros((3, 2)), [[np.nan, 0, 0]]):
        with pytest.raises(ValueError, match='invalid_query_points'):
            sampler['query'](invalid, 0)
    # Stack subsets keep row order; bank-independent stacks need no bank.
    independent, independent_grad = sampler['query'](points, None, stacks=(1, 0))
    np.testing.assert_allclose(independent, zero[[1, 0]], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(independent_grad, zero_grad[[1, 0]], rtol=1e-12, atol=1e-12)
    affected, _ = sampler['query'](points, 1, stacks=(2,))
    np.testing.assert_allclose(affected, one[2:], rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match='invalid_bank'):
        sampler['query'](points, None, stacks=(2,))
    for invalid in ((), (3,), (-1,), (True,)):
        with pytest.raises(ValueError, match='invalid_query_stacks'):
            sampler['query'](points, 0, stacks=invalid)


def test_bank_field_memo_evaluates_each_point_once_and_shares_independent_stacks():
    from gempy_engine.API.dual_contouring.adaptive_fault_extraction import _BankFieldMemo
    calls = []

    def query(points, bank=None, stacks=None):
        calls.append((len(points), bank, stacks))
        (stack,) = stacks
        offset = 0 if bank is None else 10 * (bank + 1)
        raw = points.sum(axis=1) + 100 * stack + offset
        return raw[None], np.repeat(points[None] + offset, 1, axis=0)

    memo = _BankFieldMemo(query, affected=(1,))
    points = np.array([[0., 0, 0], [1, 0, 0], [0., 0, 0], [0, 2, 0]])
    raw, gradients = memo(points, [0, 1], 0)
    assert calls == [(3, None, (0,)), (3, 0, (1,))]
    np.testing.assert_array_equal(raw, [[0, 1, 0, 2], [110, 111, 110, 112]])
    np.testing.assert_array_equal(gradients[1], points + 10)
    calls.clear()
    raw, _ = memo(points[::-1], [0, 1], 1)
    # Fault stack 0 is cached across banks; affected stack 1 is bank-specific.
    assert calls == [(3, 1, (1,))]
    np.testing.assert_array_equal(raw, [[2, 0, 1, 0], [122, 120, 121, 120]])
    calls.clear()
    memo(np.array([[0., 0, 0], [5, 0, 0]]), [1], 1)
    assert calls == [(1, 1, (1,))]


def test_cache_mismatch_rejected_before_forcing_bank(monkeypatch, numpy_backend):
    def inside(descriptor, inputs, options, levels):
        changed = copy.deepcopy(inputs)
        changed.surface_points.sp_coords[3, 0] += .01
        with pytest.raises(ValueError, match='fingerprint mismatch.*no solve'):
            sampler_api.prepare_fault_bank_sampler(descriptor, changed, options, levels)
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
        sampler = sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, levels)
        points = output.grid.values[:5]
        expected = sampler['query'](points, 0)
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
                    sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, levels)
            for left, right in zip(sampler['query'](points, 0), expected):
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
            sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, changed)
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


def test_segmentation_shift_uses_production_reference_minimum(monkeypatch, numpy_backend):
    def inside(descriptor, inputs, options, levels):
        baseline = sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, levels)
        shifted = copy.deepcopy(levels)
        # An exactly representable constant must cancel before slicing SP/ref/rest.
        for level in shifted:
            level.outputs[0].scalar_fields._values_block += 8
        actual = sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, shifted)
        for left, right in zip(baseline['snapshots'], actual['snapshots']):
            for name in ('fault_values_on_sp', 'fault_values_ref', 'fault_values_rest'):
                np.testing.assert_array_equal(getattr(left[0].fault_internal, name),
                                              getattr(right[0].fault_internal, name))
    _prepare(monkeypatch, inside=inside)


@pytest.mark.parametrize('case', ['nonplanar_scalar', 'varying_gradients', 'reversed_banks'])
def test_actual_field_and_bank_validation(monkeypatch, numpy_backend, case):
    def inside(descriptor, inputs, options, levels):
        changed = copy.deepcopy(levels)
        if case == 'reversed_banks':
            for level in changed:
                fields = level.outputs[0].exported_fields
                fields._scalar_field = -fields._scalar_field
            with pytest.raises(ValueError, match='unverified_fault_banks'):
                sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, changed)
            return
        evaluate = sampler_api._evaluate_sys_eq

        def altered(view, weights, evaluation_options):
            fields = evaluate(view, weights, evaluation_options)
            if view.fault_internal.n_faults == 0:
                if case == 'nonplanar_scalar':
                    fields._scalar_field += view.xyz_to_interpolate[:, 0] ** 2
                else:
                    fields._gx_field += view.xyz_to_interpolate[:, 0]
            return fields

        with monkeypatch.context() as patch:
            patch.setattr(sampler_api, '_evaluate_sys_eq', altered)
            with pytest.raises(ValueError, match='unsupported_nonplanar_fault'):
                sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, changed)
    _prepare(monkeypatch, inside=inside)


def test_no_affected_stacks_allow_unbanked_query(monkeypatch, numpy_backend):
    prepared, _ = _prepare(monkeypatch, relation=np.zeros((3, 3), dtype=bool))
    sampler = prepared['sampler']
    assert sampler['affected_stacks'] == () and sampler['unaffected_stacks'] == (1, 2)
    points = prepared['levels'][0].grid.values[:4]
    unbanked = sampler['query'](points)
    for bank in (0, 1):
        actual = sampler['query'](points, bank)
        for left, right in zip(unbanked, actual):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('case, message', [
    ('multiple', 'unsupported_fault_count'), ('finite', 'unsupported_finite_fault'),
    ('dependent_fault', 'unsupported_fault_relations'), ('other_edge', 'unsupported_fault_relations'),
    ('external', 'bank-aware contract'), ('micro', 'unsupported_fault_micro'),
])
def test_unsupported_contracts_fail_honestly(numpy_backend, case, message):
    inputs, options, descriptor = model7_combination_factory()
    stacks = descriptor.stack_structure
    if case == 'multiple':
        stacks.masking_descriptor[1] = R.FAULT
    elif case == 'finite':
        stacks.faults_input_data = [FaultsData(finite_fault=object()), None, None]
    elif case == 'dependent_fault':
        stacks.faults_relations[1, 0] = True
    elif case == 'other_edge':
        stacks.faults_relations[1, 2] = True
    elif case == 'external':
        stacks.interp_functions_per_stack = [None, object(), None]
    elif case == 'micro':
        inputs.micro_points = object()
    with pytest.raises((ValueError, NotImplementedError), match=message):
        sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, [])


@pytest.mark.parametrize('case', ['duck_type', 'populated_duck_type', 'empty_list', 'short_list',
                                  'long_list', 'not_a_list', 'input_duck_type'])
def test_fault_metadata_types_and_lengths_rejected_before_preprocessing(monkeypatch, numpy_backend, case):
    inputs, options, descriptor = model7_combination_factory()
    stacks = descriptor.stack_structure
    data = SimpleNamespace(finite_fault_defined=False, n_faults=0)
    if case == 'populated_duck_type':
        data.fault_values_everywhere = np.ones((1, 4))
    if case in ('duck_type', 'populated_duck_type'):
        stacks.faults_input_data = [data, None, None]
    elif case == 'input_duck_type':
        inputs._fault_values = data
    elif case == 'not_a_list':
        stacks.faults_input_data = {0: None, 1: None, 2: None}
    else:
        length = {'empty_list': 0, 'short_list': 2, 'long_list': 4}[case]
        stacks.faults_input_data = [None] * length
    original = stacks.faults_input_data

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid metadata must be rejected before preprocessing/cache access')

    monkeypatch.setattr(sampler_api, 'input_preprocess', forbidden)
    monkeypatch.setattr(sampler_api, 'resolve_weight_cache', forbidden)
    with pytest.raises(ValueError, match='invalid_fault_metadata'):
        sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, [])
    assert stacks.faults_input_data is original
    assert inputs._fault_values is (data if case == 'input_duck_type' else None)


def test_torch_fault_backend_rejected_explicitly(numpy_backend):
    pytest.importorskip('torch')
    inputs, options, descriptor = model7_combination_factory()
    BT._change_backend(engine_backend=AvailableBackends.PYTORCH, use_gpu=False,
                       use_pykeops=False, dtype='float64', grads=False)
    with pytest.raises(NotImplementedError, match='unsupported_fault_backend'):
        sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, [])


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
    monkeypatch.setattr(sampler_api, 'resolve_weight_cache', forbidden)
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
