"""Fixed-weight bank sampler (one verified infinite planar fault), without meshing."""

import copy
import importlib
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gempy_engine.API.dual_contouring import fault_bank_sampler as sampler_api
from gempy_engine.API.dual_contouring import fault_drift_sampler as drift_api
from gempy_engine.API.dual_contouring import fixed_weight_snapshots as snapshot_api
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.internal_structs import EvaluatorInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
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


# region Reference: the unmodified production solve, captured per stack

def _capture_model7(interpolation_input, options, descriptor):
    """Run the unmodified solve, retaining root-level custom-grid metadata."""
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.API.interp_single import _interp_single_feature as feature

    if BT.engine_backend is not AvailableBackends.numpy or BT.use_gpu or \
            BT.use_pykeops or np.dtype(BT.dtype) != np.dtype('float64'):
        raise ValueError('unsupported_backend: NumPy CPU float64 without KeOps required')
    relations = descriptor.stack_structure
    if tuple(relations.masking_descriptor) != (R.FAULT, R.ERODE, R.BASEMENT) or not np.array_equal(
            relations.faults_relations, [[False, True, True], [False, False, False], [False, False, False]]):
        raise ValueError('unsupported_model: one infinite fault affecting both Model7 stacks required')
    if interpolation_input.fault_values is not None and interpolation_input.fault_values.finite_fault_defined:
        raise ValueError('unsupported_finite_fault')
    if relations.faults_input_data is not None and any(
            f is not None and f.finite_fault_defined for f in relations.faults_input_data):
        raise ValueError('unsupported_finite_fault')
    captures = []
    evaluate = feature._evaluate_sys_eq

    def record(eval_input, weights, evaluation_options, grid=None):
        captures.append((copy.deepcopy(eval_input), np.array(weights, copy=True),
                         evaluation_options.model_copy(deep=True)))
        return evaluate(eval_input, weights, evaluation_options, grid=grid)

    with patch.object(feature, '_evaluate_sys_eq', record):
        solution = compute_model(interpolation_input, options, descriptor)
    if len(captures) < 3:
        raise ValueError('unsupported_evaluator: no three-stack cokriging capture')
    captures = captures[:3]
    if [c[0].fault_internal.n_faults for c in captures] != [0, 1, 1]:
        raise ValueError('unsupported_fault_count: expected one independent and two dependent fields')
    for eval_input, _, _ in captures[1:]:
        faults = eval_input.fault_internal
        values = faults.fault_values_everywhere
        if faults.finite_fault_defined or not np.isfinite(values).all() or \
                not np.isclose(values.min(), 0, atol=1e-12) or not np.isclose(values.max(), 1, atol=1e-12):
            raise ValueError('unverified_fault_banks: shifted production drift must span 0 and 1')
    fault_output = solution.root_output.outputs[0]
    fault_scalar = np.asarray(fault_output.exported_fields.scalar_field)
    coordinates = captures[0][0].xyz_to_interpolate[:len(fault_scalar)]
    design = np.column_stack((coordinates, np.ones(len(coordinates))))
    fit = np.linalg.lstsq(design, fault_scalar, rcond=None)[0]
    if np.max(np.abs(design @ fit-fault_scalar)) > 1e-10:
        raise ValueError('unsupported_nonplanar_fault: bounded infinite planar separator required')
    separator = fault_scalar-fault_output.scalar_field_at_sp[0]
    away = np.abs(separator) > 1e-10
    for eval_input, _, _ in captures[1:]:
        drift = eval_input.fault_internal.fault_values_everywhere[0, :len(separator)]
        if not np.array_equal(drift[away] > .5, separator[away] < 0):
            raise ValueError('unsupported_bank_orientation: Model7 drift 0 must be positive scalar bank')
    return solution, captures


def _query_bank(capture, xyz, bank=None):
    """Change ONLY evaluation coordinates and query drift, preserving solved data.

    Reference/rest/SP fault values remain bitwise copies of the original solve.
    Actual engine kernel gradients are evaluated with constant bank drift.
    Coordinates are not displaced to either side of the geometric separator.
    """
    from gempy_engine.API.interp_single._interp_scalar_field import _evaluate_sys_eq

    original, weights, options = capture
    xyz = np.array(xyz, dtype=np.float64, copy=True)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('invalid_query_points')
    view = copy.copy(original)
    view.xyz_to_interpolate = xyz
    if original.fault_internal.n_faults:
        if bank not in (0, 1):
            raise ValueError('invalid_bank: dependent queries require explicit 0 or 1')
        if isinstance(view, EvaluatorInput):
            view.solver_input = copy.copy(original.solver_input)
            view.solver_input.fault_internal = copy.deepcopy(original.fault_internal)
        else:
            view.fault_internal = copy.deepcopy(original.fault_internal)
        view.fault_internal.fault_values_everywhere = np.full((1, len(xyz)), bank, dtype=np.float64)
    elif bank is not None:
        raise ValueError('invalid_bank: independent fault field has no drift')
    exported = _evaluate_sys_eq(view, weights, options)
    scalar = np.array(exported._scalar_field, dtype=np.float64, copy=True)
    gradient = np.stack([np.array(g, dtype=np.float64, copy=True) for g in
                         (exported._gx_field, exported._gy_field, exported._gz_field)], axis=-1)
    if not np.isfinite(scalar).all() or not np.isfinite(gradient).all():
        raise ValueError('nonfinite_bank_query')
    return scalar, gradient

# endregion


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
        prepared['drift_sampler'] = drift_api.prepare_fault_drift_sampler(
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
    _, captures = _capture_model7(baseline_inputs, baseline_options, baseline_descriptor)
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
    monkeypatch.setattr(snapshot_api, 'resolve_weight_cache', forbidden)
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
            expected, expected_grad = _query_bank(capture, points, None if index == 0 else bank)
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
    from gempy_engine.API.dual_contouring.joint_fault_banks import _BankFieldMemo
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

    monkeypatch.setattr(snapshot_api, 'input_preprocess', forbidden)
    monkeypatch.setattr(snapshot_api, 'resolve_weight_cache', forbidden)
    with pytest.raises(ValueError, match='invalid_fault_metadata'):
        sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, [])
    assert stacks.faults_input_data is original
    assert inputs._fault_values is (data if case == 'input_duck_type' else None)


def test_torch_autograd_fault_backend_rejected_explicitly(numpy_backend):
    pytest.importorskip('torch')
    inputs, options, descriptor = model7_combination_factory()
    BT._change_backend(engine_backend=AvailableBackends.PYTORCH, use_gpu=False,
                       use_pykeops=False, dtype='float64', grads=True)
    try:
        with pytest.raises(NotImplementedError, match='unsupported_fault_backend'):
            sampler_api.prepare_fault_bank_sampler(descriptor, inputs, options, [])
    finally:
        BT._change_backend(engine_backend=AvailableBackends.numpy, use_gpu=False,
                           use_pykeops=False, dtype='float64', grads=False)
