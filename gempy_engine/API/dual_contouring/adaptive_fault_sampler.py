"""Detached, fixed-solve queries for one verified infinite planar fault.

Prepare during public compute_model's extraction hook, while its cache is live.
Values are raw stack potentials, not surface-expanded fields or lithology IDs.
Bank 0 is the positive fault-potential side; bank 1 is the negative side.
"""

import copy
from dataclasses import replace
from itertools import product

import numpy as np

from ..interp_single._aux_faults_ops import _grab_stack_fault_data, _modify_faults_values_output
from ..interp_single._interp_single_feature import input_preprocess
from ..interp_single._interp_scalar_field import _evaluate_sys_eq
from ...modules.evaluator.symbolic_evaluator import symbolic_evaluator_optimized_stacked
from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor as BT
from ...core.data import TensorsStructure
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.stack_relation_type import StackRelationType
from ...modules.activator._soft_segment import soft_segment_unbounded
from ...modules.solver.interpolation_solver import pykeops_solver_requested
from ...modules.weights_cache.weight_cache_policy import resolve_weight_cache, WeightCacheRoute


def _freeze_arrays(value):
    """Freeze arrays in an already private object graph."""
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, (list, tuple)):
        for item in value:
            _freeze_arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            _freeze_arrays(item)
    elif hasattr(value, '__dict__') and not isinstance(value, type):
        for item in vars(value).values():
            _freeze_arrays(item)


def _host(values):
    """NumPy view/copy of a NumPy array or a (possibly CUDA) Torch tensor."""
    return BT.t.to_numpy(values) if hasattr(values, 'detach') else np.asarray(values)


def _check_drift_backend():
    """NumPy, or Torch on CPU or CUDA, optionally with KeOps; float64, no autograd."""
    if (BT.engine_backend not in (AvailableBackends.numpy, AvailableBackends.PYTORCH) or BT.COMPUTE_GRADS
            or np.dtype(str(BT.dtype)) != np.dtype('float64')):
        raise NotImplementedError('unsupported_fault_backend: float64 NumPy or Torch without autograd required')


def _fault_matrix(stacks):
    n_stacks = stacks.n_stacks
    matrix = (np.zeros((n_stacks, n_stacks), dtype=bool) if stacks.faults_relations is None
              else np.asarray(stacks.faults_relations))
    if matrix.shape != (n_stacks, n_stacks) or not np.isin(matrix, (False, True)).all():
        raise ValueError('unsupported_fault_relations: directed boolean matrix required')
    return matrix.astype(bool)


def _check_fixed_query_contract(stacks, interpolation_input, octree_list):
    """Shared fixed-weight query preconditions; returns per-stack fault inputs."""
    n_stacks = stacks.n_stacks
    if stacks.faults_input_data is not None and (
            not isinstance(stacks.faults_input_data, (list, tuple)) or
            len(stacks.faults_input_data) != n_stacks):
        raise ValueError('invalid_fault_metadata: faults_input_data must contain one entry per stack')
    fault_inputs = ([None] * n_stacks if stacks.faults_input_data is None
                    else list(stacks.faults_input_data))
    for data in fault_inputs + [interpolation_input._fault_values]:
        if data is not None and not isinstance(data, FaultsData):
            raise ValueError('invalid_fault_metadata: entries must be FaultsData or None')
        if data is not None and data.finite_fault_defined:
            raise ValueError('unsupported_finite_fault')
    if interpolation_input._fault_values is not None:
        raise ValueError('unsupported_external_fault_data')
    if interpolation_input.micro_points is not None or interpolation_input.evaluation_micro_points is not None:
        raise NotImplementedError('unsupported_fault_micro_correction: fixed reference state required')
    if stacks.interp_functions_per_stack is not None and any(
            callback is not None for callback in stacks.interp_functions_per_stack):
        raise NotImplementedError('unsupported_external_fault_callback: explicit bank-aware contract required')
    if StackRelationType.NULL_SPACE in stacks.masking_descriptor:
        raise NotImplementedError('unsupported_fault_null_space')
    if stacks.ignored_grid_types_per_stack is not None and any(stacks.ignored_grid_types_per_stack):
        raise NotImplementedError('unsupported_fault_stack_grid_subset')
    if not octree_list or any(len(level.outputs) != n_stacks for level in octree_list):
        raise ValueError('invalid_fault_octree_outputs')
    if any(level.grid.octree_grid is None for level in octree_list):
        raise ValueError('invalid_fault_octree_grid')
    return fault_inputs


def _fixed_weight_snapshots(descriptor, interpolation_input, options, octree_list, fault_stacks, fault_inputs):
    """Frozen (solver_input, weights, options) per stack, validated against the live cache.

    Production fault drift for every stack in ``fault_stacks`` is rebuilt on the
    last surface level's reference grid, exactly as the production solve saw it.
    """
    stacks = descriptor.stack_structure
    n_stacks = stacks.n_stacks
    last = octree_list[-1]
    private_stacks = copy.deepcopy(stacks)
    private_stacks.faults_input_data = copy.deepcopy(fault_inputs)
    private_descriptor = replace(descriptor, stack_structure=private_stacks)
    private_input = copy.deepcopy(interpolation_input)
    # The input may already hold a deeper volume grid. Solve references belong
    # to the last *surface* output's full production grid, including all SPs.
    private_input.set_temp_grid(copy.deepcopy(last.outputs[fault_stacks[0]].grid))
    grid_size = private_input.grid.len_all_grids
    xyz = np.concatenate((_host(private_input.grid.values), _host(private_input.all_surface_points.sp_coords)))
    block = np.zeros((n_stacks, len(xyz)), dtype=np.float64)
    for fault_stack in fault_stacks:
        fault_output = last.outputs[fault_stack].scalar_fields
        if tuple(fault_output.values_on_all_xyz.shape) != (1, len(xyz)):
            raise ValueError('invalid_fault_production_reference_layout')
        block[fault_stack] = _host(_modify_faults_values_output(FaultsData(), fault_output, xyz))
    # The solver sees the drift block in the active backend, as production does.
    backend_block = block if BT.engine_backend is AvailableBackends.numpy else BT.t.array(block, dtype=BT.dtype_obj)
    snapshots = []
    fingerprints = []
    for stack_index in range(n_stacks):
        private_stacks.stack_number = stack_index
        subset = InterpolationInput.from_interpolation_input_subset(private_input, private_stacks)
        subset.fault_values = _grab_stack_fault_data(backend_block, subset, private_stacks, grid_size)
        shape = TensorsStructure.from_tensor_structure_subset(private_descriptor, stack_index)
        solver_input = input_preprocess(shape, subset)
        overrides = private_stacks.interpolation_options_per_stack
        stack_options = copy.deepcopy(overrides[stack_index] if overrides is not None and
                                      overrides[stack_index] is not None else options)
        if stack_options.number_dimensions != 3:
            raise NotImplementedError('unsupported_fault_dimensions')
        # Validate the actual unforced solve fingerprint before any bank change.
        decision = resolve_weight_cache(stack_options, stack_index, pykeops_solver_requested(), solver_input)
        if decision.route is not WeightCacheRoute.CACHED:
            raise ValueError(f'fault fixed production weights fingerprint mismatch for stack {stack_index}; no solve allowed')
        actual_weights = getattr(last.outputs[stack_index], 'weights', None)
        if actual_weights is None:
            raise ValueError(f'missing production weight provenance for stack {stack_index}; no solve allowed')
        # NumPy outputs hold a read-only copy, Torch outputs an independent clone.
        if isinstance(actual_weights, np.ndarray):
            aliased = isinstance(decision.weights, np.ndarray) and np.shares_memory(actual_weights, decision.weights)
        elif hasattr(actual_weights, 'data_ptr'):
            aliased = hasattr(decision.weights, 'data_ptr') and actual_weights.data_ptr() == decision.weights.data_ptr()
        else:
            aliased = True
        if aliased:
            raise ValueError(f'unsupported production weight provenance for stack {stack_index}: independent snapshot required')
        actual_weights = _host(actual_weights)
        cached_weights = _host(decision.weights)
        if (cached_weights.dtype != actual_weights.dtype or
                not np.isfinite(actual_weights).all() or
                not np.array_equal(cached_weights, actual_weights)):
            raise ValueError(f'cached weights differ from actual production solve for stack {stack_index}; no solve allowed')
        weights = np.array(actual_weights, dtype=np.float64, copy=True)
        stack_options.evaluation_options.compute_scalar = True
        stack_options.evaluation_options.compute_scalar_gradient = True
        solver_input = copy.deepcopy(solver_input)
        _freeze_arrays(solver_input)
        weights.flags.writeable = False
        snapshots.append((solver_input, weights, stack_options))
        fingerprints.append(decision.fingerprint)
    return tuple(snapshots), fingerprints, private_input, grid_size


def prepare_fault_bank_sampler(descriptor, interpolation_input, options, octree_list):
    """Return raw ``query(points, bank, stacks) -> (S x M, S x M x 3)`` fields.

    Queries touching an affected stack require bank 0 or 1; fault and
    unaffected stacks are bank-independent. Only query
    coordinates and query fault drift change; solved SP/ref/rest drift does not.
    NumPy and Torch (CPU or CUDA, optionally KeOps) float64 are supported: queries
    take and return host NumPy arrays. Unsupported contracts fail before emission.
    ``snapshots`` are detached inspection copies, never the query's private state.
    """
    _check_drift_backend()
    stacks = descriptor.stack_structure
    n_stacks = stacks.n_stacks
    faults = [i for i, relation in enumerate(stacks.masking_descriptor)
              if relation is StackRelationType.FAULT]
    if len(faults) != 1:
        raise ValueError('unsupported_fault_count: exactly one independent fault required')
    fault_stack = faults[0]
    matrix = _fault_matrix(stacks)
    other_rows = matrix.copy()
    other_rows[fault_stack] = False
    if other_rows.any() or matrix[:, fault_stack].any():
        raise ValueError('unsupported_fault_relations: only independent fault -> affected edges allowed')
    affected = tuple(int(i) for i in np.flatnonzero(matrix[fault_stack]))
    unaffected = tuple(i for i in range(n_stacks) if i != fault_stack and i not in affected)
    fault_inputs = _check_fixed_query_contract(stacks, interpolation_input, octree_list)
    last = octree_list[-1]
    if _host(last.outputs[fault_stack].scalar_field_at_sp).size != 1:
        raise ValueError('unsupported_fault_surfaces: one separator required')

    snapshots, fingerprints, private_input, grid_size = _fixed_weight_snapshots(
        descriptor, interpolation_input, options, octree_list, (fault_stack,), fault_inputs)
    backend = (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS)
    on_device = BT.engine_backend is not AvailableBackends.numpy

    def device(values):
        return BT.t.array(np.ascontiguousarray(values), dtype=BT.dtype_obj) if on_device else values

    # Frozen host weights stay the inspected snapshot; evaluation uses a device copy.
    device_weights = [device(weights) for _, weights, _ in snapshots]

    def evaluate_stack(stack_index, points, bank=None):
        original, _, stack_options = snapshots[stack_index]
        view = copy.copy(original)
        view.xyz_to_interpolate = device(points)
        if stack_index in affected:
            view.fault_internal = copy.copy(original.fault_internal)
            view.fault_internal.fault_values_everywhere = device(np.full((1, len(points)), bank, dtype=np.float64))
        fields = _evaluate_sys_eq(view, device_weights[stack_index], stack_options)
        values = np.array(_host(fields._scalar_field), dtype=np.float64, copy=True)
        gradients = np.stack([_host(g) for g in (fields._gx_field, fields._gy_field, fields._gz_field)], axis=-1)
        if values.shape != (len(points),) or gradients.shape != (len(points), 3) or not (
                np.isfinite(values).all() and np.isfinite(gradients).all()):
            raise ValueError('nonfinite_or_invalid_fault_query')
        return values, gradients

    def query(points, bank=None, stacks=None):
        """Rows follow ``stacks`` (default: every stack in order)."""
        if backend != (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS):
            raise RuntimeError('fault sampler backend changed after preparation')
        stacks = tuple(range(n_stacks)) if stacks is None else tuple(stacks)
        if not stacks or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                             or not 0 <= i < n_stacks for i in stacks):
            raise ValueError('invalid_query_stacks: nonempty in-bounds stack indices required')
        if (bank is not None and (not np.isscalar(bank) or bank not in (0, 1))
                or bank is None and any(i in affected for i in stacks)):
            raise ValueError('invalid_bank: affected stack queries require explicit 0 or 1')
        points = np.array(points, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('invalid_query_points: finite (M, 3) coordinates required')
        values, gradients = zip(*(evaluate_stack(i, points, bank) for i in stacks))
        return np.stack(values), np.stack(gradients)

    # Verify the separator using actual fixed-weight scalar AND gradient kernels,
    # including off-observation domain probes, not just an observation-plane fit.
    root_points = _host(octree_list[0].grid.values)
    extent = _host(octree_list[0].grid.octree_grid.orthogonal_extent).reshape(3, 2)
    probes = np.array(list(product(*[np.linspace(a, b, 3) for a, b in extent])))
    check_points = np.concatenate((root_points, probes))
    scalar, gradients = evaluate_stack(fault_stack, check_points)
    design = np.column_stack((check_points, np.ones(len(check_points))))
    plane = np.linalg.lstsq(design, scalar, rcond=None)[0]
    scalar_error = float(np.max(np.abs(design @ plane - scalar)))
    gradient_error = float(np.max(np.abs(gradients - gradients[0])))
    # Engine kernel gradients retain their production scaling convention.
    # Verify direction, but never replace them with a fitted plane normal.
    normal_length = np.linalg.norm(plane[:3])
    gradient_length = np.linalg.norm(gradients[0])
    alignment_error = (float(np.linalg.norm(gradients[0] / gradient_length - plane[:3] / normal_length))
                       if min(normal_length, gradient_length) > 1e-12 else float('inf'))
    if (np.linalg.matrix_rank(design) != 4 or np.linalg.norm(plane[:3]) < 1e-12 or
            scalar_error > 1e-10 or gradient_error > 1e-10 or alignment_error > 1e-10):
        raise ValueError('unsupported_nonplanar_fault: actual scalar and gradients must define a plane')
    drift_ranges = []
    for level in (octree_list[0], last):
        output = level.outputs[fault_stack].scalar_fields
        level_xyz = np.concatenate((_host(output.grid.values), _host(private_input.all_surface_points.sp_coords)))
        drift = _host(_modify_faults_values_output(FaultsData(), output, level_xyz))[0]
        scalar_field = _host(output.exported_fields._scalar_field)
        isovalue = float(_host(output.scalar_field_at_sp).reshape(-1)[0])
        separator = scalar_field[:output.grid.len_all_grids] - isovalue
        grid_drift = drift[:output.grid.len_all_grids]
        away = np.abs(separator) > 1e-10
        if (not np.isfinite(drift).all() or not np.isclose(drift.min(), 0, rtol=0, atol=1e-12) or
                not np.isclose(drift.max(), 1, rtol=0, atol=1e-12) or not away.any() or
                not np.array_equal(grid_drift[away] > .5, separator[away] < 0)):
            raise ValueError('unverified_fault_banks: production drift must span 0/1 with bank 0 positive')
        drift_ranges.append((float(drift.min()), float(drift.max())))
    # Inspection snapshots do not expose mutable attributes of the closure.
    inspection = copy.deepcopy(snapshots)
    _freeze_arrays(inspection)
    return dict(fault_stack=fault_stack, affected_stacks=affected, unaffected_stacks=unaffected,
                query=query, snapshots=inspection,
                diagnostics=dict(backend=f"{BT.engine_backend.name}_{'cuda' if BT.use_gpu else 'cpu'}_float64"
                                         f"{'_keops' if BT.use_pykeops else ''}", fixed_weights=True,
                                 weight_provenance='independent_production_output_snapshot',
                                 cache_fingerprints=tuple(fingerprints), plane=tuple(plane),
                                 plane_scalar_max_error=scalar_error, plane_gradient_max_error=gradient_error,
                                 plane_gradient_alignment_error=alignment_error,
                                 production_drift_ranges=tuple(drift_ranges), bank_zero_side='positive',
                                 reference_grid_size=grid_size, surface_level=len(octree_list) - 1,
                                 field_contract='raw_stack_values_and_actual_kernel_gradients'))


def prepare_fault_drift_sampler(descriptor, interpolation_input, options, octree_list):
    """Return raw ``query(points, stacks) -> (S x M, S x M x 3)`` with production fault drift.

    Unlike the bank sampler, affected stacks see the drift the production solve
    would assign at each query point: every independent infinite fault's scalar
    is evaluated with its fixed weights, soft-segmented with the production ids,
    edges and slope, and shifted by the frozen production reference minimum.
    The rule is verified against the production drift on the full reference
    grid before any query. Fault stacks themselves are drift-independent.
    Runs on NumPy or Torch (CPU or CUDA, optionally KeOps): queries take and
    return host NumPy arrays; evaluation happens on the active device.
    """
    _check_drift_backend()
    stacks = descriptor.stack_structure
    n_stacks = stacks.n_stacks
    faults = tuple(i for i, relation in enumerate(stacks.masking_descriptor)
                   if relation is StackRelationType.FAULT)
    if not faults:
        raise ValueError('unsupported_fault_count: at least one fault required')
    matrix = _fault_matrix(stacks)
    non_faults = [i for i in range(n_stacks) if i not in faults]
    if matrix[non_faults].any() or matrix[:, list(faults)].any():
        raise ValueError('unsupported_fault_relations: only independent fault -> affected edges allowed')
    if stacks.segmentation_function is not None:
        raise NotImplementedError('unsupported_fault_segmentation_function: production slope must be a constant')
    fault_inputs = _check_fixed_query_contract(stacks, interpolation_input, octree_list)
    last = octree_list[-1]
    for fault_stack in faults:
        if _host(last.outputs[fault_stack].scalar_field_at_sp).size != 1:
            raise ValueError('unsupported_fault_surfaces: one fault surface per fault stack required')
    snapshots, fingerprints, private_input, grid_size = _fixed_weight_snapshots(
        descriptor, interpolation_input, options, octree_list, faults, fault_inputs)
    affected_by = {s: tuple(int(f) for f in np.flatnonzero(matrix[:, s])) for s in range(n_stacks)}
    backend = (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS)
    on_device = BT.engine_backend is not AvailableBackends.numpy

    def device(values):
        return BT.t.array(np.ascontiguousarray(values), dtype=BT.dtype_obj) if on_device else values

    # Frozen host weights stay the inspected snapshot; evaluation uses a device copy.
    device_weights = [device(weights) for _, weights, _ in snapshots]
    scalar_options = []
    for _, _, stack_options in snapshots:
        scalar_only = copy.deepcopy(stack_options)
        scalar_only.evaluation_options.compute_scalar_gradient = False
        scalar_options.append(scalar_only)

    def evaluate_stack(stack_index, points, drift_rows=None, gradients=True):
        original, _, stack_options = snapshots[stack_index]
        if not gradients:
            stack_options = scalar_options[stack_index]
        view = copy.copy(original)
        view.xyz_to_interpolate = device(points)
        if drift_rows is not None:
            view.fault_internal = copy.copy(original.fault_internal)
            view.fault_internal.fault_values_everywhere = device(drift_rows)
        fields = _evaluate_sys_eq(view, device_weights[stack_index], stack_options)
        values = np.array(_host(fields._scalar_field), dtype=np.float64, copy=True)
        if values.shape != (len(points),) or not np.isfinite(values).all():
            raise ValueError('nonfinite_or_invalid_fault_query')
        if not gradients:
            return values, None
        result = np.stack([_host(g) for g in (fields._gx_field, fields._gy_field, fields._gz_field)], axis=-1)
        if result.shape != (len(points), 3) or not np.isfinite(result).all():
            raise ValueError('nonfinite_or_invalid_fault_query')
        return values, result

    # Production drift rule, per fault: segmentation of its fixed-weight scalar.
    rules = {}
    probe_stacks = copy.deepcopy(stacks)
    for fault_stack in faults:
        probe_stacks.stack_number = fault_stack
        ids = np.array(InterpolationInput.from_interpolation_input_subset(
            private_input, probe_stacks).unit_values)
        output = last.outputs[fault_stack].scalar_fields
        edges = np.array(_host(output.exported_fields.scalar_field_at_surface_points), dtype=np.float64)
        slope = snapshots[fault_stack][2].sigmoid_slope
        minima = []
        for level in octree_list:
            level_output = level.outputs[fault_stack].scalar_fields
            block = _host(level_output.values_on_all_xyz)
            reference = level_output.exported_fields._macro_reference_size or block.shape[1]
            minima.append(float(block[:, :reference].min()))
        if not np.allclose(minima, minima[-1], rtol=0, atol=1e-12):
            raise ValueError('unsupported_fault_drift_reference: production drift shift differs between octree levels')
        ids.flags.writeable = edges.flags.writeable = False
        rules[fault_stack] = (ids, edges, slope, minima[-1])

    def drift(fault_stack, scalar):
        ids, edges, slope, minimum = rules[fault_stack]
        values = soft_segment_unbounded(Z=device(scalar), edges=device(edges), ids=ids, sigmoid_slope=slope)
        return np.asarray(_host(values), dtype=np.float64).reshape(-1) - minimum

    reference_xyz = np.concatenate((_host(private_input.grid.values), _host(private_input.all_surface_points.sp_coords)))
    drift_errors = []
    for fault_stack in faults:
        output = last.outputs[fault_stack].scalar_fields
        expected = _host(_modify_faults_values_output(FaultsData(), output, reference_xyz)).reshape(-1)
        rebuilt = drift(fault_stack, evaluate_stack(fault_stack, reference_xyz)[0])
        error = float(np.max(np.abs(rebuilt - expected)))
        if not np.isfinite(error) or error > 1e-10:
            raise ValueError(f'unverified_fault_drift: rebuilt production drift differs for fault stack {fault_stack}')
        drift_errors.append(error)

    def query(points, stacks=None, gradients=True):
        """Rows follow ``stacks`` (default: every stack in order); gradients are None if not requested."""
        if backend != (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS):
            raise RuntimeError('fault sampler backend changed after preparation')
        stacks = tuple(range(n_stacks)) if stacks is None else tuple(stacks)
        if not stacks or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                             or not 0 <= i < n_stacks for i in stacks):
            raise ValueError('invalid_query_stacks: nonempty in-bounds stack indices required')
        points = np.array(points, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('invalid_query_points: finite (M, 3) coordinates required')
        needed = sorted({f for s in stacks for f in affected_by[s]} | {s for s in stacks if s in faults})
        # Drift needs only fault scalars; fault gradients only when a fault row is requested.
        fault_fields = {f: evaluate_stack(f, points, gradients=gradients and f in stacks) for f in needed}
        values, normals = [], []
        for s in stacks:
            if s in fault_fields:
                v, g = fault_fields[s]
            else:
                rows = (np.stack([drift(f, fault_fields[f][0]) for f in affected_by[s]])
                        if affected_by[s] else None)
                v, g = evaluate_stack(s, points, rows, gradients)
            values.append(v)
            normals.append(g)
        return np.stack(values), (np.stack(normals) if gradients else None)

    # Stacked KeOps evaluation (as in FLAT stacks): one block-sparse reduction per
    # kind for stacks with identical kernel options and drift-row counts.
    kind_options = {kind: [] for kind in ('scalar', 'gradient')}
    for _, _, stack_options in snapshots:
        for kind, flags in (('scalar', (True, False)), ('gradient', (False, True))):
            stacked = copy.deepcopy(stack_options)
            stacked.evaluation_options.compute_scalar, stacked.evaluation_options.compute_scalar_gradient = flags
            kind_options[kind].append(stacked)

    def evaluate_many(items, kind):
        """[(stack, points, drift_rows)] -> scalars (M,) or gradients (M, 3) per item."""
        results = [None]*len(items)
        pending = list(range(len(items)))
        if BT.use_pykeops:
            groups = {}
            for i in pending:
                stack, points, rows = items[i]
                signature = (repr(snapshots[stack][2].kernel_options), 0 if rows is None else len(rows))
                groups.setdefault(signature, []).append(i)
            pending = []
            for members in groups.values():
                if len(members) < 2:
                    pending.extend(members)
                    continue
                views = []
                for i in members:
                    stack, points, rows = items[i]
                    view = copy.copy(snapshots[stack][0])
                    view.xyz_to_interpolate = device(points)
                    if rows is not None:
                        view.fault_internal = copy.copy(snapshots[stack][0].fault_internal)
                        view.fault_internal.fault_values_everywhere = device(rows)
                    views.append(view)
                fields = symbolic_evaluator_optimized_stacked(
                    views, [device_weights[items[i][0]] for i in members],
                    [kind_options[kind][items[i][0]] for i in members])
                for i, field in zip(members, fields):
                    if kind == 'scalar':
                        value = np.array(_host(field._scalar_field), dtype=np.float64).reshape(-1)
                    else:
                        value = np.stack([np.asarray(_host(g), dtype=np.float64).reshape(-1)
                                          for g in (field._gx_field, field._gy_field, field._gz_field)], axis=-1)
                    if len(value) != len(items[i][1]) or not np.isfinite(value).all():
                        raise ValueError('nonfinite_or_invalid_fault_query')
                    results[i] = value
        for i in pending:
            stack, points, rows = items[i]
            values, gradients = evaluate_stack(stack, points, rows, gradients=kind == 'gradient')
            results[i] = values if kind == 'scalar' else gradients
        return results

    def query_batch(requests):
        """Evaluate ``[(points, stacks, 'scalar'|'gradient')]`` together.

        Returns, per request, rows following its ``stacks``: (S, M) scalars or
        (S, M, 3) gradients. With KeOps, points of all requests sharing a
        (stack, kind) are merged and evaluated in stacked block-sparse reductions;
        fault scalars for the drift are evaluated first, once per fault for every
        point that needs them. Dense backends evaluate each request separately.
        """
        if backend != (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS):
            raise RuntimeError('fault sampler backend changed after preparation')
        groups = {}  # (stack, kind) -> [(request, row, points)]
        for r, (points, stacks, kind) in enumerate(requests):
            stacks = tuple(stacks)
            if kind not in ('scalar', 'gradient'):
                raise ValueError("invalid_query_kind: 'scalar' or 'gradient' required")
            if not stacks or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                                 or not 0 <= i < n_stacks for i in stacks):
                raise ValueError('invalid_query_stacks: nonempty in-bounds stack indices required')
            points = np.array(points, dtype=np.float64, copy=True)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError('invalid_query_points: finite (M, 3) coordinates required')
            for row, stack in enumerate(stacks):
                groups.setdefault((int(stack), kind), []).append((r, row, points))
        if not BT.use_pykeops:
            # Dense evaluators gain nothing from merging (larger merged batches only
            # cross the evaluator chunk size); answer each request on its own.
            answers = []
            for points, stacks, kind in requests:
                values, gradients = query(points, stacks=stacks, gradients=kind == 'gradient')
                answers.append(values if kind == 'scalar' else gradients)
            return answers
        merged = {key: np.concatenate([p for _, _, p in members]) for key, members in groups.items()}
        # Phase 1: every fault scalar needed, by drift or by request, in one pass.
        fault_parts = {f: [] for f in faults}
        for (stack, kind), points in merged.items():
            for f in affected_by[stack]:
                fault_parts[f].append(((stack, kind), points))
            if stack in faults and kind == 'scalar':
                fault_parts[stack].append(((stack, kind), points))
        fault_items = [(f, np.concatenate([p for _, p in parts])) for f, parts in fault_parts.items() if parts]
        fault_scalars = dict(zip([f for f, _ in fault_items],
                                 evaluate_many([(f, pts, None) for f, pts in fault_items], 'scalar')))
        fault_slices = {}
        for f, parts in fault_parts.items():
            offset = 0
            for key, points in parts:
                fault_slices[(f, key)] = slice(offset, offset+len(points))
                offset += len(points)
        # Phase 2: all remaining (stack, kind) groups, stacked per kind.
        evaluated = {}
        for kind in ('scalar', 'gradient'):
            keys = [key for key in merged if key[1] == kind and not (key[0] in faults and kind == 'scalar')]
            items = []
            for stack, _ in keys:
                rows = None
                if affected_by[stack]:
                    rows = np.stack([drift(f, fault_scalars[f][fault_slices[(f, (stack, kind))]])
                                     for f in affected_by[stack]])
                items.append((stack, merged[(stack, kind)], rows))
            evaluated.update(zip(keys, evaluate_many(items, kind)))
        for f in faults:
            if (f, 'scalar') in merged:
                evaluated[(f, 'scalar')] = fault_scalars[f][fault_slices[(f, (f, 'scalar'))]]
        results = []
        for points, stacks, kind in requests:
            n_points = len(np.asarray(points).reshape(-1, 3))
            shape = (len(tuple(stacks)), n_points) + ((3,) if kind == 'gradient' else ())
            results.append(np.empty(shape))
        offsets = {key: 0 for key in groups}
        for key, members in groups.items():
            for r, row, points in members:
                start = offsets[key]
                results[r][row] = evaluated[key][start:start+len(points)]
                offsets[key] = start+len(points)
        return results

    inspection = copy.deepcopy(snapshots)
    _freeze_arrays(inspection)
    return dict(fault_stacks=faults, affected_by=affected_by, query=query, query_batch=query_batch,
                snapshots=inspection,
                diagnostics=dict(backend=f"{BT.engine_backend.name}_{'cuda' if BT.use_gpu else 'cpu'}_float64"
                                         f"{'_keops' if BT.use_pykeops else ''}", fixed_weights=True,
                                 weight_provenance='independent_production_output_snapshot',
                                 cache_fingerprints=tuple(fingerprints), drift='production_soft_segmentation',
                                 drift_rebuild_max_error=tuple(drift_errors),
                                 reference_grid_size=grid_size, surface_level=len(octree_list) - 1,
                                 field_contract='raw_stack_values_and_actual_kernel_gradients'))
