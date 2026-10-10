"""Detached fixed-weight field queries with the production fault drift.

Prepare during public compute_model's extraction hook (or later, from the
outputs' retained weights). Values are raw stack potentials, not
surface-expanded fields or lithology IDs.
"""

import copy

import numpy as np

from ..interp_single._aux_faults_ops import _modify_faults_values_output
from ..interp_single._interp_scalar_field import _evaluate_sys_eq
from .fixed_weight_snapshots import (check_fixed_query_contract, check_fixed_weight_backend, fault_matrix,
                                     fixed_weight_snapshots, freeze_arrays, host)
from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor as BT
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.stack_relation_type import StackRelationType
from ...modules.activator._soft_segment import soft_segment_unbounded
from ...modules.evaluator.symbolic_evaluator import symbolic_evaluator_optimized_stacked


def prepare_fault_drift_sampler(descriptor, interpolation_input, options, octree_list):
    """Return raw ``query(points, stacks) -> (S x M, S x M x 3)`` with production fault drift.

    Affected stacks see the drift the production solve would assign at each
    query point: every independent infinite fault's scalar
    is evaluated with its fixed weights, soft-segmented with the production ids,
    edges and slope, and shifted by the frozen production reference minimum.
    The rule is verified against the production drift on the full reference
    grid before any query. Fault stacks themselves are drift-independent.
    Runs on NumPy or Torch (CPU or CUDA, optionally KeOps): queries take and
    return host NumPy arrays; evaluation happens on the active device.
    """
    check_fixed_weight_backend()
    stacks = descriptor.stack_structure
    n_stacks = stacks.n_stacks
    faults = tuple(i for i, relation in enumerate(stacks.masking_descriptor)
                   if relation is StackRelationType.FAULT)
    if not faults:
        raise ValueError('unsupported_fault_count: at least one fault required')
    matrix = fault_matrix(stacks)
    non_faults = [i for i in range(n_stacks) if i not in faults]
    if matrix[non_faults].any() or matrix[:, list(faults)].any():
        raise ValueError('unsupported_fault_relations: only independent fault -> affected edges allowed')
    if stacks.segmentation_function is not None:
        raise NotImplementedError('unsupported_fault_segmentation_function: production slope must be a constant')
    fault_inputs = check_fixed_query_contract(stacks, interpolation_input, octree_list)
    last = octree_list[-1]
    for fault_stack in faults:
        if host(last.outputs[fault_stack].scalar_field_at_sp).size != 1:
            raise ValueError('unsupported_fault_surfaces: one fault surface per fault stack required')
    snapshots, fingerprints, private_input, grid_size = fixed_weight_snapshots(
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
        values = np.array(host(fields._scalar_field), dtype=np.float64, copy=True)
        if values.shape != (len(points),) or not np.isfinite(values).all():
            raise ValueError('nonfinite_or_invalid_fault_query')
        if not gradients:
            return values, None
        result = np.stack([host(g) for g in (fields._gx_field, fields._gy_field, fields._gz_field)], axis=-1)
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
        edges = np.array(host(output.exported_fields.scalar_field_at_surface_points), dtype=np.float64)
        slope = snapshots[fault_stack][2].sigmoid_slope
        minima = []
        for level in octree_list:
            level_output = level.outputs[fault_stack].scalar_fields
            block = host(level_output.values_on_all_xyz)
            reference = level_output.exported_fields._macro_reference_size or block.shape[1]
            minima.append(float(block[:, :reference].min()))
        if not np.allclose(minima, minima[-1], rtol=0, atol=1e-12):
            raise ValueError('unsupported_fault_drift_reference: production drift shift differs between octree levels')
        ids.flags.writeable = edges.flags.writeable = False
        rules[fault_stack] = (ids, edges, slope, minima[-1])

    def drift(fault_stack, scalar):
        ids, edges, slope, minimum = rules[fault_stack]
        values = soft_segment_unbounded(Z=device(scalar), edges=device(edges), ids=ids, sigmoid_slope=slope)
        return np.asarray(host(values), dtype=np.float64).reshape(-1) - minimum

    reference_xyz = np.concatenate((host(private_input.grid.values), host(private_input.all_surface_points.sp_coords)))
    drift_errors = []
    for fault_stack in faults:
        output = last.outputs[fault_stack].scalar_fields
        expected = host(_modify_faults_values_output(FaultsData(), output, reference_xyz)).reshape(-1)
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
                        value = np.array(host(field._scalar_field), dtype=np.float64).reshape(-1)
                    else:
                        value = np.stack([np.asarray(host(g), dtype=np.float64).reshape(-1)
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
    freeze_arrays(inspection)
    return dict(fault_stacks=faults, affected_by=affected_by, query=query, query_batch=query_batch,
                snapshots=inspection,
                diagnostics=dict(backend=f"{BT.engine_backend.name}_{'cuda' if BT.use_gpu else 'cpu'}_float64"
                                         f"{'_keops' if BT.use_pykeops else ''}", fixed_weights=True,
                                 weight_provenance='independent_production_output_snapshot',
                                 cache_fingerprints=tuple(fingerprints), drift='production_soft_segmentation',
                                 drift_rebuild_max_error=tuple(drift_errors),
                                 reference_grid_size=grid_size, surface_level=len(octree_list) - 1,
                                 field_contract='raw_stack_values_and_actual_kernel_gradients'))
