"""Detached fixed-weight field queries with the production fault drift.

Prepare during public compute_model's extraction hook (or later, from the
outputs' retained weights). Values are raw stack potentials, not
surface-expanded fields or lithology IDs.
"""

import copy

import numpy as np

from ..interp_single._aux_faults_ops import _modify_faults_values_output, finite_fault_frame_indices, finite_fault_taper
from ..interp_single._interp_scalar_field import _evaluate_sys_eq
from .fixed_weight_snapshots import (check_fixed_query_contract, check_fixed_weight_backend, fault_matrix,
                                     fixed_weight_snapshots, freeze_arrays, host, reference_points)
from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor as BT
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.stack_relation_type import StackRelationType
from ...modules.activator._soft_segment import soft_segment_unbounded
from ...modules.dual_contouring.joint_lattice import precision_tolerance
from ...modules.evaluator.micro_correction import apply_micro_correction
from ...modules.evaluator.symbolic_evaluator import symbolic_evaluator_optimized_stacked


def prepare_fault_drift_sampler(descriptor, interpolation_input, options, octree_list):
    """Return raw ``query(points, stacks) -> (S x M, S x M x 3)`` with production fault drift.

    Affected stacks see the drift the production solve would assign at each
    query point: every independent fault's scalar
    is evaluated with its fixed weights, soft-segmented with the production ids,
    edges and slope, and shifted by the frozen production reference minimum.
    A finite fault's drift is further scaled by its slip taper at the query
    point's projection onto the fault, in the frozen production slip frame.
    The rule is verified against the production drift on the full reference
    grid before any query. A fault offset by earlier faults takes their drift
    like any other stack, and each stack's production micro correction is added
    at the query point, never refitted.
    Runs on NumPy or Torch (CPU or CUDA, optionally KeOps): queries take and
    return host NumPy arrays; evaluation happens on the active device. Without
    faults every stack is a plain fixed-weight evaluation.
    ``query_sides(points, stacks, sides)`` instead fixes each fault's drift to
    the plateau of one side per point (see its docstring).
    Non-fault stacks with external callbacks (e.g. mesh-field null space) are
    evaluated by their callbacks with no drift, exactly as production does.
    """
    check_fixed_weight_backend()
    stacks = descriptor.stack_structure
    n_stacks = stacks.n_stacks
    faults = tuple(i for i, relation in enumerate(stacks.masking_descriptor)
                   if relation is StackRelationType.FAULT)
    matrix = fault_matrix(stacks)
    non_faults = [i for i in range(n_stacks) if i not in faults]
    # Production evaluates stacks in order, so a fault only takes drift from earlier faults.
    if matrix[non_faults].any() or np.tril(matrix[np.ix_(faults, faults)]).any():
        raise ValueError('unsupported_fault_relations: only earlier fault -> affected edges allowed')
    if stacks.segmentation_function is not None:
        raise NotImplementedError('unsupported_fault_segmentation_function: production slope must be a constant')
    fault_inputs = check_fixed_query_contract(stacks, interpolation_input, octree_list, allow_finite_faults=True,
                                              allow_external_stacks=True, allow_micro=True)
    external = {s: callback for s, callback in enumerate(stacks.interp_functions_per_stack or ())
                if callback is not None}
    last = octree_list[-1]
    for fault_stack in faults:
        if host(last.outputs[fault_stack].scalar_field_at_sp).size != 1:
            raise ValueError('unsupported_fault_surfaces: one fault surface per fault stack required')
    snapshots, fingerprints, private_input, grid_size = fixed_weight_snapshots(
        descriptor, interpolation_input, options, octree_list, faults, fault_inputs, tuple(external))
    # Production never adds fault drift to an external callback's field.
    affected_by = {s: () if s in external else tuple(int(f) for f in np.flatnonzero(matrix[:, s]))
                   for s in range(n_stacks)}
    upstream, depth = {}, {}  # Per fault: every fault its drift depends on, and its evaluation tier.
    for f in faults:
        upstream[f] = set(affected_by[f]).union(*(upstream[g] for g in affected_by[f]))
        depth[f] = 1 + max((depth[g] for g in affected_by[f]), default=-1)
    tiers = [[f for f in faults if depth[f] == tier] for tier in range(1 + max(depth.values(), default=-1))]
    corrections = [copy.deepcopy(output.exported_fields.micro_correction) for output in last.outputs]
    freeze_arrays(corrections)
    backend = (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS)
    on_device = BT.engine_backend is not AvailableBackends.numpy

    def device(values):
        # Production precision on every backend; NumPy float64 stays a view.
        return BT.t.array(np.ascontiguousarray(values), dtype=BT.dtype_obj) if on_device else np.asarray(values, dtype=BT.dtype)

    # Frozen host weights stay the inspected snapshot; evaluation uses a device copy.
    device_weights = [None if snapshot is None else device(snapshot[1]) for snapshot in snapshots]
    scalar_options = []
    for snapshot in snapshots:
        scalar_only = None if snapshot is None else copy.deepcopy(snapshot[2])
        if scalar_only is not None:
            scalar_only.evaluation_options.compute_scalar_gradient = False
        scalar_options.append(scalar_only)

    def evaluate_external(stack_index, points, gradients):
        callback = external[stack_index]
        functions = (callback.implicit_function,) + (
            (callback.gx_function, callback.gy_function, callback.gz_function) if gradients else ())
        if any(function is None for function in functions):
            raise ValueError(f'unsupported_external_fault_callback: stack {stack_index} needs gradient callbacks')
        xyz = device(points)
        fields = [np.array(host(BT.t.array(function(xyz), dtype=BT.dtype)), dtype=np.float64).reshape(-1)
                  for function in functions]
        if any(field.shape != (len(points),) or not np.isfinite(field).all() for field in fields):
            raise ValueError('nonfinite_or_invalid_fault_query')
        return fields[0], (np.stack(fields[1:], axis=-1) if gradients else None)

    def evaluate_stack(stack_index, points, drift_rows=None, gradients=True):
        if stack_index in external:
            return evaluate_external(stack_index, points, gradients)
        original, _, stack_options = snapshots[stack_index]
        if not gradients:
            stack_options = scalar_options[stack_index]
        view = copy.copy(original)
        view.xyz_to_interpolate = device(points)
        if drift_rows is not None:
            view.fault_internal = copy.copy(original.fault_internal)
            view.fault_internal.fault_values_everywhere = device(drift_rows)
        fields = apply_micro_correction(_evaluate_sys_eq(view, device_weights[stack_index], stack_options),
                                        view.xyz_to_interpolate, corrections[stack_index])
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
        if not np.allclose(minima, minima[-1], rtol=0, atol=precision_tolerance(1e-12, BT.dtype)):
            raise ValueError('unsupported_fault_drift_reference: production drift shift differs between octree levels')
        ids.flags.writeable = edges.flags.writeable = False
        rules[fault_stack] = (ids, edges, slope, minima[-1], _finite_slip_frame(
            fault_inputs[fault_stack], octree_list, fault_stack, private_input))

    finite_faults = {f for f in faults if rules[f][4] is not None}

    def step(fault_stack, scalar):
        """Shifted soft segmentation of the fault's scalar: the whole drift of an infinite fault."""
        ids, edges, slope, minimum, _ = rules[fault_stack]
        values = soft_segment_unbounded(Z=device(scalar), edges=device(edges), ids=ids, sigmoid_slope=slope)
        return np.asarray(host(values), dtype=np.float64).reshape(-1) - minimum

    def drift(fault_stack, scalar, gradients=None, points=None):
        """Production drift from the fault's scalar; finite faults also need its gradients and the points."""
        values = step(fault_stack, scalar)
        return values if fault_stack not in finite_faults else values * taper(fault_stack, scalar, gradients, points)

    def drift_rows(stack_index, fault_fields, points):
        """Drift of every fault offsetting a stack; ``fault_fields`` holds their (scalar, gradients)."""
        offsetting = affected_by[stack_index]
        return np.stack([drift(f, *fault_fields[f], points) for f in offsetting]) if offsetting else None

    def taper(fault_stack, scalar, gradients, points):
        finite_fault, isovalue, normal = rules[fault_stack][4]
        return finite_fault_taper(finite_fault, np.asarray(points, dtype=np.float64), np.asarray(scalar),
                                  tuple(np.asarray(gradients).T), isovalue, normal=normal)[0]

    reference_xyz = reference_points(private_input)
    drift_errors = []
    for fault_stack in faults:
        output = last.outputs[fault_stack].scalar_fields
        expected = host(_modify_faults_values_output(
            fault_inputs[fault_stack] or FaultsData(), copy.copy(output), reference_xyz)).reshape(-1)
        # Only a finite fault's taper needs the fault gradients. At these points
        # the snapshot's own rows are the production drift of earlier faults.
        scalar, gradients = evaluate_stack(fault_stack, reference_xyz, gradients=fault_stack in finite_faults)
        rebuilt = drift(fault_stack, scalar, gradients, reference_xyz)
        tolerance = precision_tolerance(1e-10, BT.dtype)
        error = np.abs(rebuilt - expected)
        if np.isfinite(error).all() and (error > tolerance).any():
            # On the fault the steep step amplifies the rebuilt scalar's rounding (another
            # evaluation order than production): accept drifts the rule gives within it.
            rounding = 64 * np.finfo(BT.dtype).eps * max(float(np.max(np.abs(scalar))), 1.0)
            low, high = (drift(fault_stack, scalar + shift, gradients, reference_xyz) for shift in (-rounding, rounding))
            within = (expected >= np.minimum(low, high) - tolerance) & (expected <= np.maximum(low, high) + tolerance)
            error = np.where(within, 0.0, error)
        error = float(np.max(error))
        if not np.isfinite(error) or error > tolerance:
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
        requested = {f for s in stacks for f in affected_by[s]} | {s for s in stacks if s in faults}
        needed = sorted(requested.union(*(upstream[f] for f in requested)))
        # Drift needs fault scalars, and a finite fault's gradients for the projection;
        # otherwise fault gradients only when a fault row is requested.
        fault_fields = {}
        for f in needed:
            fault_fields[f] = evaluate_stack(f, points, drift_rows(f, fault_fields, points),
                                             gradients=(gradients and f in stacks) or f in finite_faults)
        values, normals = [], []
        for s in stacks:
            if s in fault_fields:
                v, g = fault_fields[s]
                g = g if gradients else None
            else:
                v, g = evaluate_stack(s, points, drift_rows(s, fault_fields, points), gradients)
            values.append(v)
            normals.append(g)
        return np.stack(values), (np.stack(normals) if gradients else None)

    # Stacked KeOps evaluation (as in FLAT stacks): one block-sparse reduction per
    # kind for stacks with identical kernel options and drift-row counts.
    kind_options = {kind: [] for kind in ('scalar', 'gradient')}
    for snapshot in snapshots:
        for kind, flags in (('scalar', (True, False)), ('gradient', (False, True))):
            if snapshot is None:
                kind_options[kind].append(None)
                continue
            stacked = copy.deepcopy(snapshot[2])
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
                # External stacks never stack: a unique signature sends them to evaluate_stack.
                signature = (('external', i) if stack in external else
                             (repr(snapshots[stack][2].kernel_options), 0 if rows is None else len(rows)))
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
                for i, view, field in zip(members, views, fields):
                    field = apply_micro_correction(field, view.xyz_to_interpolate, corrections[items[i][0]])
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
        point that needs them, each tier with the drift of the tiers before it.
        Dense backends evaluate each request separately.
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
        # Phase 1: every fault scalar needed, by drift or by request, tier by tier over one point pool.
        pool = np.concatenate((np.empty((0, 3)), *merged.values()))
        ends = np.cumsum([len(points) for points in merged.values()])
        slices = {key: slice(end-len(points), end) for (key, points), end in zip(merged.items(), ends)}
        need = {f: np.zeros(len(pool), dtype=bool) for f in faults}
        for (stack, kind), rows in slices.items():
            for f in affected_by[stack]:
                need[f][rows] = True
            if stack in faults and kind == 'scalar':
                need[stack][rows] = True
        for f in reversed(faults):
            for earlier in affected_by[f]:
                need[earlier] |= need[f]
        fault_scalars = {f: np.empty(len(pool)) for f in faults}
        fault_gradients = {f: np.empty((len(pool), 3)) for f in finite_faults}

        def rows_at(stack, selection):
            return drift_rows(stack, {f: (fault_scalars[f][selection],
                                          fault_gradients[f][selection] if f in finite_faults else None)
                                      for f in affected_by[stack]}, pool[selection])

        for tier in tiers:
            items = [(f, pool[need[f]], rows_at(f, need[f])) for f in tier if need[f].any()]
            for (f, _, _), values in zip(items, evaluate_many(items, 'scalar')):
                fault_scalars[f][need[f]] = values
            finite_items = [item for item in items if item[0] in finite_faults]
            for (f, _, _), gradients in zip(finite_items, evaluate_many(finite_items, 'gradient')):
                fault_gradients[f][need[f]] = gradients
        # Phase 2: all remaining (stack, kind) groups, stacked per kind.
        evaluated = {}
        for kind in ('scalar', 'gradient'):
            keys = [key for key in merged if key[1] == kind and not (key[0] in faults and kind == 'scalar')]
            items = [(stack, merged[(stack, kind)], rows_at(stack, slices[(stack, kind)])) for stack, _ in keys]
            evaluated.update(zip(keys, evaluate_many(items, kind)))
        for f in faults:
            if (f, 'scalar') in merged:
                evaluated[(f, 'scalar')] = fault_scalars[f][slices[(f, 'scalar')]]
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

    # Drift plateaus far on either side of each fault (production drift saturates there),
    # before any finite-fault taper.
    plateaus = {}
    for f in faults:
        edges = rules[f][1]
        far = 1e3*(1+float(np.max(np.abs(edges))))
        negative, positive = step(f, np.array([float(edges.min())-far, float(edges.max())+far]))
        plateaus[f] = (float(negative), float(positive))

    def query_sides(points, stacks, sides):
        """Raw scalars of ``stacks`` (S, M) with every fault drift fixed to one side.

        ``sides`` (len(fault_stacks), M) holds +1 for the positive and -1 for the
        negative fault-potential side of each point's fault block: affected
        stacks see the drift plateau of that side instead of the soft step, so
        each block's field is smooth up to and across its faults. A finite
        fault's plateau is scaled by its slip taper at each point. Independent
        faults only.
        """
        if backend != (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS):
            raise RuntimeError('fault sampler backend changed after preparation')
        if any(affected_by[f] for f in faults):
            raise ValueError('unsupported_fault_relations: fault sides require independent faults')
        stacks = tuple(stacks)
        if not stacks or any(not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                             or not 0 <= i < n_stacks for i in stacks):
            raise ValueError('invalid_query_stacks: nonempty in-bounds stack indices required')
        points = np.array(points, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('invalid_query_points: finite (M, 3) coordinates required')
        sides = np.asarray(sides)
        if sides.shape != (len(faults), len(points)) or not np.isin(sides, (-1, 1)).all():
            raise ValueError('invalid_fault_sides: +1 or -1 per fault and point required')
        if not len(points):
            return np.empty((len(stacks), 0))
        used = sorted({f for s in stacks for f in affected_by[s]} & finite_faults)
        tapers = {f: taper(f, *evaluate_stack(f, points), points) for f in used}
        items = []
        for s in stacks:
            rows = None
            if affected_by[s]:
                rows = np.stack([np.where(sides[faults.index(f)] > 0, *plateaus[f][::-1]) * tapers.get(f, 1.)
                                 for f in affected_by[s]])
            items.append((int(s), points, rows))
        return np.stack(evaluate_many(items, 'scalar'))

    inspection = copy.deepcopy(snapshots)
    freeze_arrays(inspection)
    return dict(fault_stacks=faults, affected_by=affected_by, query=query, query_batch=query_batch,
                query_sides=query_sides, drift_plateaus=plateaus, snapshots=inspection,
                diagnostics=dict(backend=f"{BT.engine_backend.name}_{'cuda' if BT.use_gpu else 'cpu'}_{BT.dtype}"
                                         f"{'_keops' if BT.use_pykeops else ''}", fixed_weights=True,
                                 weight_provenance='independent_production_output_snapshot',
                                 cache_fingerprints=tuple(fingerprints), drift='production_soft_segmentation',
                                 drift_rebuild_max_error=tuple(drift_errors),
                                 finite_fault_stacks=tuple(sorted(finite_faults)),
                                 external_stacks=tuple(sorted(external)),
                                 reference_grid_size=grid_size, surface_level=len(octree_list) - 1,
                                 field_contract='raw_stack_values_and_actual_kernel_gradients'))


def _finite_slip_frame(fault_input, octree_list, fault_stack, private_input):
    """``(finite fault, isovalue, frozen slip-frame normal)`` of a finite fault stack, else None.

    Production picks the frame normal on each surface level from the fault's
    surface points, so every level normally agrees; the drift is rebuilt from
    one frame, so a level that disagrees is still rejected.
    """
    if fault_input is None or not fault_input.finite_fault_defined:
        return None
    normals = []
    for level in octree_list:
        output = level.outputs[fault_stack].scalar_fields
        fields = output.exported_fields
        gradients = (fields.gx_field_everywhere, fields.gy_field_everywhere, fields.gz_field_everywhere)
        if any(gradient is None for gradient in gradients):
            raise ValueError('invalid_finite_fault_reference: production fault gradients required')
        xyz = reference_points(private_input, output.grid)
        isovalue = float(host(output.scalar_field_at_sp).reshape(-1)[0])
        _, normal = finite_fault_taper(
            fault_input.finite_fault, xyz, np.asarray(host(fields.scalar_field_everywhere), dtype=np.float64),
            tuple(np.asarray(host(g), dtype=np.float64) for g in gradients), isovalue,
            frame_indices=finite_fault_frame_indices(fields, fields._macro_reference_size or len(xyz)))
        normals.append(np.asarray(normal, dtype=np.float64))
    units = np.array([n / np.linalg.norm(n) for n in normals])
    if not np.allclose(units, units[-1], rtol=0, atol=precision_tolerance(1e-10, BT.dtype)):
        raise ValueError('unsupported_finite_fault_frame: production slip-frame normal differs between octree levels')
    normal = normals[-1].copy()
    normal.flags.writeable = False
    return fault_input.finite_fault, isovalue, normal
