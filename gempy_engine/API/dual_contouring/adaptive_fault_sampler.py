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
from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor as BT
from ...core.data import TensorsStructure
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.stack_relation_type import StackRelationType
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


def prepare_fault_bank_sampler(descriptor, interpolation_input, options, octree_list):
    """Return raw ``query(points, bank) -> (S x M, S x M x 3)`` fields.

    Full queries require bank 0 or 1 when any stack is affected. Only query
    coordinates and query fault drift change; solved SP/ref/rest drift does not.
    NumPy CPU float64 is supported. Unsupported contracts fail before emission.
    ``snapshots`` are detached inspection copies, never the query's private state.
    """
    if (BT.engine_backend is not AvailableBackends.numpy or BT.use_gpu or
            BT.use_pykeops or BT.COMPUTE_GRADS or np.dtype(BT.dtype) != np.dtype('float64') or
            pykeops_solver_requested()):
        raise NotImplementedError('unsupported_fault_backend: NumPy CPU float64 without KeOps required')
    stacks = descriptor.stack_structure
    n_stacks = stacks.n_stacks
    faults = [i for i, relation in enumerate(stacks.masking_descriptor)
              if relation is StackRelationType.FAULT]
    if len(faults) != 1:
        raise ValueError('unsupported_fault_count: exactly one independent fault required')
    fault_stack = faults[0]
    matrix = (np.zeros((n_stacks, n_stacks), dtype=bool) if stacks.faults_relations is None
              else np.asarray(stacks.faults_relations))
    if matrix.shape != (n_stacks, n_stacks) or not np.isin(matrix, (False, True)).all():
        raise ValueError('unsupported_fault_relations: directed boolean matrix required')
    matrix = matrix.astype(bool)
    other_rows = matrix.copy()
    other_rows[fault_stack] = False
    if other_rows.any() or matrix[:, fault_stack].any():
        raise ValueError('unsupported_fault_relations: only independent fault -> affected edges allowed')
    affected = tuple(int(i) for i in np.flatnonzero(matrix[fault_stack]))
    unaffected = tuple(i for i in range(n_stacks) if i != fault_stack and i not in affected)
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
    last = octree_list[-1]
    if np.asarray(last.outputs[fault_stack].scalar_field_at_sp).size != 1:
        raise ValueError('unsupported_fault_surfaces: one separator required')

    private_stacks = copy.deepcopy(stacks)
    private_stacks.faults_input_data = copy.deepcopy(fault_inputs)
    private_descriptor = replace(descriptor, stack_structure=private_stacks)
    private_input = copy.deepcopy(interpolation_input)
    # The input may already hold a deeper volume grid. Solve references belong
    # to the last *surface* output's full production grid, including all SPs.
    private_input.set_temp_grid(copy.deepcopy(last.outputs[fault_stack].grid))
    grid_size = private_input.grid.len_all_grids
    xyz = np.concatenate((private_input.grid.values, private_input.all_surface_points.sp_coords))
    fault_output = last.outputs[fault_stack].scalar_fields
    if fault_output.values_on_all_xyz.shape != (1, len(xyz)):
        raise ValueError('invalid_fault_production_reference_layout')
    block = np.zeros((n_stacks, len(xyz)), dtype=np.float64)
    block[fault_stack] = _modify_faults_values_output(FaultsData(), fault_output, xyz)
    snapshots = []
    fingerprints = []
    for stack_index in range(n_stacks):
        private_stacks.stack_number = stack_index
        subset = InterpolationInput.from_interpolation_input_subset(private_input, private_stacks)
        subset.fault_values = _grab_stack_fault_data(block, subset, private_stacks, grid_size)
        shape = TensorsStructure.from_tensor_structure_subset(private_descriptor, stack_index)
        solver_input = input_preprocess(shape, subset)
        overrides = private_stacks.interpolation_options_per_stack
        stack_options = copy.deepcopy(overrides[stack_index] if overrides is not None and
                                      overrides[stack_index] is not None else options)
        if stack_options.number_dimensions != 3:
            raise NotImplementedError('unsupported_fault_dimensions')
        # Validate the actual unforced solve fingerprint before any bank change.
        decision = resolve_weight_cache(stack_options, stack_index, False, solver_input)
        if decision.route is not WeightCacheRoute.CACHED:
            raise ValueError(f'fault fixed production weights fingerprint mismatch for stack {stack_index}; no solve allowed')
        actual_weights = getattr(last.outputs[stack_index], 'weights', None)
        if actual_weights is None:
            raise ValueError(f'missing production weight provenance for stack {stack_index}; no solve allowed')
        if not isinstance(actual_weights, np.ndarray) or np.shares_memory(actual_weights, decision.weights):
            raise ValueError(f'unsupported production weight provenance for stack {stack_index}: independent snapshot required')
        cached_weights = np.asarray(decision.weights)
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
    snapshots = tuple(snapshots)
    backend = (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS)

    def evaluate_stack(stack_index, points, bank=None):
        original, weights, stack_options = snapshots[stack_index]
        view = copy.copy(original)
        view.xyz_to_interpolate = points
        if stack_index in affected:
            view.fault_internal = copy.copy(original.fault_internal)
            view.fault_internal.fault_values_everywhere = np.full((1, len(points)), bank, dtype=np.float64)
        fields = _evaluate_sys_eq(view, weights, stack_options)
        values = np.array(fields._scalar_field, copy=True)
        gradients = np.stack((fields._gx_field, fields._gy_field, fields._gz_field), axis=-1)
        if values.shape != (len(points),) or gradients.shape != (len(points), 3) or not (
                np.isfinite(values).all() and np.isfinite(gradients).all()):
            raise ValueError('nonfinite_or_invalid_fault_query')
        return values, gradients

    def query(points, bank=None):
        if backend != (BT.engine_backend, BT.dtype, BT.use_gpu, BT.use_pykeops, BT.COMPUTE_GRADS):
            raise RuntimeError('fault sampler backend changed after preparation')
        if bank is not None and (not np.isscalar(bank) or bank not in (0, 1)) or affected and bank is None:
            raise ValueError('invalid_bank: affected full queries require explicit 0 or 1')
        points = np.array(points, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('invalid_query_points: finite (M, 3) coordinates required')
        values, gradients = zip(*(evaluate_stack(i, points, bank) for i in range(n_stacks)))
        return np.stack(values), np.stack(gradients)

    # Verify the separator using actual fixed-weight scalar AND gradient kernels,
    # including off-observation domain probes, not just an observation-plane fit.
    root_points = np.asarray(octree_list[0].grid.values)
    extent = np.asarray(octree_list[0].grid.octree_grid.orthogonal_extent).reshape(3, 2)
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
        level_xyz = np.concatenate((output.grid.values, private_input.all_surface_points.sp_coords))
        drift = np.asarray(_modify_faults_values_output(FaultsData(), output, level_xyz))[0]
        scalar_field = np.asarray(output.exported_fields._scalar_field)
        isovalue = float(np.asarray(output.scalar_field_at_sp).reshape(-1)[0])
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
                diagnostics=dict(backend='numpy_cpu_float64', fixed_weights=True,
                                 weight_provenance='independent_production_output_snapshot',
                                 cache_fingerprints=tuple(fingerprints), plane=tuple(plane),
                                 plane_scalar_max_error=scalar_error, plane_gradient_max_error=gradient_error,
                                 plane_gradient_alignment_error=alignment_error,
                                 production_drift_ranges=tuple(drift_ranges), bank_zero_side='positive',
                                 reference_grid_size=grid_size, surface_level=len(octree_list) - 1,
                                 field_contract='raw_stack_values_and_actual_kernel_gradients'))
