"""Frozen fixed-weight evaluation state of a finished production solve.

Weights come from each output (``InterpOutput.weights``, an independent copy
of the actual solve). While the production weight cache is live, its
fingerprint must match the rebuilt solver input and its payload the output
weights, so a query can never evaluate stale or re-solved weights. Nothing
here solves or writes the cache.
"""

import copy
from dataclasses import replace

import numpy as np

from ..interp_single._aux_faults_ops import _grab_stack_fault_data, _modify_faults_values_output
from ..interp_single._interp_single_feature import input_preprocess
from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor as BT
from ...core.data import TensorsStructure
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.options import InterpolationOptions
from ...core.data.stack_relation_type import StackRelationType
from ...modules.solver.interpolation_solver import pykeops_solver_requested
from ...modules.weights_cache.weight_cache_policy import resolve_weight_cache, WeightCacheRoute
from ...modules.weights_cache.weights_cache_interface import WeightCache


def host(values):
    """NumPy view/copy of a NumPy array or a (possibly CUDA) Torch tensor."""
    return BT.t.to_numpy(values) if hasattr(values, 'detach') else np.asarray(values)


def freeze_arrays(value):
    """Freeze arrays in an already private object graph."""
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, (list, tuple)):
        for item in value:
            freeze_arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            freeze_arrays(item)
    elif hasattr(value, '__dict__') and not isinstance(value, type):
        for item in vars(value).values():
            freeze_arrays(item)


def check_fixed_weight_backend():
    """NumPy, or Torch on CPU or CUDA, optionally with KeOps; float64, no autograd."""
    if (BT.engine_backend not in (AvailableBackends.numpy, AvailableBackends.PYTORCH) or BT.COMPUTE_GRADS
            or np.dtype(str(BT.dtype)) != np.dtype('float64')):
        raise NotImplementedError('unsupported_fault_backend: float64 NumPy or Torch without autograd required')


def fault_matrix(stacks):
    n_stacks = stacks.n_stacks
    matrix = (np.zeros((n_stacks, n_stacks), dtype=bool) if stacks.faults_relations is None
              else np.asarray(host(stacks.faults_relations)))
    if matrix.shape != (n_stacks, n_stacks) or not np.isin(matrix, (False, True)).all():
        raise ValueError('unsupported_fault_relations: directed boolean matrix required')
    return matrix.astype(bool)


def check_fixed_query_contract(stacks, interpolation_input, octree_list):
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


def production_weights(stack_options, stack_index, solver_input, output_weights, label='fixed'):
    """Independent frozen copy of a stack's actual production weights, and the cache fingerprint.

    The fingerprint is None when the production cache is not live (cache off,
    or ``compute_model`` already returned and cleared it).
    """
    if output_weights is None:
        raise ValueError(f'missing production weight provenance for stack {stack_index}; no solve allowed')
    fingerprint = None
    if _cache_live(stack_options, stack_index):
        decision = resolve_weight_cache(stack_options, stack_index, pykeops_solver_requested(), solver_input)
        if decision.route is not WeightCacheRoute.CACHED:
            raise ValueError(f'{label} fixed production weights fingerprint mismatch for stack {stack_index}; '
                             'no solve allowed')
        # NumPy outputs hold a read-only copy, Torch outputs an independent clone.
        if isinstance(output_weights, np.ndarray):
            aliased = isinstance(decision.weights, np.ndarray) and np.shares_memory(output_weights, decision.weights)
        elif hasattr(output_weights, 'data_ptr'):
            aliased = hasattr(decision.weights, 'data_ptr') and output_weights.data_ptr() == decision.weights.data_ptr()
        else:
            aliased = True
        if aliased:
            raise ValueError(f'unsupported production weight provenance for stack {stack_index}: '
                             'independent snapshot required')
        cached, actual = host(decision.weights), host(output_weights)
        if cached.dtype != actual.dtype or not np.array_equal(cached, actual):
            raise ValueError(f'cached weights differ from actual production solve for stack {stack_index}; '
                             'no solve allowed')
        fingerprint = decision.fingerprint
    if not np.isfinite(host(output_weights)).all():
        raise ValueError(f'nonfinite production weights for stack {stack_index}')
    if hasattr(output_weights, 'detach'):
        return output_weights.detach().clone(), fingerprint
    weights = np.array(output_weights, copy=True)
    weights.flags.writeable = False
    return weights, fingerprint


def _cache_live(stack_options, stack_index):
    return (stack_options.cache_mode in (InterpolationOptions.CacheMode.CACHE,
                                         InterpolationOptions.CacheMode.IN_MEMORY_CACHE)
            and not BT.COMPUTE_GRADS and stack_options.temp_interpolation_values.start_computation_ts != -1
            and WeightCache.load_weights(f'{stack_options.cache_model_name}.{stack_index}', False) is not None)


def fixed_weight_snapshots(descriptor, interpolation_input, options, octree_list, fault_stacks, fault_inputs):
    """Frozen (solver_input, weights, options) per stack, from the outputs' production weights.

    Production fault drift for every stack in ``fault_stacks`` is rebuilt on the
    last surface level's reference grid, exactly as the production solve saw it.
    Returns ``(snapshots, fingerprints, private_input, reference_grid_size)``.
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
    private_input.set_temp_grid(copy.deepcopy(last.outputs[fault_stacks[0] if fault_stacks else 0].grid))
    grid_size = private_input.grid.len_all_grids
    xyz = np.concatenate((host(private_input.grid.values), host(private_input.all_surface_points.sp_coords)))
    block = np.zeros((n_stacks, len(xyz)), dtype=np.float64)
    for fault_stack in fault_stacks:
        fault_output = last.outputs[fault_stack].scalar_fields
        if tuple(fault_output.values_on_all_xyz.shape) != (1, len(xyz)):
            raise ValueError('invalid_fault_production_reference_layout')
        block[fault_stack] = host(_modify_faults_values_output(FaultsData(), fault_output, xyz))
    # The solver sees the drift block in the active backend, as production does.
    backend_block = block if BT.engine_backend is AvailableBackends.numpy else BT.t.array(block, dtype=BT.dtype_obj)
    snapshots, fingerprints = [], []
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
        weights, fingerprint = production_weights(stack_options, stack_index, solver_input,
                                                  getattr(last.outputs[stack_index], 'weights', None), 'fault')
        weights = np.array(host(weights), dtype=np.float64, copy=True)
        stack_options.evaluation_options.compute_scalar = True
        stack_options.evaluation_options.compute_scalar_gradient = True
        solver_input = copy.deepcopy(solver_input)
        freeze_arrays(solver_input)
        weights.flags.writeable = False
        snapshots.append((solver_input, weights, stack_options))
        fingerprints.append(fingerprint)
    return tuple(snapshots), fingerprints, private_input, grid_size
