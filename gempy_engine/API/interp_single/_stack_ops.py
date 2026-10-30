import concurrent.futures
import numbers
import os
from copy import copy

from ._aux_faults_ops import _grab_stack_fault_data, _modify_faults_values_output, _options_with_finite_fault_gradients
from ._interp_scalar_field import _deduplicate_corners, _evaluate_sys_eq, _restore_corner_fields, compute_weights
from ._interp_single_feature import interpolate_feature_with_external_function, preprocess_stack_data
from ... import config
from ...config import AvailableBackends
from ...core.backend_tensor import BackendTensor
from ...core.compute_scope import (checking, StackCache, assert_same, options_key, scope_cache, stack_cache,
                                   stack_cache_of)
from ...core.data import TensorsStructure, SurfacePoints, Orientations, SurfacePointsInternals, OrientationsInternals
from ...core.data.exported_fields import ExportedFields
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.internal_structs import EvaluatorInput, SegmentationInput, SolverInput_v2
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.options import InterpolationOptions
from ...core.data.scalar_field_output import ScalarFieldOutput
from ...core.data.stack_relation_type import StackRelationType
from ...core.data.stacks_structure import StacksStructure
from dataclasses import dataclass
from typing import Any

from numpy import ndarray, dtype

from ...modules.activator import activator_interface
from ...modules.activator._soft_segment import segmentation_constants, soft_segment_stacked
from ...modules.data_preprocess._input_preparation import chunk_preprocess
from ...modules.data_preprocess import data_preprocess_interface
from ...modules.evaluator.micro_correction import fit_micro_fields, micro_evaluation_options
from ...modules.kernel_constructor.stacked_covariance import MAX_STACKED_SIZE, covariance_size, stacked_covariances
from ...modules.solver.interpolation_solver import pykeops_solver_requested
from ...modules.weights_cache.weight_cache_policy import fingerprint_needed, prefetch_fingerprints


@dataclass
class InterpolationState:
    root_data_descriptor: InputDataDescriptor
    root_interpolation_input: InterpolationInput
    options: InterpolationOptions
    stack_structure: StacksStructure
    all_stack_values_block: ndarray[tuple[Any, ...], dtype[Any]]
    interpolation_inputs: list[InterpolationInput | None]
    tensor_structs: list[TensorsStructure | None]
    solver_inputs: list
    eval_inputs: list
    all_scalar_fields_outputs: list[ScalarFieldOutput | None]


def process_chunk(state: InterpolationState, chunk: list[int]):
    if _is_external_chunk(state.stack_structure, chunk):
        _process_external_chunk(state, chunk)
        return

    # region preparation - build inputs for this chunk
    chunk_interpolation_inputs: list[InterpolationInput] = []
    chunk_tensor_structs: list[TensorsStructure] = []
    chunk_options: list[InterpolationOptions] = []

    BackendTensor.initialize_linalg()

    for i in chunk:
        state.stack_structure.stack_number = i
        tensor_struct_i: TensorsStructure = _stack_tensors_structure(state.root_data_descriptor, i)
        interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(
            all_interpolation_input=state.root_interpolation_input,
            stack_structure=state.stack_structure
        )

        fault_input: FaultsData = _grab_stack_fault_data(  # * FAULTS
            _all_stack_values_block=state.all_stack_values_block,
            _interpolation_input_i=interpolation_input_i,
            _stack_structure=state.stack_structure,
            grid_size=interpolation_input_i.grid.len_all_grids
        )
        interpolation_input_i.fault_values = fault_input

        state.interpolation_inputs[i] = interpolation_input_i
        state.tensor_structs[i] = tensor_struct_i
        chunk_interpolation_inputs.append(interpolation_input_i)
        chunk_tensor_structs.append(tensor_struct_i)

        options_i = state.stack_structure.interpolation_options if state.stack_structure.interpolation_options is not None else state.options
        chunk_options.append(_options_with_finite_fault_gradients(options_i, fault_input))
    # endregion

    # Looked up here: context variables do not reach the executor threads below.
    chunk_caches = [stack_cache(state.root_interpolation_input, i, options_key(options_i.kernel_options))
                    for i, options_i in zip(chunk, chunk_options)]

    # Compute weights for this chunk
    chunk_solver_inputs = _compute_weights_for_stacks(
        interpolation_inputs=chunk_interpolation_inputs,
        options=state.options,
        stack_structure=state.stack_structure,
        tensor_structs=chunk_tensor_structs,
        stack_indices=chunk,
        options_per_stack=chunk_options,
        caches=chunk_caches,
    )
    for idx, i in enumerate(chunk):
        state.solver_inputs[i] = chunk_solver_inputs[idx]

    # Evaluate this chunk
    # The fused PyKeOps evaluator cannot concatenate scalar and gradient lazy kernels.
    if (any(item.fault_values.finite_fault_defined for item in chunk_interpolation_inputs)
            or any(micro_evaluation_options(opt, item).compute_scalar_gradient
                   for opt, item in zip(chunk_options, chunk_interpolation_inputs))):
        chunk_eval_inputs, chunk_exported_fields = _evaluate(
            interpolation_inputs=chunk_interpolation_inputs,
            options=state.options,
            solver_inputs=chunk_solver_inputs,
            stack_structure=state.stack_structure,
            tensor_structs=chunk_tensor_structs,
            stack_indices=chunk,
            options_per_stack=chunk_options,
        )
    else:
        # The fused evaluator takes kernel options and evaluation flags from its
        # first stack and lays fault-drift rows out per block, so only stacks that
        # agree on those are fused together. Results keep chunk order.
        chunk_eval_inputs = [None] * len(chunk)
        chunk_exported_fields = [None] * len(chunk)
        for members in _fused_evaluation_groups(chunk_options, chunk_interpolation_inputs):
            group_eval_inputs, group_exported_fields = _evaluate_optimized(
                interpolation_inputs=[chunk_interpolation_inputs[i] for i in members],
                options=state.options,
                solver_inputs=[chunk_solver_inputs[i] for i in members],
                stack_structure=state.stack_structure,
                tensor_structs=[chunk_tensor_structs[i] for i in members],
                stack_indices=[chunk[i] for i in members],
                options_per_stack=[chunk_options[i] for i in members],
            )
            for i, eval_input, exported_fields in zip(members, group_eval_inputs, group_exported_fields):
                chunk_eval_inputs[i] = eval_input
                chunk_exported_fields[i] = exported_fields

    for idx, i in enumerate(chunk):
        state.eval_inputs[i] = chunk_eval_inputs[idx]
        fit_micro_fields(chunk_interpolation_inputs[idx], chunk_exported_fields[idx], chunk_options[idx],
                         chunk_tensor_structs[idx].number_of_points_per_surface,
                         chunk_eval_inputs[idx].xyz_to_interpolate, chunk_options[idx].compute_scalar_gradient,
                         chunk_caches[idx])

    # Segment this chunk
    chunk_outputs = _segment(
        eval_inputs=chunk_eval_inputs,
        exported_fields_per_stack=chunk_exported_fields,
        interpolation_inputs=chunk_interpolation_inputs,
        options=state.options,
        solver_inputs=chunk_solver_inputs,
        stack_structure=state.stack_structure,
        stack_indices=chunk,
        options_per_stack=chunk_options,
    )

    for idx, i in enumerate(chunk):
        state.all_scalar_fields_outputs[i] = chunk_outputs[idx]

    # Update fault values for fault stacks in this chunk so next chunks can use them
    for idx, i in enumerate(chunk):
        if chunk_interpolation_inputs[idx].stack_relation is not StackRelationType.FAULT:
            continue

        values_output = _modify_faults_values_output(
            fault_input=chunk_interpolation_inputs[idx].fault_values,
            output=state.all_scalar_fields_outputs[i],
            xyz_to_interpolate=state.eval_inputs[i].xyz_to_interpolate
        )
        state.all_stack_values_block[i, :] = values_output


def _stack_tensors_structure(descriptor: InputDataDescriptor, stack_index: int) -> TensorsStructure:
    """The stack's ``TensorsStructure``; within a compute built once, as it does not depend on the grid."""
    shared = scope_cache()
    if shared is None:
        return TensorsStructure.from_tensor_structure_subset(descriptor, stack_index)
    key = ("tensors_structure", id(descriptor), stack_index)
    if key not in shared:  # holding the descriptor keeps its id from being reused during the scope
        shared[key] = (descriptor, TensorsStructure.from_tensor_structure_subset(descriptor, stack_index))
    return shared[key][1]


def _segment(
        eval_inputs: list[EvaluatorInput],
        exported_fields_per_stack: list[ExportedFields],
        interpolation_inputs: list[InterpolationInput],
        options: InterpolationOptions,
        solver_inputs,
        stack_structure: StacksStructure,
        stack_indices: list[int],
        options_per_stack: list[InterpolationOptions],
) -> list[ScalarFieldOutput | None]:
    def _run_segment(idx_global_i: tuple[int, int]) -> ScalarFieldOutput:
        idx, global_i = idx_global_i
        # Copy stack_structure to avoid race conditions on stack_number
        from copy import copy
        local_stack_structure = copy(stack_structure)
        local_stack_structure.stack_number = global_i

        options_i = options_per_stack[idx]

        return _on_stack_stream(_segment_stack, idx, global_i, local_stack_structure, options_i)

    stacked_blocks = _stacked_values_blocks(exported_fields_per_stack, interpolation_inputs, solver_inputs,
                                            stack_structure, stack_indices, options_per_stack)

    def _segment_stack(idx, global_i, local_stack_structure, options_i):
        # region segmentation
        values_block = stacked_blocks.get(idx)
        if values_block is None:
            segmentation_input = SegmentationInput(
                unit_values=interpolation_inputs[idx].unit_values,
                sigmoid_slope=(local_stack_structure.segmentation_function(eval_inputs[idx].xyz_to_interpolate)
                               if local_stack_structure.segmentation_function is not None
                               else options_i.sigmoid_slope)
            )
            values_block = activator_interface.activate_formation_block(
                exported_fields=exported_fields_per_stack[idx],
                ids=segmentation_input.unit_values,
                sigmoid_slope=segmentation_input.sigmoid_slope
            )

        # endregion
        output = ScalarFieldOutput(
            weights=_host_weights(solver_inputs[idx]),
            grid=interpolation_inputs[idx].grid,
            exported_fields=exported_fields_per_stack[idx],
            values_block=values_block,
            stack_relation=interpolation_inputs[idx].stack_relation
        )
        return output

    return _map_stacks(_run_segment, stack_indices)


def _stacked_values_blocks(exported_fields_per_stack: list[ExportedFields], interpolation_inputs: list[InterpolationInput],
                           solver_inputs, stack_structure: StacksStructure, stack_indices: list[int],
                           options_per_stack: list[InterpolationOptions]) -> dict:
    """Values blocks of the chunk's stacks segmented together (``soft_segment_stacked``), by chunk position.

    Stacks with the same number of units and field length share one segmentation; the reversed ids and
    temperatures come from the stack cache (they do not depend on the grid). Stacks with a segmentation
    function, an array slope, a fault shell or fewer than two units are left to the per-stack path."""
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH or _stack_execution() != "inline":
        return {}
    groups: dict[tuple, list] = {}
    for idx, global_i in enumerate(stack_indices):
        local_stack_structure = copy(stack_structure)
        local_stack_structure.stack_number = global_i
        fields, ids = exported_fields_per_stack[idx], interpolation_inputs[idx].unit_values
        slope = options_per_stack[idx].sigmoid_slope
        cache = stack_cache_of(solver_inputs[idx])
        if (cache is None or local_stack_structure.segmentation_function is not None
                or not isinstance(slope, numbers.Number) or fields.scalar_field_at_fault_shell is not None
                or len(ids) < 2):
            continue
        field, edges = fields.scalar_field_everywhere, fields.scalar_field_at_surface_points
        if edges.shape != (len(ids) - 1,):
            continue
        constants = cache.segmentation.get(float(slope))
        if constants is None or checking():
            fresh = segmentation_constants(ids, slope)
            if constants is not None:
                assert_same(constants[0], fresh[0], "segmentation ids")
                assert_same(constants[1], fresh[1], "segmentation temperatures")
            constants = cache.segmentation.setdefault(float(slope), fresh)
        groups.setdefault((len(ids), field.shape[0], field.dtype), []).append((idx, field, edges, *constants))

    def rows(values):
        return values[0][None] if len(values) == 1 else BackendTensor.t.stack(values)

    blocks = {}
    for members in groups.values():
        z, edges, ids, tau = (rows([member[k] for member in members]) for k in range(1, 5))
        values = soft_segment_stacked(z, edges, ids, tau)
        for row, member in enumerate(members):
            blocks[member[0]] = values[row][None, :]
    return blocks


def _map_stacks(function, stack_indices):
    """``function((position, stack_index))`` for every stack of a chunk, as ``GEMPY_STACK_EXECUTION`` says:
    ``inline`` (default; in order, in the calling thread, on the current stream) or ``threads`` (a thread
    pool, each stack on its own CUDA stream).

    Threads overlap stacks only while one waits on the device, and the per-stack work is now mostly
    Python issuing small kernels (GIL-bound). A10, same container: Perth Basin interpolation 0.349 s
    with threads vs 0.235 s inline; sprember_micro 0.146 vs 0.135 s."""
    if _stack_execution() == "inline":
        return [function(item) for item in enumerate(stack_indices)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        return list(executor.map(function, enumerate(stack_indices)))


def _on_stack_stream(function, *args):
    import torch
    if _stack_execution() == "inline" or not torch.cuda.is_available():
        return function(*args)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        result = function(*args)
    stream.synchronize()
    return result


def _stack_execution() -> str:
    return os.getenv("GEMPY_STACK_EXECUTION", "inline").lower()


def _host_weights(solver_input):
    cache = stack_cache_of(solver_input)
    if cache is None:
        return BackendTensor.t.to_numpy(solver_input.weights_x0)
    if cache.weights_host is None:
        cache.weights_host = BackendTensor.t.to_numpy(solver_input.weights_x0)
    return cache.weights_host


def _evaluate(interpolation_inputs: list[InterpolationInput], options: InterpolationOptions, solver_inputs, stack_structure: StacksStructure,
              tensor_structs: list[TensorsStructure], stack_indices: list[int] | None = None,
              options_per_stack: list[InterpolationOptions] | None = None) -> tuple[list[EvaluatorInput], list[ExportedFields]]:
    eval_inputs: list[EvaluatorInput] = []
    exported_fields_per_stack: list[ExportedFields] = []
    for idx, global_i in enumerate(stack_indices):
        stack_structure.stack_number = global_i

        eval_input: EvaluatorInput = EvaluatorInput(
            solver_input=solver_inputs[idx],
            interpolation_input=(interpolation_inputs[idx]),
            tensor_struct=tensor_structs[idx],
            only_surface_points=False
        )

        # region evaluate
        exported_fields: ExportedFields = _evaluate_sys_eq(
            eval_input=eval_input,
            weights=eval_input.solver_input.weights_x0,
            options=micro_evaluation_options(options_per_stack[idx] if options_per_stack is not None else options,
                                             interpolation_inputs[idx]),
            grid=interpolation_inputs[idx].grid,
        )

        exported_fields.set_structure_values_from_eval_input(eval_input)
        exported_fields.debug = eval_input.solver_input.debug

        eval_inputs.append(eval_input)
        exported_fields_per_stack.append(exported_fields)
        # endregion
    return eval_inputs, exported_fields_per_stack


def _fused_evaluation_groups(options_per_stack: list[InterpolationOptions],
                             interpolation_inputs: list[InterpolationInput]) -> list[list[int]]:
    """Chunk positions that one fused PyKeOps evaluation can serve exactly.

    The stacked kernel uses its first stack's kernel options and scalar/gradient
    flags for every block, and needs every block to carry the same number of
    fault-drift rows. Groups keep first-appearance order; members keep chunk order.
    """
    groups: dict[tuple, list[int]] = {}
    for idx, (options_i, interpolation_input) in enumerate(zip(options_per_stack, interpolation_inputs)):
        evaluation = micro_evaluation_options(options_i, interpolation_input).evaluation_options
        key = (options_key(options_i.kernel_options), evaluation.compute_scalar, evaluation.compute_scalar_gradient,
               interpolation_input.fault_values.n_faults)
        groups.setdefault(key, []).append(idx)
    return list(groups.values())


def _evaluate_optimized(interpolation_inputs: list[InterpolationInput], options: InterpolationOptions, solver_inputs, stack_structure: StacksStructure,
                          tensor_structs: list[TensorsStructure], stack_indices: list[int] | None = None,
                          options_per_stack: list[InterpolationOptions] | None = None) -> tuple[list[EvaluatorInput], list[ExportedFields]]:
    from gempy_engine.modules.evaluator.symbolic_evaluator import symbolic_evaluator_optimized_stacked

    eval_inputs: list[EvaluatorInput] = []
    for idx, global_i in enumerate(stack_indices):
        stack_structure.stack_number = global_i

        eval_input: EvaluatorInput = EvaluatorInput(
            solver_input=solver_inputs[idx],
            interpolation_input=(interpolation_inputs[idx]),
            tensor_struct=tensor_structs[idx],
            only_surface_points=False
        )
        eval_inputs.append(eval_input)

    # Collect weights per stack
    weights_list = [ei.solver_input.weights_x0 for ei in eval_inputs]

    options_list = options_per_stack or [options] * len(stack_indices)
    reduced_inputs = []
    inverses = []
    for eval_input, interpolation_input, options_i in zip(eval_inputs, interpolation_inputs, options_list):
        reduced, inverse = (_deduplicate_corners(eval_input, interpolation_input.grid)
                            if options_i.evaluation_options.deduplicate_octree_corners else (eval_input, None))
        reduced_inputs.append(reduced)
        inverses.append(inverse)

    # Call the stacked evaluator (single PyKeOps call with block-sparse ranges)
    try:
        exported_fields_list: list[ExportedFields] = symbolic_evaluator_optimized_stacked(
            eval_inputs=reduced_inputs,
            weights_list=weights_list,
            options_list=options_list,
        )
    except Exception as exc:
        try:
            groups = []
            for idx, global_i in enumerate(stack_indices):
                groups.append({
                    'group': idx,
                    'stack_index': global_i,
                    'solver_input_id': hex(id(solver_inputs[idx])),
                    'original_grid_shape': tuple(eval_inputs[idx].xyz_to_interpolate.shape),
                    'reduced_grid_shape': tuple(reduced_inputs[idx].xyz_to_interpolate.shape),
                    'deduplicated': inverses[idx] is not None,
                    'inverse_shape': tuple(inverses[idx].shape) if inverses[idx] is not None else None,
                    'octree_level': options_list[idx].temp_interpolation_values.current_octree_level,
                })
            context = f"groups={groups}"
        except Exception as diagnostic_exc:
            context = f"Group diagnostic collection failed: {diagnostic_exc}"
        raise RuntimeError(f"FLAT chunk stack_indices={stack_indices}; {context}\n{exc}") from exc

    for idx, exported_fields in enumerate(exported_fields_list):
        if inverses[idx] is not None:
            _restore_corner_fields(exported_fields, inverses[idx])
        exported_fields.set_structure_values_from_eval_input(eval_inputs[idx])
        exported_fields.debug = eval_inputs[idx].solver_input.debug

    return eval_inputs, exported_fields_list


def _compute_weights_for_stacks(
        interpolation_inputs: list[InterpolationInput],
        options: InterpolationOptions,
        stack_structure: StacksStructure,
        tensor_structs: list[TensorsStructure],
        stack_indices: list[int] | None = None,
        options_per_stack: list[InterpolationOptions] | None = None,
        caches: list[StackCache | None] | None = None,
) -> list[SolverInput_v2]:
    if stack_indices is None:
        stack_indices = list(range(stack_structure.n_stacks))

    def _run_prep(idx_global_i: tuple[int, int]) -> SolverInput_v2:
        idx, global_i = idx_global_i
        # Copy stack_structure to avoid race conditions on stack_number
        from copy import copy
        local_stack_structure = copy(stack_structure)
        local_stack_structure.stack_number = global_i

        options_i = options_per_stack[idx] if options_per_stack is not None else options

        return _on_stack_stream(_prepare_solver_input, idx, global_i, local_stack_structure, options_i)

    def _preprocess(idx, internals=None):
        return input_preprocess_v2(
            data_shape=tensor_structs[idx],
            interpolation_input=interpolation_inputs[idx],
            cache=caches[idx] if caches is not None else None,
            internals=internals,
        )

    def _options(idx):
        return options_per_stack[idx] if options_per_stack is not None else options

    if caches is not None and _stack_execution() == "inline":
        # All stacks preprocessed first, so their covariances can be built together; the ones built for
        # the first time in this compute from one gather per array for the whole chunk.
        first = [idx for idx, cache in enumerate(caches) if cache is not None and cache.sp_internal is None]
        chunk = (chunk_preprocess([interpolation_inputs[idx].surface_points for idx in first],
                                  [interpolation_inputs[idx].orientations for idx in first],
                                  [tensor_structs[idx] for idx in first])
                 if len(first) > 1 and not BackendTensor.COMPUTE_GRADS else None)
        internals = dict(zip(first, chunk or ()))
        solver_inputs = [_preprocess(idx, internals.get(idx)) for idx in range(len(stack_indices))]
        # The weight-cache fingerprints of the stacks to solve, from one device read for the chunk.
        prefetch_fingerprints([solver_input for solver_input, cache, options_i
                               in zip(solver_inputs, caches, map(_options, range(len(stack_indices))))
                               if cache is not None and (cache.weights is None or checking())
                               and fingerprint_needed(options_i)])
        _prebuild_covariances(solver_inputs, [_options(idx) for idx in range(len(stack_indices))], caches)
        return [_solve_weights(solver_inputs[idx], global_i, _options(idx), caches[idx])
                for idx, global_i in enumerate(stack_indices)]

    def _prepare_solver_input(idx, global_i, local_stack_structure, options_i):
        cache = caches[idx] if caches is not None else None
        return _solve_weights(_preprocess(idx), global_i, options_i, cache)

    return _map_stacks(_run_prep, stack_indices)


def _solve_weights(solver_input: SolverInput_v2, global_i: int, options_i: InterpolationOptions,
                   cache: StackCache | None) -> SolverInput_v2:
    # region compute weights
    # Weights do not depend on the grid: within a compute they are solved (or looked up) once per stack.
    if cache is not None and cache.weights is not None and not checking():
        solver_input.weights_x0, solver_input.debug = cache.weights, cache.debug
        return solver_input
    weights = compute_weights(
        solver_input=solver_input,
        stack_number=global_i,
        options=options_i
    )
    if cache is not None:
        if cache.weights is None:
            cache.weights, cache.debug = weights, solver_input.debug
        else:
            assert_same(cache.weights, weights, "weights")
            weights, solver_input.debug = cache.weights, cache.debug
    solver_input.weights_x0 = weights
    return solver_input


def _prebuild_covariances(solver_inputs: list, options_per_stack: list[InterpolationOptions],
                          caches: list[StackCache | None]) -> None:
    """Build the dense covariances of the stacks that still need a solve together (``stacked_covariances``)
    and leave each in its stack cache, where the dense solve takes it.

    Stacks are grouped by kernel options and number of faults, and a group is split so that no build
    passes ``MAX_STACKED_SIZE`` rows. Single stacks, gradients, debug dumps and the PyKeOps solver keep
    the per-stack build."""
    if (BackendTensor.engine_backend is not AvailableBackends.PYTORCH or BackendTensor.COMPUTE_GRADS
            or config.DEBUG_MODE or pykeops_solver_requested()):
        return
    groups: dict[tuple, list] = {}
    for solver_input, options_i, cache in zip(solver_inputs, options_per_stack, caches):
        kernel_options = options_i.kernel_options
        if (cache is None or (cache.weights is not None and not checking())
                or kernel_options.optimizing_condition_number):
            continue
        key = (options_key(kernel_options), solver_input.fault_internal.n_faults)
        groups.setdefault(key, []).append((solver_input, kernel_options, cache))
    for members in groups.values():
        batch, rows = [], 0
        for member in members:
            size = covariance_size(member[0], member[1])
            if batch and rows + size > MAX_STACKED_SIZE:
                _stacked_build(batch)
                batch, rows = [], 0
            batch.append(member)
            rows += size
        _stacked_build(batch)


def _stacked_build(batch: list) -> None:
    if len(batch) < 2:
        return
    matrices = stacked_covariances([solver_input for solver_input, _, _ in batch], batch[0][1])
    for (_, _, cache), matrix in zip(batch, matrices):
        cache.covariance = matrix


def input_preprocess_v2(data_shape: TensorsStructure, interpolation_input: InterpolationInput,
                        cache: StackCache | None = None, internals: tuple | None = None) -> SolverInput_v2:
    sp_internal, ori_internal, fault_values = preprocess_stack_data(data_shape, interpolation_input, cache, internals)

    solver_input = SolverInput_v2(
        sp_internal=sp_internal,
        ori_internal=ori_internal,
        fault_internal=fault_values
    )
    if cache is not None and cache.weights is not None and not checking():
        solver_input.weights_x0 = cache.weights  # skip uploading the initial guess; the solve is reused
    else:
        solver_input.weights_x0 = BackendTensor.t.array(interpolation_input.weights)
    solver_input.stack_cache = cache

    return solver_input


def _is_external_chunk(stack_structure: StacksStructure, chunk: list[int]) -> bool:
    if stack_structure.interp_functions_per_stack is None:
        return False
    if len(chunk) != 1:
        return False
    return stack_structure.interp_functions_per_stack[chunk[0]] is not None


def _process_external_chunk(state: InterpolationState, chunk: list[int]):
    i = chunk[0]
    state.stack_structure.stack_number = i

    options_i = (state.stack_structure.interpolation_options
                 if state.stack_structure.interpolation_options is not None
                 else state.options)

    interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(
        all_interpolation_input=state.root_interpolation_input,
        stack_structure=state.stack_structure
    )
    output: ScalarFieldOutput = interpolate_feature_with_external_function(
        interpolation_input=interpolation_input_i,
        options=options_i,
        external_interp_funct=state.stack_structure.interp_function,
        external_segment_funct=state.stack_structure.segmentation_function,
    )

    state.all_scalar_fields_outputs[i] = output
    if interpolation_input_i.stack_relation is StackRelationType.FAULT:
        xyz = data_preprocess_interface.prepare_grid(
            interpolation_input_i.grid.values, interpolation_input_i.all_surface_points,
            interpolation_input_i.evaluation_micro_points)
        state.all_stack_values_block[i, :] = _modify_faults_values_output(
            interpolation_input_i.fault_values, output, xyz)
