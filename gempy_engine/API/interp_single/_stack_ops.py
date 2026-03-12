from typing import List

import numpy as np

from ._aux_faults_ops import _grab_stack_fault_data, _modify_faults_values_output
from ._interp_scalar_field import _evaluate_sys_eq, compute_weights
from ._interp_single_feature import scalar_field_segmentation_v2, input_preprocess_v2
from ...core.data import TensorsStructure
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


@dataclass
class _InterpolationState:
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


def _process_chunk(state: _InterpolationState, chunk: list[int]):
    # region preparation - build inputs for this chunk
    chunk_interpolation_inputs: list[InterpolationInput] = []
    chunk_tensor_structs: list[TensorsStructure] = []

    for i in chunk:
        state.stack_structure.stack_number = i
        tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(state.root_data_descriptor, i)
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
    # endregion

    # Compute weights for this chunk
    chunk_solver_inputs = compute_weights_for_stacks(
        interpolation_inputs=chunk_interpolation_inputs,
        options=state.options,
        stack_structure=state.stack_structure,
        tensor_structs=chunk_tensor_structs,
        stack_indices=chunk
    )
    for idx, i in enumerate(chunk):
        state.solver_inputs[i] = chunk_solver_inputs[idx]

    # Evaluate this chunk
    chunk_eval_inputs, chunk_exported_fields = evaluate(
        interpolation_inputs=chunk_interpolation_inputs,
        options=state.options,
        solver_inputs=chunk_solver_inputs,
        stack_structure=state.stack_structure,
        tensor_structs=chunk_tensor_structs,
        stack_indices=chunk
    )

    for idx, i in enumerate(chunk):
        state.eval_inputs[i] = chunk_eval_inputs[idx]

    # Segment this chunk
    chunk_outputs = segment(
        eval_inputs=chunk_eval_inputs,
        exported_fields_per_stack=chunk_exported_fields,
        interpolation_inputs=chunk_interpolation_inputs,
        options=state.options,
        solver_inputs=chunk_solver_inputs,
        stack_structure=state.stack_structure,
        stack_indices=chunk
    )

    for idx, i in enumerate(chunk):
        state.all_scalar_fields_outputs[i] = chunk_outputs[idx]

    # Update fault values for fault stacks in this chunk so next chunks can use them
    for idx, i in enumerate(chunk):
        if chunk_interpolation_inputs[idx].stack_relation is StackRelationType.FAULT:
            values_output = _modify_faults_values_output(
                fault_input=chunk_interpolation_inputs[idx].fault_values,
                values_on_all_xyz=state.all_scalar_fields_outputs[i].values_on_all_xyz,
                xyz_to_interpolate=state.eval_inputs[i].xyz_to_interpolate
            )
            state.all_stack_values_block[i, :] = values_output


def segment(eval_inputs: list[EvaluatorInput], exported_fields_per_stack: list[ExportedFields], interpolation_inputs: list[InterpolationInput], options: InterpolationOptions, solver_inputs, stack_structure: StacksStructure,
            stack_indices: list[int] | None = None) -> list[ScalarFieldOutput | None]:
    if stack_indices is None:
        stack_indices = list(range(stack_structure.n_stacks))
    all_scalar_fields_outputs: List[ScalarFieldOutput | None] = [None] * len(stack_indices)
    for idx, global_i in enumerate(stack_indices):
        stack_structure.stack_number = global_i
        # region segmentation
        values_block = scalar_field_segmentation_v2(
            exported_fields=exported_fields_per_stack[idx],
            segmentation_input=SegmentationInput(
                unit_values=interpolation_inputs[idx].unit_values,
                sigmoid_slope=(stack_structure.segmentation_function(eval_inputs[idx].xyz_to_interpolate) if stack_structure.segmentation_function is not None
                               else options.sigmoid_slope
                               )
            )
        )
        # endregion
        output = ScalarFieldOutput(
            weights=solver_inputs[idx].weights_x0,
            grid=interpolation_inputs[idx].grid,
            exported_fields=exported_fields_per_stack[idx],
            values_block=values_block,
            stack_relation=interpolation_inputs[idx].stack_relation
        )

        # @on
        all_scalar_fields_outputs[idx] = output
    return all_scalar_fields_outputs


def evaluate(interpolation_inputs: list[InterpolationInput], options: InterpolationOptions, solver_inputs, stack_structure: StacksStructure,
             tensor_structs: list[TensorsStructure], stack_indices: list[int] | None = None) -> tuple[list[EvaluatorInput], list[ExportedFields]]:
    if stack_indices is None:
        stack_indices = list(range(stack_structure.n_stacks))
    eval_inputs: list[EvaluatorInput] = []
    exported_fields_per_stack: list[ExportedFields] = []
    for idx, global_i in enumerate(stack_indices):
        stack_structure.stack_number = global_i

        input_to_evaluate = interpolation_inputs[idx]

        eval_input: EvaluatorInput = EvaluatorInput(
            solver_input=solver_inputs[idx],
            interpolation_input=input_to_evaluate,
            tensor_struct=tensor_structs[idx],
            only_surface_points=False
        )

        eval_inputs.append(eval_input)

        # region evaluate
        exported_fields = _construct_experted_fields(eval_input, options)

        exported_fields_per_stack.append(exported_fields)
        # endregion
    return eval_inputs, exported_fields_per_stack


def _construct_experted_fields(eval_input: EvaluatorInput, options: InterpolationOptions) -> ExportedFields:
    exported_fields: ExportedFields = _evaluate_sys_eq(
        eval_input=eval_input,
        weights=eval_input.solver_input.weights_x0,
        options=options
    )

    exported_fields.set_structure_values_from_eval_input(eval_input)
    exported_fields.debug = eval_input.solver_input.debug
    return exported_fields


def compute_weights_for_stacks(interpolation_inputs: list[InterpolationInput], options: InterpolationOptions, stack_structure: StacksStructure,
                               tensor_structs: list[TensorsStructure], stack_indices: list[int] | None = None) -> list[SolverInput_v2]:
    solver_inputs: list[SolverInput_v2] = []

    if stack_indices is None:
        stack_indices = list(range(stack_structure.n_stacks))

    for idx, global_i in enumerate(stack_indices):
        stack_structure.stack_number = global_i
        solver_input: SolverInput_v2 = input_preprocess_v2(
            data_shape=tensor_structs[idx],
            interpolation_input=interpolation_inputs[idx]
        )
        solver_inputs.append(solver_input)

        # region compute weights
        # TODO: Adding external function
        weights = compute_weights(
            solver_input=solver_input,
            stack_number=global_i,
            options=options
        )
        solver_input.weights_x0 = weights

        # endregion

    return solver_inputs


def _compute_independent_chunks(faults_relations: np.ndarray, n_stacks: int, masking_descriptor: list) -> list[list[int]]:
    """Analyze the fault_relations matrix to find chunks of independent stacks.
    
    faults_relations[j, i] == True means stack i depends on fault stack j.
    Stacks are processed in chunks where all stacks in a chunk have their
    fault dependencies already resolved by previous chunks.
    """
    if faults_relations is None:
        return [list(range(n_stacks))]

    fault_stacks = {i for i in range(n_stacks) if masking_descriptor[i] is StackRelationType.FAULT}

    chunks: list[list[int]] = []
    resolved: set[int] = set()
    remaining = set(range(n_stacks))

    while remaining:
        chunk = []
        for i in sorted(remaining):
            deps = {j for j in range(n_stacks) if faults_relations[j, i] and j in fault_stacks}
            if deps.issubset(resolved):
                chunk.append(i)

        if not chunk:
            raise RuntimeError("Circular fault dependency detected")

        chunks.append(chunk)
        remaining -= set(chunk)
        resolved.update(i for i in chunk if i in fault_stacks)

    return chunks
