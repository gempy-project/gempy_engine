from typing import List

import numpy as np

from ._aux_faults_ops import _modify_faults_values_output, _grab_stack_fault_data
from ._interp_single_feature import interpolate_feature_with_cokrig, input_preprocess
from ._making_ops import _lithology_mask, _faults_mask, _combine_scalar_fields
from ._stack_ops import evaluate, segment
from .compute_weights import compute_weights_for_stacks
from ...core.backend_tensor import BackendTensor
from ...core.data import TensorsStructure
from ...core.data.exported_structs import CombinedScalarFieldsOutput
from ...core.data.input_data_descriptor import InputDataDescriptor
from ...core.data.internal_structs import SolverInput
from ...core.data.interp_output import InterpOutput
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.options import InterpolationOptions
from ...core.data.scalar_field_output import ScalarFieldOutput
from ...core.data.stack_relation_type import StackRelationType


# @off
# @profile
def interpolate_all_fields(interpolation_input: InterpolationInput, options: InterpolationOptions,
                           data_descriptor: InputDataDescriptor) -> List[InterpOutput]:
    """Interpolate all scalar fields given a xyz array of points"""

    if False:
        all_scalar_fields_outputs: List[ScalarFieldOutput] = _interpolate_stack(data_descriptor, interpolation_input, options)
    else:
        all_scalar_fields_outputs: List[ScalarFieldOutput] = _interpolate_stack_flat(data_descriptor, interpolation_input, options)

    combined_scalar_output: List[CombinedScalarFieldsOutput] = _combine_scalar_fields(
        all_scalar_fields_outputs=all_scalar_fields_outputs,
        lithology_mask=_lithology_mask(all_scalar_fields_outputs, data_descriptor.stack_relation),
        faults_mask=_faults_mask(all_scalar_fields_outputs, data_descriptor.stack_relation),
        compute_scalar_grad=options.compute_scalar_gradient
    )

    all_outputs = []
    for e, _ in enumerate(all_scalar_fields_outputs):
        output: InterpOutput = InterpOutput(all_scalar_fields_outputs[e], combined_scalar_output[e])
        all_outputs.append(output)
    return all_outputs


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


def _interpolate_stack_flat(root_data_descriptor: InputDataDescriptor, root_interpolation_input: InterpolationInput,
                            options: InterpolationOptions) -> ScalarFieldOutput | List[ScalarFieldOutput]:
    stack_structure = root_data_descriptor.stack_structure

    xyz_to_interpolate_size: int = root_interpolation_input.grid.len_all_grids + root_interpolation_input.surface_points.n_points
    all_stack_values_block: np.ndarray = BackendTensor.t.zeros(
        (stack_structure.n_stacks, xyz_to_interpolate_size),
        dtype=BackendTensor.dtype_obj)  # * Used for faults

    # Compute independent chunks from fault relations
    chunks: list[list[int]] = _compute_independent_chunks(
        faults_relations=stack_structure.faults_relations,
        n_stacks=stack_structure.n_stacks,
        masking_descriptor=stack_structure.masking_descriptor
    )

    # Pre-allocate result lists indexed by global stack number
    interpolation_inputs: list[InterpolationInput | None] = [None] * stack_structure.n_stacks
    tensor_structs: list[TensorsStructure | None] = [None] * stack_structure.n_stacks
    solver_inputs: list = [None] * stack_structure.n_stacks
    eval_inputs: list = [None] * stack_structure.n_stacks
    all_scalar_fields_outputs: list[ScalarFieldOutput | None] = [None] * stack_structure.n_stacks

    for chunk in chunks:
        # region preparation - build inputs for this chunk
        chunk_interpolation_inputs: list[InterpolationInput] = []
        chunk_tensor_structs: list[TensorsStructure] = []

        for i in chunk:
            stack_structure.stack_number = i
            tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(root_data_descriptor, i)
            interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(
                all_interpolation_input=root_interpolation_input,
                stack_structure=stack_structure
            )

            fault_input: FaultsData = _grab_stack_fault_data(  # * FAULTS
                _all_stack_values_block=all_stack_values_block,
                _interpolation_input_i=interpolation_input_i,
                _stack_structure=stack_structure,
                grid_size=interpolation_input_i.grid.len_all_grids
            )
            interpolation_input_i.fault_values = fault_input

            interpolation_inputs[i] = interpolation_input_i
            tensor_structs[i] = tensor_struct_i
            chunk_interpolation_inputs.append(interpolation_input_i)
            chunk_tensor_structs.append(tensor_struct_i)
        # endregion

        # Compute weights for this chunk
        chunk_solver_inputs = compute_weights_for_stacks(
            interpolation_inputs=chunk_interpolation_inputs,
            options=options,
            stack_structure=stack_structure,
            tensor_structs=chunk_tensor_structs,
            all_surface_points_size=root_interpolation_input.surface_points.n_points,
            stack_indices=chunk
        )
        for idx, i in enumerate(chunk):
            solver_inputs[i] = chunk_solver_inputs[idx]

        # Evaluate this chunk
        chunk_eval_inputs, chunk_exported_fields = evaluate(
            interpolation_inputs=chunk_interpolation_inputs,
            options=options,
            solver_inputs=chunk_solver_inputs,
            stack_structure=stack_structure,
            tensor_structs=chunk_tensor_structs,
            xyz_to_interpolate_size=xyz_to_interpolate_size,
            stack_indices=chunk
        )

        for idx, i in enumerate(chunk):
            eval_inputs[i] = chunk_eval_inputs[idx]

        # Segment this chunk
        chunk_outputs = segment(
            eval_inputs=chunk_eval_inputs,
            exported_fields_per_stack=chunk_exported_fields,
            interpolation_inputs=chunk_interpolation_inputs,
            options=options,
            solver_inputs=chunk_solver_inputs,
            stack_structure=stack_structure,
            stack_indices=chunk
        )

        for idx, i in enumerate(chunk):
            all_scalar_fields_outputs[i] = chunk_outputs[idx]

        # Update fault values for fault stacks in this chunk so next chunks can use them
        for idx, i in enumerate(chunk):
            if chunk_interpolation_inputs[idx].stack_relation is StackRelationType.FAULT:
                values_output = _modify_faults_values_output(
                    fault_input=chunk_interpolation_inputs[idx].fault_values,
                    values_on_all_xyz=all_scalar_fields_outputs[i].values_on_all_xyz,
                    xyz_to_interpolate=eval_inputs[i].xyz_to_interpolate
                )
                all_stack_values_block[i, :] = values_output

    return all_scalar_fields_outputs


def _interpolate_stack(root_data_descriptor: InputDataDescriptor, root_interpolation_input: InterpolationInput,
                       options: InterpolationOptions) -> ScalarFieldOutput | List[ScalarFieldOutput]:
    stack_structure = root_data_descriptor.stack_structure

    all_scalar_fields_outputs: List[ScalarFieldOutput | None] = [None] * stack_structure.n_stacks

    xyz_to_interpolate_size: int = root_interpolation_input.grid.len_all_grids + root_interpolation_input.surface_points.n_points
    all_stack_values_block: np.ndarray = BackendTensor.t.zeros(
        (stack_structure.n_stacks, xyz_to_interpolate_size),
        dtype=BackendTensor.dtype_obj)  # * Used for faults

    for i in range(stack_structure.n_stacks):  # TODO: This is the loop we need to split
        stack_structure.stack_number = i

        tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(root_data_descriptor, i)
        interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(
            all_interpolation_input=root_interpolation_input,
            stack_structure=stack_structure
        )

        fault_input: FaultsData = _grab_stack_fault_data(  # * FAULTS
            _all_stack_values_block=all_stack_values_block,
            _interpolation_input_i=interpolation_input_i,
            _stack_structure=stack_structure,
            grid_size=interpolation_input_i.grid.len_all_grids
        )
        interpolation_input_i.fault_values = fault_input

        solver_input: SolverInput = input_preprocess(tensor_struct_i, interpolation_input_i)
        # TODO: Add external function!
        output: ScalarFieldOutput = interpolate_feature_with_cokrig(
            interpolation_input=interpolation_input_i,
            options=options,
            data_shape=tensor_struct_i,
            solver_input=solver_input,
            external_segment_funct=stack_structure.segmentation_function,
            stack_number=i
        )

        # @on
        all_scalar_fields_outputs[i] = output

        # # * Modify the values for Fault stacks
        if interpolation_input_i.stack_relation is StackRelationType.FAULT:  # * This is also for faults!
            values_output = _modify_faults_values_output(  # ! This is all_STACK_values_block (not all_scalar_fields_outputs)
                fault_input=fault_input,
                values_on_all_xyz=output.values_on_all_xyz,
                xyz_to_interpolate=solver_input.xyz_to_interpolate
            )
            all_stack_values_block[i, :] = values_output

    return all_scalar_fields_outputs
