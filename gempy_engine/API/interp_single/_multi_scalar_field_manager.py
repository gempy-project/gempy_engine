from typing import List, Optional

import numpy as np
from numpy import ndarray

from ._aux_faults_ops import _modify_faults_values_output
from ._interp_single_feature import interpolate_feature_with_cokrig, input_preprocess
from ._stack_ops import evaluate, segment
from .compute_weights import compute_weights_for_stacks
from ...core.backend_tensor import BackendTensor
from ...core.data import TensorsStructure
from ...core.data.exported_fields import ExportedFields
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


def _interpolate_stack_flat(root_data_descriptor: InputDataDescriptor, root_interpolation_input: InterpolationInput,
                            options: InterpolationOptions) -> ScalarFieldOutput | List[ScalarFieldOutput]:
    stack_structure = root_data_descriptor.stack_structure

    # region preparation
    interpolation_inputs: list[InterpolationInput] = []
    tensor_structs: list[TensorsStructure] = []

    for i in range(stack_structure.n_stacks):  # TODO: This is the loop we need to split
        tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(root_data_descriptor, i)
        interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(
            all_interpolation_input=root_interpolation_input,
            stack_structure=stack_structure,
            stack_number=i
        )

        interpolation_inputs.append(interpolation_input_i)
        tensor_structs.append(tensor_struct_i)

    # endregion
    solver_inputs = compute_weights_for_stacks(
        interpolation_inputs=interpolation_inputs,
        options=options,
        stack_structure=stack_structure,
        tensor_structs=tensor_structs,
        all_surface_points_size=root_interpolation_input.surface_points.n_points
    )
    eval_inputs, exported_fields_per_stack = evaluate(
        interpolation_inputs=interpolation_inputs,
        options=options,
        solver_inputs=solver_inputs,
        stack_structure=stack_structure,
        tensor_structs=tensor_structs
    )
    all_scalar_fields_outputs = segment(
        eval_inputs=eval_inputs,
        exported_fields_per_stack=exported_fields_per_stack,
        interpolation_inputs=interpolation_inputs,
        options=options,
        solver_inputs=solver_inputs,
        stack_structure=stack_structure
    )

    return all_scalar_fields_outputs


def _interpolate_stack(root_data_descriptor: InputDataDescriptor, root_interpolation_input: InterpolationInput,
                       options: InterpolationOptions) -> ScalarFieldOutput | List[ScalarFieldOutput]:
    # region === Local functions ===
    def _grab_stack_fault_data(_all_stack_values_block, _interpolation_input_i, _stack_structure) -> FaultsData:
        fault_data = _interpolation_input_i.fault_values or FaultsData()

        fault_data.fault_values_everywhere = _all_stack_values_block[_stack_structure.active_faults_relations]

        fv_on_all_sp = fault_data.fault_values_everywhere[:, _interpolation_input_i.grid.len_all_grids:]
        fault_data.fault_values_on_sp = fv_on_all_sp[:, _interpolation_input_i.slice_feature]
        return fault_data

    # endregion

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
            _stack_structure=stack_structure
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


# @off
def _lithology_mask(all_scalar_fields_outputs: List[ScalarFieldOutput], stack_relation: List[StackRelationType]) -> np.ndarray:
    n_scalar_fields = len(all_scalar_fields_outputs)
    grid_size = all_scalar_fields_outputs[0].grid_size
    mask_matrix = BackendTensor.t.zeros((n_scalar_fields, grid_size), dtype=bool)

    onlap_chain_counter = 0
    # Setting the mask matrix
    for i in range(n_scalar_fields):
        onlap_chain_cont: bool = stack_relation[i - 1] in [StackRelationType.ONLAP, StackRelationType.FAULT]
        onlap_chain_began: bool = stack_relation[i - 1 - onlap_chain_counter] is StackRelationType.ONLAP
        onlap_chain_counter: int = (onlap_chain_counter + 1) * onlap_chain_cont * onlap_chain_began

        if onlap_chain_counter:
            mask_matrix[i - 1] = all_scalar_fields_outputs[i].mask_components_erode_components_onlap

            x_mask = mask_matrix[(i - onlap_chain_counter):i, :]
            reversed_x_mask = BackendTensor.t.flip(x_mask, axis=0)
            cumprod_mask = BackendTensor.t.cumprod(reversed_x_mask, axis=0)
            reversed_cumprod_mask = BackendTensor.t.flip(cumprod_mask, axis=0)
            mask_matrix[i - onlap_chain_counter: i] = reversed_cumprod_mask

        # convert to match
        match stack_relation[i]:
            case StackRelationType.ONLAP:
                pass
            case StackRelationType.ERODE:
                mask_lith = all_scalar_fields_outputs[i].mask_components_erode
                mask_matrix[i, :] = mask_lith
            case StackRelationType.FAULT:
                mask_matrix[i, :] = all_scalar_fields_outputs[i].mask_components_fault
            case False | StackRelationType.BASEMENT:
                mask_matrix[i, :] = all_scalar_fields_outputs[i].mask_components_basement
            case _:
                raise ValueError(f"Stack relation {stack_relation[i]} not recognized")

    # Doing the black magic
    final_mask_array = BackendTensor.t.zeros((n_scalar_fields, grid_size), dtype=bool)
    final_mask_array[0] = mask_matrix[-1]
    final_mask_array[1:] = BackendTensor.t.cumprod(BackendTensor.t.invert(mask_matrix[:-1]), axis=0)
    final_mask_array *= mask_matrix

    return final_mask_array


def _faults_mask(all_scalar_fields_outputs: List[ScalarFieldOutput], stack_relation: List[StackRelationType]) -> np.ndarray:
    n_scalar_fields = len(all_scalar_fields_outputs)
    grid_size = all_scalar_fields_outputs[0].grid_size
    mask_matrix = BackendTensor.t.zeros((n_scalar_fields, grid_size), dtype=bool)

    for i in range(len(all_scalar_fields_outputs)):
        match stack_relation[i]:
            case StackRelationType.FAULT:
                mask_matrix[i, :] = all_scalar_fields_outputs[i].mask_components_erode  # * Faults behave as erosion contacts for the fault block
            case _:
                mask_matrix[i, :] = all_scalar_fields_outputs[i].mask_components_fault

    return mask_matrix


def _combine_scalar_fields(all_scalar_fields_outputs: List[ScalarFieldOutput],
                           lithology_mask: np.ndarray,
                           faults_mask: np.ndarray,
                           compute_scalar_grad: bool = False) -> List[CombinedScalarFieldsOutput]:
    n_scalar_fields: int = len(all_scalar_fields_outputs)
    squeezed_value_block: ndarray = BackendTensor.t.zeros((1, lithology_mask.shape[1]))
    squeezed_fault_block: ndarray = BackendTensor.t.zeros((1, lithology_mask.shape[1]))
    squeezed_scalar_field_block: ndarray = BackendTensor.t.zeros((1, lithology_mask.shape[1]))

    def _apply_mask(block_to_squeeze: np.ndarray, squeezed_mask_array: np.ndarray, previous_block: np.ndarray) -> np.ndarray:
        return (previous_block + block_to_squeeze * squeezed_mask_array).reshape(-1)

    all_combined_scalar_fields = []
    for i in range(n_scalar_fields):
        interp_output: ScalarFieldOutput = all_scalar_fields_outputs[i]

        squeezed_value_block = _apply_mask(
            block_to_squeeze=interp_output.values_block,
            squeezed_mask_array=(lithology_mask[i]),
            previous_block=squeezed_value_block
        )

        squeezed_scalar_field_block = _apply_mask(
            block_to_squeeze=interp_output.exported_fields.scalar_field,
            squeezed_mask_array=(lithology_mask[i]),
            previous_block=squeezed_scalar_field_block
        )

        squeezed_fault_block = _apply_mask(
            block_to_squeeze=interp_output.values_block,
            squeezed_mask_array=faults_mask[i],
            previous_block=squeezed_fault_block
        )

        if compute_scalar_grad is True:
            squeezed_gx_block: Optional[ndarray] = BackendTensor.t.zeros((1, lithology_mask.shape[1]))
            squeezed_gy_block: Optional[ndarray] = BackendTensor.t.zeros((1, lithology_mask.shape[1]))
            squeezed_gz_block: Optional[ndarray] = BackendTensor.t.zeros((1, lithology_mask.shape[1]))

            squeezed_gx_block = _apply_mask(interp_output.exported_fields.gx_field, lithology_mask[i], squeezed_gx_block)
            squeezed_gy_block = _apply_mask(interp_output.exported_fields.gy_field, lithology_mask[i], squeezed_gy_block)
            squeezed_gz_block = _apply_mask(interp_output.exported_fields.gz_field, lithology_mask[i], squeezed_gz_block)
        else:
            squeezed_gx_block = None
            squeezed_gy_block = None
            squeezed_gz_block = None

        final_exported_fields = ExportedFields(
            _scalar_field=squeezed_scalar_field_block,
            _gx_field=squeezed_gx_block,
            _gy_field=squeezed_gy_block,
            _gz_field=squeezed_gz_block,
            _n_points_per_surface=interp_output.exported_fields._n_points_per_surface,
            _slice_feature=slice(None)
        )

        combined_scalar_fields = CombinedScalarFieldsOutput(
            squeezed_mask_array=(lithology_mask[i]),
            final_block=squeezed_value_block,
            faults_block=squeezed_fault_block,
            final_exported_fields=final_exported_fields
        )

        all_combined_scalar_fields.append(combined_scalar_fields)

    return all_combined_scalar_fields
