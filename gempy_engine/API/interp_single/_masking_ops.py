from typing import List, Optional

import numpy as np
from numpy import ndarray

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.exported_fields import ExportedFields
from gempy_engine.core.data.exported_structs import CombinedScalarFieldsOutput
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.core.data.stack_relation_type import StackRelationType


def combine_scalar_fields(all_scalar_fields_outputs: List[ScalarFieldOutput],
                          data_descriptor: InputDataDescriptor,
                          compute_scalar_grad: bool = False) -> List[CombinedScalarFieldsOutput]:
    return _combine_scalar_fields(
        all_scalar_fields_outputs=all_scalar_fields_outputs,
        lithology_mask=_lithology_mask(all_scalar_fields_outputs, data_descriptor.stack_relation),
        faults_mask=_faults_mask(all_scalar_fields_outputs, data_descriptor.stack_relation),
        compute_scalar_grad=compute_scalar_grad
    )


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
