import warnings
from typing import List

import numpy as np
from numpy import ndarray

from ...core.data.exported_structs import InterpOutput, ExportedFields, ScalarFieldOutput, CombinedScalarFieldsOutput
from ...core.data.input_data_descriptor import StackRelationType, InputDataDescriptor, TensorsStructure
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.options import KernelOptions

from ._interp_single_feature import interpolate_feature


def interpolate_all_fields(interpolation_input: InterpolationInput, options: KernelOptions,
                           data_descriptor: InputDataDescriptor) -> List[InterpOutput]:
    """Interpolate all scalar fields given a xyz array of points"""

    all_scalar_fields_outputs: List[ScalarFieldOutput] = _interpolate_stack(data_descriptor, interpolation_input, options)
    final_mask_matrix: np.ndarray = _squeeze_mask(all_scalar_fields_outputs, data_descriptor.stack_relation)

    combined_scalar_output: List[CombinedScalarFieldsOutput] = _compute_final_block(all_scalar_fields_outputs, final_mask_matrix)
    all_outputs = []
    for e, _ in enumerate(all_scalar_fields_outputs):
        output: InterpOutput = InterpOutput(all_scalar_fields_outputs[e], combined_scalar_output[e])
        all_outputs.append(output)

    return all_outputs


def _interpolate_stack(root_data_descriptor: InputDataDescriptor, interpolation_input: InterpolationInput,
                       options: KernelOptions) -> ScalarFieldOutput | List[ScalarFieldOutput]:
    all_scalar_fields_outputs: List[ScalarFieldOutput] = []

    stack_structure = root_data_descriptor.stack_structure

    if stack_structure is None:  # ! This branch is just for backward compatibility but we should try to get rid of it as soon as possible
        warnings.warn("Deprecated: stack_structure is not defined in the input data descriptor", DeprecationWarning) 
        output = interpolate_feature(interpolation_input, options, root_data_descriptor.tensors_structure)
        all_scalar_fields_outputs.append(output)
        return all_scalar_fields_outputs
    else:
        for i in range(stack_structure.n_stacks):
            stack_structure.stack_number = i

            tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(root_data_descriptor, i)
            interpolation_input_i = InterpolationInput.from_interpolation_input_subset(interpolation_input, stack_structure)
            
            output: ScalarFieldOutput = interpolate_feature(interpolation_input_i, options, tensor_struct_i,
                                                            stack_structure.interp_function)
        
            all_scalar_fields_outputs.append(output)

    return all_scalar_fields_outputs


def _squeeze_mask(all_scalar_fields_outputs: List[ScalarFieldOutput], stack_relation: List[StackRelationType]) -> np.ndarray:
    n_scalar_fields = len(all_scalar_fields_outputs)
    grid_size = all_scalar_fields_outputs[0].grid_size
    mask_matrix = np.zeros((n_scalar_fields, grid_size), dtype=np.bool)

    # Setting the mask matrix
    for i in range(n_scalar_fields):
        mask_lith = all_scalar_fields_outputs[i].mask_components.mask_lith
        match stack_relation[i]:
            case StackRelationType.ERODE:
                mask_matrix[i, :] = mask_lith
            case StackRelationType.ONLAP:
                pass
            case StackRelationType.FAULT:
                pass
            case False:
                mask_matrix[i, :] = mask_lith
            case _:
                raise ValueError(f"Unknown stack relation type: {stack_relation[i]}")

    # Doing the black magic
    final_mask_array = np.zeros((n_scalar_fields, grid_size), dtype=bool)
    final_mask_array[0] = mask_matrix[-1]
    final_mask_array[1:] = np.cumprod(np.invert(mask_matrix[:-1]), axis=0)
    final_mask_array *= mask_matrix

    return final_mask_array


def _compute_final_block(all_scalar_fields_outputs: List[ScalarFieldOutput], squeezed_mask_arrays: np.ndarray) -> List[CombinedScalarFieldsOutput]:
    n_scalar_fields = len(all_scalar_fields_outputs)
    squeezed_value_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))
    squeezed_scalar_field_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))
    squeezed_gx_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))
    squeezed_gy_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))
    squeezed_gz_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))

    def _mask_and_squeeze(block_to_squeeze: np.ndarray, squeezed_mask_array: np.ndarray, previous_block: np.ndarray) -> np.ndarray:
        return (previous_block + block_to_squeeze * squeezed_mask_array).reshape(-1)

    all_combined_scalar_fields = []
    for i in range(n_scalar_fields):
        interp_output: ScalarFieldOutput = all_scalar_fields_outputs[i]
        squeezed_array = squeezed_mask_arrays[i]

        squeezed_value_block = _mask_and_squeeze(interp_output.values_block, squeezed_array, squeezed_value_block)

        scalar_field = interp_output.exported_fields.scalar_field
        squeezed_scalar_field_block = _mask_and_squeeze(scalar_field, squeezed_array, squeezed_scalar_field_block)

        squeezed_gx_block = _mask_and_squeeze(interp_output.exported_fields.gx_field, squeezed_array, squeezed_gx_block)
        squeezed_gy_block = _mask_and_squeeze(interp_output.exported_fields.gy_field, squeezed_array, squeezed_gy_block)
        squeezed_gz_block = _mask_and_squeeze(interp_output.exported_fields.gz_field, squeezed_array, squeezed_gz_block)

        final_block = squeezed_value_block
        final_exported_fields = ExportedFields(
            _scalar_field=squeezed_scalar_field_block,
            _gx_field=squeezed_gx_block,
            _gy_field=squeezed_gy_block,
            _gz_field=squeezed_gz_block,
            n_points_per_surface=interp_output.exported_fields.n_points_per_surface,
            n_surface_points=None
        )

        combined_scalar_fields = CombinedScalarFieldsOutput(
            squeezed_mask_array=squeezed_array,
            final_block=final_block,
            final_exported_fields=final_exported_fields
        )

        all_combined_scalar_fields.append(combined_scalar_fields)

    return all_combined_scalar_fields
