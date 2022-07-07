import warnings
from typing import List, Iterable, Optional

import numpy as np
from numpy import ndarray

from ...core.data.kernel_classes.faults import FaultsData
from ...core.data.exported_structs import CombinedScalarFieldsOutput
from ...core.data.interp_output import InterpOutput
from ...core.data.scalar_field_output import ScalarFieldOutput
from ...core.data.exported_fields import ExportedFields
from ...core.data.input_data_descriptor import StackRelationType, InputDataDescriptor, TensorsStructure
from ...core.data.interpolation_input import InterpolationInput
from ...core.data.options import InterpolationOptions

from ._interp_single_feature import interpolate_feature


# @profile
def interpolate_all_fields(interpolation_input: InterpolationInput, options: InterpolationOptions,
                           data_descriptor: InputDataDescriptor) -> List[InterpOutput]:
    """Interpolate all scalar fields given a xyz array of points"""

    all_scalar_fields_outputs: List[ScalarFieldOutput] = _interpolate_stack(data_descriptor, interpolation_input, options)
    final_mask_matrix: np.ndarray = _squeeze_mask(all_scalar_fields_outputs, data_descriptor.stack_relation)

    combined_scalar_output: List[CombinedScalarFieldsOutput] = _compute_final_block(
        all_scalar_fields_outputs, final_mask_matrix, options.compute_scalar_gradient)
    all_outputs = []
    for e, _ in enumerate(all_scalar_fields_outputs):
        output: InterpOutput = InterpOutput(all_scalar_fields_outputs[e], combined_scalar_output[e])
        all_outputs.append(output)

    return all_outputs


def _interpolate_stack(root_data_descriptor: InputDataDescriptor, root_interpolation_input: InterpolationInput,
                       options: InterpolationOptions) -> ScalarFieldOutput | List[ScalarFieldOutput]:
    stack_structure = root_data_descriptor.stack_structure

    all_scalar_fields_outputs: List[ScalarFieldOutput | None] = [None] * stack_structure.n_stacks

    xyz_to_interpolate_size: int = root_interpolation_input.grid.len_all_grids + root_interpolation_input.surface_points.n_points
    all_stack_values_block: np.ndarray = np.zeros((stack_structure.n_stacks, xyz_to_interpolate_size))  # Used for faults

    if stack_structure is None:  # ! This branch is just for backward compatibility but we should try to get rid of it as soon as possible
        raise ValueError("Deprecated: stack_structure is not defined in the input data descriptor")

    for i in range(stack_structure.n_stacks):
        stack_structure.stack_number = i

        tensor_struct_i: TensorsStructure = TensorsStructure.from_tensor_structure_subset(root_data_descriptor, i)
        interpolation_input_i: InterpolationInput = InterpolationInput.from_interpolation_input_subset(
            root_interpolation_input, stack_structure)

        # region Set fault input if needed
        fault_relation_on_this_stack: Iterable[bool] = stack_structure.active_faults_relations

        fault_values_all = all_stack_values_block[fault_relation_on_this_stack]
        fv_on_all_sp = fault_values_all[:, interpolation_input_i.grid.len_all_grids:]
        fv_on_sp = fv_on_all_sp[:, interpolation_input_i.slice_feature]
        # Grab Faults data given by the user
        fault_data = interpolation_input_i.fault_values

        if interpolation_input_i.not_fault_input:  # * Set default fault data
            fault_data = FaultsData(fault_values_everywhere=fault_values_all, fault_values_on_sp=fv_on_sp)
        else:  # * Use user given fault data
            fault_data.fault_values_on_sp = fv_on_sp
            fault_data.fault_values_everywhere = fault_values_all

        interpolation_input_i.fault_values = fault_data

        # endregion
        
        output: ScalarFieldOutput = interpolate_feature(
            interpolation_input=interpolation_input_i,
            options=options,
            data_shape=tensor_struct_i,
            external_interp_funct=stack_structure.interp_function,
            external_segment_funct=stack_structure.segmentation_function
        )

        all_scalar_fields_outputs[i] = output

        # This is also for faults!
        if interpolation_input_i.stack_relation is StackRelationType.FAULT:
            val_min = np.min(output.values_on_all_xyz, axis=1).reshape(-1, 1)  # ? Is this as good as it gets?
            shifted_vals = (output.values_on_all_xyz - val_min) * interpolation_input_i.fault_values.offset
            all_stack_values_block[i, :] = shifted_vals

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
                mask_matrix[i, :] = mask_lith
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


def _compute_final_block(all_scalar_fields_outputs: List[ScalarFieldOutput], squeezed_mask_arrays: np.ndarray,
                         compute_scalar_grad: bool = False) -> List[CombinedScalarFieldsOutput]:
    n_scalar_fields = len(all_scalar_fields_outputs)
    squeezed_value_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))
    squeezed_scalar_field_block: ndarray = np.zeros((1, squeezed_mask_arrays.shape[1]))

    def _mask_and_squeeze(block_to_squeeze: np.ndarray, squeezed_mask_array: np.ndarray, previous_block: np.ndarray) -> np.ndarray:
        return (previous_block + block_to_squeeze * squeezed_mask_array).reshape(-1)

    all_combined_scalar_fields = []
    for i in range(n_scalar_fields):
        interp_output: ScalarFieldOutput = all_scalar_fields_outputs[i]
        squeezed_array = squeezed_mask_arrays[i]

        squeezed_value_block = _mask_and_squeeze(interp_output.values_block, squeezed_array, squeezed_value_block)

        scalar_field = interp_output.exported_fields.scalar_field
        squeezed_scalar_field_block = _mask_and_squeeze(scalar_field, squeezed_array, squeezed_scalar_field_block)

        if compute_scalar_grad is True:
            squeezed_gx_block: Optional[ndarray] = np.zeros((1, squeezed_mask_arrays.shape[1]))
            squeezed_gy_block: Optional[ndarray] = np.zeros((1, squeezed_mask_arrays.shape[1]))
            squeezed_gz_block: Optional[ndarray] = np.zeros((1, squeezed_mask_arrays.shape[1]))

            squeezed_gx_block = _mask_and_squeeze(interp_output.exported_fields.gx_field, squeezed_array, squeezed_gx_block)
            squeezed_gy_block = _mask_and_squeeze(interp_output.exported_fields.gy_field, squeezed_array, squeezed_gy_block)
            squeezed_gz_block = _mask_and_squeeze(interp_output.exported_fields.gz_field, squeezed_array, squeezed_gz_block)
        else:
            squeezed_gx_block = None
            squeezed_gy_block = None
            squeezed_gz_block = None

        final_block = squeezed_value_block
        final_exported_fields = ExportedFields(
            _scalar_field=squeezed_scalar_field_block,
            _gx_field=squeezed_gx_block,
            _gy_field=squeezed_gy_block,
            _gz_field=squeezed_gz_block,
            _n_points_per_surface=interp_output.exported_fields._n_points_per_surface,
            _slice_feature=slice(None)
        )

        combined_scalar_fields = CombinedScalarFieldsOutput(
            squeezed_mask_array=squeezed_array,
            final_block=final_block,
            final_exported_fields=final_exported_fields
        )

        all_combined_scalar_fields.append(combined_scalar_fields)

    return all_combined_scalar_fields
