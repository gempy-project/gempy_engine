import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.kernel_classes.faults import FaultsData


def _grab_stack_fault_data(_all_stack_values_block, _interpolation_input_i, _stack_structure) -> FaultsData:
    fault_data = _interpolation_input_i.fault_values or FaultsData()
    fault_data.fault_values_everywhere = _all_stack_values_block[_stack_structure.active_faults_relations]
    fv_on_all_sp = fault_data.fault_values_everywhere[:, _interpolation_input_i.grid.len_all_grids:]
    fault_data.fault_values_on_sp = fv_on_all_sp[:, _interpolation_input_i.slice_feature]
    return fault_data


def _grab_stack_fault_data_for_input(_all_stack_values_block, _interpolation_input_i, _stack_structure) -> FaultsData:
    fault_data = _interpolation_input_i.fault_values or FaultsData()
    fv_on_all_sp = _all_stack_values_block[_stack_structure.active_faults_relations]
    fault_data.fault_values_on_sp = fv_on_all_sp[:, _interpolation_input_i.slice_feature]
    return fault_data



def _modify_faults_values_output(fault_input: FaultsData, values_on_all_xyz: np.ndarray,
                                 xyz_to_interpolate: np.ndarray) -> np.ndarray:
    val_min = BackendTensor.t.min(values_on_all_xyz, axis=1).reshape(-1, 1)  # ? Is this as good as it gets?
    shifted_vals = (values_on_all_xyz - val_min)  # * Shift values between 0 and 1... hopefully
    if fault_input.finite_faults_defined:
        # TODO: Rescale scalar field parameters
        finite_fault_scalar: np.ndarray = fault_input.finite_fault_data.apply(
            points=xyz_to_interpolate
        )
        fault_scalar_field = shifted_vals * finite_fault_scalar
    else:
        fault_scalar_field = shifted_vals
    return fault_scalar_field
