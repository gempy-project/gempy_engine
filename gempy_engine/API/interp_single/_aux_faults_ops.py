import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.modules.faults.finite_faults import project_points_onto_surface


def _grab_stack_fault_data(_all_stack_values_block, _interpolation_input_i, _stack_structure, grid_size:int) -> FaultsData:
    fault_data = _interpolation_input_i.fault_values or FaultsData()
    fault_data.fault_values_everywhere = _all_stack_values_block[_stack_structure.active_faults_relations]
    fv_on_all_sp = fault_data.fault_values_everywhere[:, grid_size:]
    fault_data.fault_values_on_sp = fv_on_all_sp[:, _interpolation_input_i.slice_feature]
    return fault_data



def _options_with_finite_fault_gradients(
        options: InterpolationOptions,
        fault_input: FaultsData | None,
) -> InterpolationOptions:
    if fault_input is None or not fault_input.finite_fault_defined or options.evaluation_options.compute_scalar_gradient:
        return options

    options_copy = options.model_copy(deep=True)
    options_copy.evaluation_options.compute_scalar_gradient = True
    return options_copy


def _modify_faults_values_output(
        fault_input: FaultsData,
        output: ScalarFieldOutput,
        xyz_to_interpolate: np.ndarray,
) -> np.ndarray:
    values_on_all_xyz = output.values_on_all_xyz
    val_min = BackendTensor.t.min(values_on_all_xyz, axis=1).reshape(-1, 1)  # ? Is this as good as it gets?
    shifted_vals = (values_on_all_xyz - val_min)  # * Shift values between 0 and 1... hopefully
    if not fault_input.finite_fault_defined:
        return shifted_vals

    exported_fields = output.exported_fields
    gradients = (
        exported_fields.gx_field_everywhere,
        exported_fields.gy_field_everywhere,
        exported_fields.gz_field_everywhere,
    )
    if any(gradient is None for gradient in gradients):
        raise ValueError("Finite-fault projection requires scalar gradients")

    to_numpy = BackendTensor.t.to_numpy
    xyz_np = np.asarray(to_numpy(xyz_to_interpolate))
    scalar_field_np = np.asarray(to_numpy(exported_fields.scalar_field_everywhere))
    gradients_np = tuple(np.asarray(to_numpy(gradient)) for gradient in gradients)
    target_values = np.asarray(to_numpy(output.scalar_field_at_sp)).reshape(-1)
    if target_values.size != 1:
        raise ValueError("Finite-fault projection requires exactly one fault surface isovalue")

    projected_points = project_points_onto_surface(
        points=xyz_np,
        scalar_field_values=scalar_field_np,
        gradient_fields=gradients_np,
        target_scalar_value=float(target_values[0]),
    )

    gradient_matrix = np.stack(gradients_np, axis=-1)
    valid_gradient = np.linalg.norm(gradient_matrix, axis=1) > 1e-12
    if not np.any(valid_gradient):
        raise ValueError("Cannot determine finite-fault frame from near-zero gradients")

    center = np.asarray(fault_input.finite_fault.center)
    distances_to_center = np.linalg.norm(projected_points - center, axis=1)
    center_index = np.argmin(np.where(valid_gradient, distances_to_center, np.inf))
    finite_fault_scalar_np = fault_input.finite_fault.calculate_slip(
        points=projected_points,
        normal=gradient_matrix[center_index],
    )
    finite_fault_scalar = BackendTensor.t.array(
        finite_fault_scalar_np,
        dtype=shifted_vals.dtype,
    )
    return shifted_vals * finite_fault_scalar
