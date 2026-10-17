import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.config import include_raw_scalar_fields
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.scalar_field_output import ScalarFieldOutput
from gempy_engine.modules.faults.finite_faults import project_points_onto_surface


def _grab_stack_fault_data(_all_stack_values_block, _interpolation_input_i, _stack_structure, grid_size:int) -> FaultsData:
    fault_data = _interpolation_input_i.fault_values or FaultsData()
    fault_data.fault_values_everywhere = _all_stack_values_block[_stack_structure.active_faults_relations]
    fv_on_all_sp = fault_data.fault_values_everywhere[:, grid_size:_interpolation_input_i.macro_reference_size]
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
    reference_size = output.exported_fields._macro_reference_size or len(xyz_to_interpolate)
    val_min = BackendTensor.t.min(values_on_all_xyz[:, :reference_size], axis=1).reshape(-1, 1)
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
    target_values = np.asarray(to_numpy(output.scalar_field_at_sp)).reshape(-1)
    if target_values.size != 1:
        raise ValueError("Finite-fault projection requires exactly one fault surface isovalue")

    finite_fault_scalar_np, _ = finite_fault_taper(
        finite_fault=fault_input.finite_fault,
        points=np.asarray(to_numpy(xyz_to_interpolate)),
        scalar_field_values=np.asarray(to_numpy(exported_fields.scalar_field_everywhere)),
        gradient_fields=tuple(np.asarray(to_numpy(gradient)) for gradient in gradients),
        target_scalar_value=float(target_values[0]),
        frame_indices=finite_fault_frame_indices(exported_fields, reference_size),
    )
    if isinstance(shifted_vals, np.ndarray):
        finite_fault_scalar = finite_fault_scalar_np.astype(shifted_vals.dtype, copy=False)
    else:
        finite_fault_scalar = BackendTensor.t.array(
            finite_fault_scalar_np,
            dtype=shifted_vals.dtype,
        )
    if include_raw_scalar_fields():
        output.finite_fault_scalar = finite_fault_scalar
    return shifted_vals * finite_fault_scalar


def finite_fault_frame_indices(exported_fields, reference_size: int) -> np.ndarray:
    """Indices of the points that may set a finite fault's slip frame.

    These are the fault's own surface points: they do not depend on the grid,
    so every octree level picks the same frame. Without surface points in the
    evaluated points, the whole reference set (grid included) is used.
    """
    grid_size = exported_fields._grid_size
    if grid_size is not None and grid_size < reference_size:
        sp_indices = np.arange(grid_size, reference_size)[exported_fields.slice_feature or slice(None)]
        if len(sp_indices):
            return sp_indices
    return np.arange(reference_size)


def finite_fault_taper(finite_fault, points, scalar_field_values, gradient_fields, target_scalar_value,
                       frame_indices=None, normal=None):
    """Slip multiplier of ``points`` (host NumPy) and the slip-frame normal used.

    Points are projected onto the fault surface with their own scalar and
    gradient. Without ``normal`` the frame normal is the gradient at the
    ``frame_indices`` point (default: all points) whose projection lies
    nearest the fault center.
    """
    projected_points = project_points_onto_surface(
        points=points,
        scalar_field_values=scalar_field_values,
        gradient_fields=gradient_fields,
        target_scalar_value=target_scalar_value,
    )
    if normal is None:
        frame_indices = np.arange(len(points)) if frame_indices is None else np.asarray(frame_indices)
        gradient_matrix = np.stack(gradient_fields, axis=-1)[frame_indices]
        valid_gradient = np.linalg.norm(gradient_matrix, axis=1) > 1e-12
        if not np.any(valid_gradient):
            raise ValueError("Cannot determine finite-fault frame from near-zero gradients")

        center = np.asarray(finite_fault.center)
        distances_to_center = np.linalg.norm(projected_points[frame_indices] - center, axis=1)
        center_index = np.argmin(np.where(valid_gradient, distances_to_center, np.inf))
        normal = gradient_matrix[center_index]
    return finite_fault.calculate_slip(points=projected_points, normal=normal), normal
