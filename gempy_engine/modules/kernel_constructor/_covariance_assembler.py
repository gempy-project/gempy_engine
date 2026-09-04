import numpy as np
from gempy_engine.config import AvailableBackends

import gempy_engine.config
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.options import KernelOptions, NuggetImplementation
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.modules.kernel_constructor import _structs
from gempy_engine.modules.kernel_constructor._structs import KernelInput, SurfacePointNuggets
from gempy_engine.modules.kernel_constructor.execution_mode import KernelExecutionMode

# ! Important for loading the pickle in test_distance_matrix
from gempy_engine.modules.kernel_constructor._internalDistancesMatrices import InternalDistancesMatrices

global_nugget = 1e-5


def get_covariance(
        c_o,
        dm,
        k_a,
        k_p_ref,
        k_p_rest,
        k_ref_ref,
        k_ref_rest,
        k_rest_ref,
        k_rest_rest,
        ki: KernelInput,
        options,
        execution_mode: KernelExecutionMode = KernelExecutionMode.DENSE,
):
    cov_grad = _get_cov_grad(
        dm,
        k_a,
        k_p_ref,
        ki.nugget_grad,
        options.nugget_implementation,
        execution_mode,
    )
    cov_sp = _get_cov_surface_points(dm, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest,
                                     options, ki.nugget_scalar, ki.nugget_grad.shape[0], execution_mode)
    cov_grad_sp = _get_cross_cov_grad_sp(dm, k_p_ref, k_p_rest, options)  # C
    
    # Universal drift
    usp = _get_universal_sp_terms(ki, options)
    ug = _get_universal_gradient_terms(ki, options)
    uni_drift = usp + ug

    # Fault component
    if ki.ref_fault is not None:
        faults_drift = _get_faults_terms(ki, execution_mode)
        cov = c_o * (cov_grad + cov_sp + cov_grad_sp) + uni_drift + faults_drift  # *  NOTE: (miguel) The magic terms are real and now they are already included
    else:
        faults_drift = np.zeros(cov_grad.shape)
        cov = c_o * (cov_grad + cov_sp + cov_grad_sp) + uni_drift

    if gempy_engine.config.DEBUG_MODE:
        Solutions.debug_input_data['cov_grad'] = cov_grad
        Solutions.debug_input_data['cov_sp'] = cov_sp
        Solutions.debug_input_data['cov_grad_sp'] = cov_grad_sp
        Solutions.debug_input_data['usp'] = usp
        Solutions.debug_input_data['ug'] = ug
        Solutions.debug_input_data['uni_drift'] = uni_drift
        Solutions.debug_input_data['faults_drift'] = faults_drift

    return cov


def _get_cov_grad(
        dm,
        k_a,
        k_p_ref,
        nugget,
        nugget_implementation: NuggetImplementation,
        execution_mode: KernelExecutionMode = KernelExecutionMode.DENSE,
):
    cov_grad = dm.hu * dm.hv / (dm.r_ref_ref ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * dm.perp_matrix  # C
    if nugget_implementation is NuggetImplementation.LEGACY:
        return _get_cov_grad_legacy(cov_grad, dm, nugget, execution_mode)

    nugget_matrix = _nugget_diagonal(
        matrix_size=cov_grad.shape[0],
        nugget=nugget,
        start=0,
        execution_mode=execution_mode,
    )
    return cov_grad + nugget_matrix * dm.perp_matrix


def _get_cov_grad_legacy(cov_grad, dm, nugget, execution_mode: KernelExecutionMode):
    grad_nugget = nugget[0]
    if execution_mode is KernelExecutionMode.DENSE:
        eye = BackendTensor.t.eye(cov_grad.shape[0])
        nugget_selector = eye * dm.perp_matrix
        nugget_matrix = nugget_selector * grad_nugget
        cov_grad += nugget_matrix
        return cov_grad

    matrix_shape = dm.hu.shape[0]
    LazyTensor = _lazy_tensor_class()
    diag_ = BackendTensor.arange(matrix_shape, dtype=BackendTensor.dtype_obj).reshape(-1, 1)

    diag_i = LazyTensor(diag_[:, None])
    diag_j = LazyTensor(diag_[None, :])
    nugget_matrix = ((0.5 - (diag_i - diag_j) ** 2).step() * grad_nugget) * dm.perp_matrix
    cov_grad += nugget_matrix
    return cov_grad


def _get_cov_surface_points(
        dm,
        k_ref_ref,
        k_ref_rest,
        k_rest_ref,
        k_rest_rest,
        options: KernelOptions,
        nugget_effect,
        grad_matrix_size,
        execution_mode: KernelExecutionMode = KernelExecutionMode.DENSE,
):
    cov_surface_points = options.i_res * (k_rest_rest - k_rest_ref - k_ref_rest + k_ref_ref)

    if options.nugget_implementation is NuggetImplementation.LEGACY:
        return _get_cov_surface_points_legacy(
            cov_surface_points,
            dm,
            k_rest_ref,
            nugget_effect.rest,
            grad_matrix_size,
            execution_mode,
        )

    nugget_matrix = _surface_point_nugget_matrix(
        matrix_size=cov_surface_points.shape[0],
        nuggets=nugget_effect,
        start=grad_matrix_size,
        mode=options.nugget_implementation,
        execution_mode=execution_mode,
    )
    flipped_perp_matrix = (dm.perp_matrix - 1) * -1
    return cov_surface_points + nugget_matrix * flipped_perp_matrix


def _get_cov_surface_points_legacy(
        cov_surface_points,
        dm,
        k_rest_ref,
        nugget,
        grad_matrix_size: int,
        execution_mode: KernelExecutionMode,
):
    if execution_mode is KernelExecutionMode.DENSE:
        cov_shape = cov_surface_points.shape[0]
        shape_sp_size = nugget.shape[0]
        diag = BackendTensor.t.eye(cov_shape)
        modified_diag = BackendTensor.t.zeros((cov_shape, cov_shape), dtype=BackendTensor.dtype_obj)
        modified_diag[
            grad_matrix_size:grad_matrix_size + shape_sp_size,
            grad_matrix_size:grad_matrix_size + shape_sp_size,
        ] = nugget
        cov_surface_points += modified_diag * diag
        return cov_surface_points

    matrix_shape = k_rest_ref.shape[0]
    LazyTensor = _lazy_tensor_class()
    diag_ = BackendTensor.arange(matrix_shape, dtype=BackendTensor.dtype_obj).reshape(-1, 1)

    nuggets = BackendTensor.t.zeros(matrix_shape, dtype=BackendTensor.dtype_obj)
    nuggets[grad_matrix_size:grad_matrix_size + nugget.shape[0]] += nugget
    nuggets_lazy = LazyTensor(nuggets[None, :, None])
    diag_i = LazyTensor(diag_[:, None])
    diag_j = LazyTensor(diag_[None, :])
    nugget_matrix = (0.5 - (diag_i - diag_j) ** 2).step() * nuggets_lazy
    flipped_perp_matrix = (dm.perp_matrix - 1) * -1
    cov_surface_points += nugget_matrix * flipped_perp_matrix
    return cov_surface_points


def _surface_point_nugget_matrix(
        matrix_size: int,
        nuggets: SurfacePointNuggets,
        start: int,
        mode: NuggetImplementation,
        execution_mode: KernelExecutionMode,
):
    match mode:
        case NuggetImplementation.DIAGONAL_REF_REST:
            return _nugget_diagonal(
                matrix_size,
                nuggets.rest + nuggets.reference,
                start,
                execution_mode,
            )
        case NuggetImplementation.FULL_POINT_COVARIANCE:
            return _full_point_nugget_covariance(matrix_size, nuggets, start, execution_mode)
        case _:
            raise ValueError(f"Unknown nugget implementation: {mode}")


def _full_point_nugget_covariance(
        matrix_size: int,
        nuggets: SurfacePointNuggets,
        start: int,
        execution_mode: KernelExecutionMode,
):
    rest_diagonal = _nugget_diagonal(
        matrix_size,
        nuggets.rest,
        start,
        execution_mode,
    )
    reference = _pad_surface_values(matrix_size, nuggets.reference, start)
    surface_ids = _pad_surface_values(matrix_size, nuggets.surface_ids, start)
    surface_mask = _pad_surface_values(
        matrix_size,
        BackendTensor.t.ones(nuggets.rest.shape[0], dtype=nuggets.rest.dtype),
        start,
    )

    if execution_mode is KernelExecutionMode.DENSE:
        same_surface = surface_ids[:, None] == surface_ids[None, :]
        shared_reference = (
            same_surface
            * surface_mask[:, None]
            * surface_mask[None, :]
            * reference[:, None]
        )
        return rest_diagonal + shared_reference

    LazyTensor = _lazy_tensor_class()
    surface_i = LazyTensor(surface_ids[:, None, None])
    surface_j = LazyTensor(surface_ids[None, :, None])
    mask_i = LazyTensor(surface_mask[:, None, None])
    mask_j = LazyTensor(surface_mask[None, :, None])
    reference_i = LazyTensor(reference[:, None, None])
    same_surface = (0.5 - (surface_i - surface_j) ** 2).step()
    return rest_diagonal + same_surface * mask_i * mask_j * reference_i


def _pad_surface_values(matrix_size: int, values, start: int):
    return BackendTensor.tfnp.concatenate((
        BackendTensor.t.zeros(start, dtype=values.dtype),
        values,
        BackendTensor.t.zeros(matrix_size - start - values.shape[0], dtype=values.dtype),
    ))


def _nugget_diagonal(
        matrix_size: int,
        nugget,
        start: int,
        execution_mode: KernelExecutionMode = KernelExecutionMode.DENSE,
):
    """Build a dense or lazy diagonal while preserving nugget gradients."""
    values = BackendTensor.tfnp.concatenate((
            BackendTensor.t.zeros(start, dtype=nugget.dtype),
            nugget,
            BackendTensor.t.zeros(matrix_size - start - nugget.shape[0], dtype=nugget.dtype),
    ))
    if execution_mode is KernelExecutionMode.DENSE:
        return BackendTensor.t.eye(matrix_size, dtype=nugget.dtype) * values

    LazyTensor = _lazy_tensor_class()

    diag_ = BackendTensor.arange(matrix_size, dtype=nugget.dtype).reshape(-1, 1)
    diag_i = LazyTensor(diag_[:, None])
    diag_j = LazyTensor(diag_[None, :])
    values_j = LazyTensor(values[None, :, None])
    return (0.5 - (diag_i - diag_j) ** 2).step() * values_j


def _lazy_tensor_class():
    if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
        from pykeops.torch import LazyTensor
    elif BackendTensor.engine_backend == AvailableBackends.numpy:
        from pykeops.numpy import LazyTensor
    else:
        raise NotImplementedError("PyKeOps is not implemented for this backend")
    return LazyTensor


def _get_cross_cov_grad_sp(dm, k_p_ref, k_p_rest, options):
    cov_grad_sp = options.gi_res * (- dm.huv_rest * k_p_rest + dm.huv_ref * k_p_ref)
    return cov_grad_sp


def _get_universal_gradient_terms(ki, options):
    # First term
    ug = (ki.ori_drift.dips_ug_ai * ki.ori_drift.dips_ug_aj).sum(axis=-1)
    # Second term
    ug2 = (ki.ori_drift.dips_ug_bi * ki.ori_drift.dips_ug_bj).sum(axis=-1)
    # Third term
    ug3_aux = (ki.ori_drift.dips_ug_ci * ki.ori_drift.dips_ug_cj).sum(axis=-1)
    third_term_selector = -1 * (-2 + (ki.ori_drift.selector_ci * ki.ori_drift.selector_cj).sum(axis=-1))
    ug3 = ug3_aux * third_term_selector
    selector = (ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1)).sum(-1)
    total_ug = selector * (ug + options.gi_res * ug2 + options.gi_res * ug3)
    return total_ug


def _get_faults_terms(
        ki: KernelInput,
        execution_mode: KernelExecutionMode = KernelExecutionMode.DENSE,
) -> np.ndarray:
    fault_ref = (ki.ref_fault.faults_i * ki.ref_fault.faults_j).sum(axis=-1)
    fault_rest = (ki.rest_fault.faults_i * ki.rest_fault.faults_j).sum(axis=-1)

    cov_size = ki.ref_fault.faults_i.shape[0]
    fault_n = ki.ref_fault.n_faults_i  # TODO: Here we are going to have to loop

    selector_components = _structs.DriftMatrixSelector(
        x_size=cov_size,
        y_size=cov_size,
        n_drift_eq=fault_n,
        drift_start_post_x=cov_size - fault_n,
        drift_start_post_y=cov_size - fault_n
    )
    
    if execution_mode is KernelExecutionMode.SYMBOLIC:
        selector_components = selector_components.upgrade_tensors()
    
    selector = (selector_components.sel_ui * (selector_components.sel_vj + 1)).sum(axis=-1)

    fault_matrix = selector * (fault_ref - fault_rest + 0.00000001) * 1
    return fault_matrix


def _get_universal_sp_terms(ki, options):
    # degree 1
    usp_ref = (ki.ref_drift.dipsPoints_ui_ai * ki.ref_drift.dipsPoints_ui_aj).sum(axis=-1)
    usp_rest = (ki.rest_drift.dipsPoints_ui_ai * ki.rest_drift.dipsPoints_ui_aj).sum(axis=-1)

    # degree 2
    usp_ref_d2b = (ki.ref_drift.dipsPoints_ui_bi1 * ki.ref_drift.dipsPoints_ui_bj1).sum(axis=-1)
    usp_ref_d2c = (ki.ref_drift.dipsPoints_ui_bi2 * ki.ref_drift.dipsPoints_ui_bj2).sum(axis=-1)
    usp_ref_d2 = usp_ref_d2b * usp_ref_d2c

    usp_rest_d2b = (ki.rest_drift.dipsPoints_ui_bi1 * ki.rest_drift.dipsPoints_ui_bj1).sum(axis=-1)
    usp_rest_d2c = (ki.rest_drift.dipsPoints_ui_bi2 * ki.rest_drift.dipsPoints_ui_bj2).sum(axis=-1)
    usp_rest_d2 = usp_rest_d2b * usp_rest_d2c

    selector = (ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1)).sum(-1)
    usp_d2 = -1 * selector * ((options.i_res * (usp_rest_d2 - usp_ref_d2)) + (options.gi_res * (usp_rest - usp_ref)))
    return usp_d2
