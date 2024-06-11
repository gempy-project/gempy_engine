import numpy as np
from gempy_engine.config import AvailableBackends

import gempy_engine.config
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.options import KernelOptions
from gempy_engine.core.data.solutions import Solutions
from gempy_engine.modules.kernel_constructor import _structs
from gempy_engine.modules.kernel_constructor._structs import KernelInput

# ! Important for loading the pickle in test_distance_matrix
from gempy_engine.modules.kernel_constructor._internalDistancesMatrices import InternalDistancesMatrices

global_nugget = 1e-5


def get_covariance(c_o, dm, k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest, ki: KernelInput, options):
    cov_grad = _get_cov_grad(dm, k_a, k_p_ref, ki.nugget_grad)
    cov_sp = _get_cov_surface_points(dm, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest,
                                     options, ki.nugget_scalar, ki.nugget_grad.shape[1])  # TODO: Add nugget effect properly (individual) # cov_sp += np.eye(cov_sp.shape[0]) * .00000001
    cov_grad_sp = _get_cross_cov_grad_sp(dm, k_p_ref, k_p_rest, options)  # C
    
    # Universal drift
    usp = _get_universal_sp_terms(ki, options)
    ug = _get_universal_gradient_terms(ki, options)
    uni_drift = usp + ug

    # Fault component
    if ki.ref_fault is not None:
        faults_drift = _get_faults_terms(ki)
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


def _get_cov_grad(dm, k_a, k_p_ref, nugget):
    cov_grad = dm.hu * dm.hv / (dm.r_ref_ref ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * dm.perp_matrix  # C
    grad_nugget = nugget[0, 0]
    if BackendTensor.pykeops_enabled is False:
        eye = BackendTensor.t.array(np.eye(cov_grad.shape[0], dtype=BackendTensor.dtype))
        nugget_selector = eye * dm.perp_matrix
        nugget_matrix = nugget_selector * grad_nugget
        cov_grad += nugget_matrix
    else:
        matrix_shape = dm.hu.shape[0]
        if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
            from pykeops.torch import LazyTensor
            diag_ = BackendTensor.t.arange(matrix_shape).reshape(-1, 1).type(BackendTensor.dtype_obj)
        elif BackendTensor.engine_backend == AvailableBackends.numpy:
            from pykeops.numpy import LazyTensor
            diag_ = np.arange(matrix_shape).reshape(-1, 1).astype(BackendTensor.dtype)
        else:
            raise NotImplementedError("Pykeops is not implemented for this backend")
        
        diag_i = LazyTensor(diag_[:, None])
        diag_j = LazyTensor(diag_[None, :])
        
        nugget_matrix = (((0.5 - (diag_i - diag_j)**2).step()) * grad_nugget) * dm.perp_matrix
        cov_grad += nugget_matrix
        
    return cov_grad


def _get_cov_surface_points(dm, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest, options: KernelOptions, nugget_effect, grad_matrix_size):
    cov_surface_points = options.i_res * (k_rest_rest - k_rest_ref - k_ref_rest + k_ref_ref)
    
    if BackendTensor.pykeops_enabled is False: # Add nugget effect for ref and rest point
        cov_shape = cov_surface_points.shape[0]
        shape_sp_size = nugget_effect.shape[0]
        
        diag = BackendTensor.t.array(np.eye(cov_shape, dtype=BackendTensor.dtype))
        # Nullify all the diagonal values that are not in the surface points block
        modified_diag = BackendTensor.t.zeros((cov_shape, cov_shape), dtype=BackendTensor.dtype_obj)
        modified_diag[
            grad_matrix_size:grad_matrix_size+shape_sp_size,
            grad_matrix_size:grad_matrix_size+shape_sp_size
            ] = nugget_effect

        cov_surface_points += modified_diag * diag  
    else:
        matrix_shape = k_rest_ref.shape[0]
        if BackendTensor.engine_backend == AvailableBackends.PYTORCH:
            from pykeops.torch import LazyTensor
            diag_ = BackendTensor.t.arange(matrix_shape).reshape(-1, 1).type(BackendTensor.dtype_obj)
        elif BackendTensor.engine_backend == AvailableBackends.numpy:
            from pykeops.numpy import LazyTensor
            diag_ = np.arange(matrix_shape).reshape(-1, 1).astype(BackendTensor.dtype)
        else:
            raise NotImplementedError("Pykeops is not implemented for this backend")
        
        nuggets = BackendTensor.t.zeros(matrix_shape, dtype=BackendTensor.dtype_obj)
        nuggets[grad_matrix_size:grad_matrix_size+nugget_effect.shape[0]] += nugget_effect

        nuggets_lazy = LazyTensor(nuggets[None, :, None])  # Reshaping for proper broadcasting
        diag_i = LazyTensor(diag_[:, None])
        diag_j = LazyTensor(diag_[None, :])
        nugget_matrix = (((0.5 - (diag_i - diag_j) ** 2).step()) * nuggets_lazy)

        flipped_perp_matrix = (dm.perp_matrix - 1) * -1
        cov_surface_points += nugget_matrix * flipped_perp_matrix

    return cov_surface_points


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


def _get_faults_terms(ki: KernelInput) -> np.ndarray:
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
