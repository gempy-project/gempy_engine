from . import _structs
from ._covariance_assembler import _get_covariance
from ._internalDistancesMatrices import InternalDistancesMatrices
from ._structs import KernelInput, CartesianSelector, OrientationSurfacePointsCoords
from ...core.backend_tensor import BackendTensor as bt
from ...core.data.kernel_classes.kernel_functions import KernelFunction
from ...core.data.options import KernelOptions

tensor_types = bt.tensor_types


def create_cov_kernel(ki: KernelInput, options: KernelOptions) -> tensor_types:
    kernel_f: KernelFunction = options.kernel_function.value
    a = options.range
    c_o = options.c_o

    dm = _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices, kernel_f.consume_sq_distance)

    k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest = \
        _compute_all_kernel_terms(a, kernel_f, dm.r_ref_ref, dm.r_ref_rest, dm.r_rest_ref, dm.r_rest_rest)

    cov = _get_covariance(c_o, dm, k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest, ki, options)

    return cov


# noinspection DuplicatedCode
def create_scalar_kernel(ki: KernelInput, options: KernelOptions) -> tensor_types:
    kernel_f = options.kernel_function.value
    a = options.range
    c_o = options.c_o

    # region distances
    dm = _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices, kernel_f.consume_sq_distance)

    k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest = \
        _compute_all_kernel_terms(a, kernel_f, dm.r_ref_ref, dm.r_ref_rest, dm.r_rest_ref, dm.r_rest_rest)

    # endregion

    # region sp and grad-sp
    sigma_0_sp = - options.i_res * (k_rest_rest - k_ref_ref)
    sigma_0_grad_sp = options.gi_res * (dm.hu_ref * k_p_ref)
    # endregion

    # region universal_sp
    usp_ref     = bt.t.sum(ki.ref_drift.dipsPoints_ui_ai * ki.ref_drift.dipsPoints_ui_aj, axis=-1)
    usp_ref_d2b = bt.t.sum(ki.ref_drift.dipsPoints_ui_bi1 * ki.ref_drift.dipsPoints_ui_bj1, axis=-1)
    usp_ref_d2c = bt.t.sum(ki.ref_drift.dipsPoints_ui_bi2 * ki.ref_drift.dipsPoints_ui_bj2, axis=-1)
    
    usp_ref_d2 = usp_ref_d2b * usp_ref_d2c
    
    selector = bt.t.sum(ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1), -1)
    uni_drift = selector * (options.gi_res * usp_ref + options.i_res * usp_ref_d2)
    # endregion

    # region faults_sp

    if ki.ref_fault is not None:
        cov_size = ki.ref_fault.faults_i.shape[0]
        j_size = ki.ref_fault.faults_j.shape[1]
        fault_n = ki.ref_fault.n_faults_i

        selector_components = _structs.DriftMatrixSelector(
            x_size=cov_size,
            y_size=j_size,
            n_drift_eq=fault_n,
            drift_start_post_x=cov_size - fault_n,
            drift_start_post_y=j_size - fault_n
        )

        selector = bt.t.sum(selector_components.sel_ui * (selector_components.sel_vj + 1), axis=-1)
        fault_drift = selector * bt.t.sum(ki.ref_fault.faults_i * ki.ref_fault.faults_j, axis=-1)
    else:
        fault_drift = 0

    # endregion
    return c_o * (sigma_0_sp + sigma_0_grad_sp) + uni_drift + fault_drift


# noinspection DuplicatedCode
def create_grad_kernel(ki: KernelInput, options: KernelOptions) -> tensor_types:
    kernel_f = options.kernel_function.value
    a = options.range
    c_o = options.c_o

    dm = _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices, kernel_f.consume_sq_distance)

    k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest = \
        _compute_all_kernel_terms(a, kernel_f, dm.r_ref_ref, dm.r_ref_rest, dm.r_rest_ref, dm.r_rest_rest)

    sigma_0_grad = (+1) * dm.hu * dm.hv / (dm.r_ref_ref ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * dm.perp_matrix
    sigma_0_sp_grad = -options.gi_res * (dm.hu_ref_grad * k_p_ref - dm.hu_rest_grad * k_p_rest)

    # region drift
    
    ug = bt.t.sum(ki.ori_drift.dips_ug_ai * ki.ori_drift.dips_ug_aj, axis=-1)  # First term
    ug2 = bt.t.sum(ki.ori_drift.dips_ug_bi * ki.ori_drift.dips_ug_bj, axis=-1)  # Second term
    ug3_aux = bt.t.sum(ki.ori_drift.dips_ug_ci * ki.ori_drift.dips_ug_cj, axis=-1)  # Third term

    third_term_selector = bt.t.sum(ki.ori_drift.selector_ci * ki.ori_drift.dips_ug_aj, axis=-1)
    ug3 = ug3_aux * third_term_selector

    selector = bt.t.sum(ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1), -1)
    drift = selector * (ug + options.gi_res * ug2 + options.gi_res * ug3)

    # endregion

    grad_kernel = c_o * (sigma_0_grad + sigma_0_sp_grad) + drift
    return grad_kernel


def _compute_all_kernel_terms(a: int, kernel_f: KernelFunction, r_ref_ref, r_ref_rest, r_rest_ref, r_rest_rest):
    k_rest_rest = kernel_f.base_function(r_rest_rest, a)
    k_ref_ref = kernel_f.base_function(r_ref_ref, a)
    k_ref_rest = kernel_f.base_function(r_ref_rest, a)
    k_rest_ref = kernel_f.base_function(r_rest_ref, a)
    k_p_ref = kernel_f.derivative_div_r(r_ref_ref, a)  # First derivative DIVIDED BY r C'(r)/r
    k_p_rest = kernel_f.derivative_div_r(r_rest_rest, a)  # First derivative DIVIDED BY r C'(r)/r
    k_a = kernel_f.second_derivative(r_ref_ref, a)  # Second derivative of the kernel
    return k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest


def _compute_all_distance_matrices(cs: CartesianSelector, ori_sp_matrices: OrientationSurfacePointsCoords,
                                   square_distance: bool) -> InternalDistancesMatrices:
    dif_ref_ref = ori_sp_matrices.dip_ref_i - ori_sp_matrices.dip_ref_j
    dif_rest_rest = ori_sp_matrices.diprest_i - ori_sp_matrices.diprest_j

    hu = bt.t.sum(dif_ref_ref * (cs.hu_sel_i * cs.hu_sel_j), axis=-1)  # C
    hv = -bt.t.sum(dif_ref_ref * (cs.hv_sel_i * cs.hv_sel_j), axis=-1)  # C

    hu_ref = dif_ref_ref * (cs.hu_sel_i * cs.h_sel_ref_j)
    hv_ref = dif_ref_ref * (cs.h_sel_ref_i * cs.hv_sel_j)
    huv_ref = bt.t.sum(hu_ref, axis=-1) - bt.t.sum(hv_ref, axis=-1)  # C

    hu_rest = dif_rest_rest * (cs.hu_sel_i * cs.h_sel_rest_j)
    hv_rest = dif_rest_rest * (cs.h_sel_rest_i * cs.hv_sel_j)
    huv_rest = bt.t.sum(hu_rest, axis=-1) - bt.t.sum(hv_rest, axis=-1)  # C

    hu_ref = bt.t.sum(hu_ref, axis=-1, keepdims=False)
    hu_rest = bt.t.sum(hu_rest, axis=-1, keepdims=False)

    perp_matrix = bt.t.sum(cs.hu_sel_i * cs.hv_sel_j, axis=-1)

    # For gradients
    hu_ref_grad = bt.t.sum(dif_ref_ref * (cs.h_sel_ref_i * cs.hu_sel_j), axis=-1)
    hu_rest_grad = bt.t.sum(dif_rest_rest * (cs.h_sel_ref_i * cs.hu_sel_j), axis=-1)

    r_ref_ref = bt.t.sum(dif_ref_ref ** 2, axis=-1)
    r_rest_rest = bt.t.sum(dif_rest_rest ** 2, axis=-1)
    r_ref_rest = bt.t.sum((ori_sp_matrices.dip_ref_i - ori_sp_matrices.diprest_j) ** 2, axis=-1)
    r_rest_ref = bt.t.sum((ori_sp_matrices.diprest_i - ori_sp_matrices.dip_ref_j) ** 2, axis=-1)

    if square_distance is False:
        r_ref_ref = bt.t.sqrt(r_ref_ref)
        r_rest_rest = bt.t.sqrt(r_rest_rest)
        r_ref_rest = bt.t.sqrt(r_ref_rest)
        r_rest_ref = bt.t.sqrt(r_rest_ref)

    return InternalDistancesMatrices(
        dif_ref_ref=dif_ref_ref,
        dif_rest_rest=dif_rest_rest,
        hu=hu,
        hv=hv,
        huv_ref=huv_ref,
        huv_rest=huv_rest,
        perp_matrix=perp_matrix,
        r_ref_ref=r_ref_ref,
        r_ref_rest=r_ref_rest,
        r_rest_ref=r_rest_ref,
        r_rest_rest=r_rest_rest,
        hu_ref=hu_ref,
        hu_rest=hu_rest,
        hu_ref_grad=hu_ref_grad,
        hu_rest_grad=hu_rest_grad,
    )
