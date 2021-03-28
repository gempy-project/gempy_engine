from gempy_engine.config import BackendTensor
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.covariance._structs import KernelInput

tfnp = BackendTensor.tfnp
tensor_types = BackendTensor.tensor_types

def create_covariance(ki: KernelInput, options: InterpolationOptions) -> tensor_types:
    kernel_f = options.kernel_function
    a = options.range
    c_o = options.c_o

    dif_ref_ref, dif_rest_rest, hu, hv, huv_ref, huv_rest, perp_matrix, r_ref_ref, r_ref_rest, r_rest_ref, r_rest_rest = \
        _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices)

    k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest = \
        _compute_all_kernel_terms(a, kernel_f, r_ref_ref, r_ref_rest, r_rest_ref, r_rest_rest)

    cov_grad = hu * hv / (r_ref_ref ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * perp_matrix  # C
    cov_sp = k_rest_rest - k_rest_ref - k_ref_rest + k_ref_ref  # It is expanding towards cross
    cov_grad_sp = - huv_rest * k_p_rest + huv_ref * k_p_ref  # C

    # TODO: This Universal term seems buggy. It should also have a rest component!
    usp = (ki.ref_drift.dipsPoints_ui_ai * ki.ref_drift.dipsPoints_ui_aj).sum(axis=-1)
    ug = (ki.ori_drift.dips_ug_ai * ki.ori_drift.dips_ug_aj).sum(axis=-1)
    drift = (usp + ug) * (ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1)).sum(-1)
    cov = c_o * (cov_grad + cov_sp + cov_grad_sp + drift)

    return cov


def _compute_all_kernel_terms(a, kernel_f, r_ref_ref, r_ref_rest, r_rest_ref, r_rest_rest):

    k_rest_rest = kernel_f.base_function(r_rest_rest, a)
    k_ref_ref = kernel_f.base_function(r_ref_ref, a)
    k_ref_rest = kernel_f.base_function(r_ref_rest, a)
    k_rest_ref = kernel_f.base_function(r_rest_ref, a)
    k_p_ref = kernel_f.derivative_div_r(r_ref_ref, a)  # First derivative DIVIDED BY r C'(r)/r
    k_p_rest = kernel_f.derivative_div_r(r_rest_rest, a)  # First derivative DIVIDED BY r C'(r)/r
    k_a = kernel_f.second_derivative(r_ref_ref, a)  # Second derivative of the kernel
    return k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest


def _compute_all_distance_matrices(cs, ori_sp_matrices):
    dif_ref_ref = ori_sp_matrices.dip_ref_i - ori_sp_matrices.dip_ref_j
    dif_rest_rest = ori_sp_matrices.diprest_i - ori_sp_matrices.diprest_j
    hu = (dif_ref_ref * (cs.hu_sel_i * cs.hu_sel_j)).sum(axis=-1)  # C
    hv = -(dif_ref_ref * (cs.hv_sel_i * cs.hv_sel_j)).sum(axis=-1)  # C

    hu_ref = dif_ref_ref * (cs.hu_sel_i * cs.hu_sel_points_j)
    hv_ref = dif_ref_ref * (cs.hv_sel_points_i * cs.hv_sel_j)
    huv_ref = hu_ref.sum(axis=-1) - hv_ref.sum(axis=-1)  # C

    hu_rest = dif_rest_rest * (cs.hu_sel_i * cs.hu_sel_points_j)
    hv_rest = dif_rest_rest * (cs.hv_sel_points_i * cs.hv_sel_j)
    huv_rest = hu_rest.sum(axis=-1) - hv_rest.sum(axis=-1)  # C

    perp_matrix = (cs.hu_sel_i * cs.hv_sel_j).sum(axis=-1)
    if BackendTensor.pykeops_enabled is True:
        r_ref_ref = ((dif_ref_ref ** 2).sum(-1)).sqrt()
        r_rest_rest = ((dif_rest_rest ** 2).sum(-1)).sqrt()
        r_ref_rest = (((ori_sp_matrices.dip_ref_i - ori_sp_matrices.diprest_j) ** 2).sum(-1)).sqrt()
        r_rest_ref = (((ori_sp_matrices.diprest_i - ori_sp_matrices.dip_ref_j) ** 2).sum(-1)).sqrt()
    else:
        r_ref_ref = tfnp.sqrt((dif_ref_ref ** 2).sum(-1))
        r_rest_rest = tfnp.sqrt((dif_rest_rest ** 2).sum(-1))
        r_ref_rest = tfnp.sqrt(((ori_sp_matrices.dip_ref_i - ori_sp_matrices.diprest_j) ** 2).sum(-1))
        r_rest_ref = tfnp.sqrt(((ori_sp_matrices.diprest_i - ori_sp_matrices.dip_ref_j) ** 2).sum(-1))

    return dif_ref_ref, dif_rest_rest, hu, hv, huv_ref, huv_rest, perp_matrix,\
           r_ref_ref, r_ref_rest, r_rest_ref, r_rest_rest
