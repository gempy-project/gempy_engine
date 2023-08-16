import warnings

from gempy_engine.core.data.options import KernelOptions
from gempy_engine.modules.kernel_constructor._covariance_assembler import _get_cov_grad, _get_cov_surface_points, _get_cross_cov_grad_sp, _get_universal_gradient_terms, _get_universal_sp_terms, get_covariance
from gempy_engine.modules.kernel_constructor._kernels_assembler import _compute_all_distance_matrices, _compute_all_kernel_terms
from gempy_engine.modules.kernel_constructor._structs import KernelInput


def _test_covariance_items(ki: KernelInput, options: KernelOptions, item):
    """This method is not to be executed in production. Just for sanity check
    """
    # Deprecation warning
    warnings.warn(DeprecationWarning("This method is not to be executed in production. Just for sanity check"))

    kernel_f = options.kernel_function.value
    a = options.range
    c_o = options.c_o
    
    dm = _compute_all_distance_matrices(
        cs=ki.cartesian_selector,
        ori_sp_matrices=ki.ori_sp_matrices,
        square_distance=True,
        is_gradient=True,
        is_testing=True
    )

    k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest = \
        _compute_all_kernel_terms(a, kernel_f, dm.r_ref_ref, dm.r_ref_rest, dm.r_rest_ref, dm.r_rest_rest)

    if item == "cov_grad":
        cov_grad = _get_cov_grad(dm, k_a, k_p_ref, ki.nugget_grad)
        return cov_grad

    elif item == "cov_sp":
        return _get_cov_surface_points(k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest, options)

    elif item == "cov_grad_sp":
        cov_grad_sp = _get_cross_cov_grad_sp(dm, k_p_ref, k_p_rest, options)

        return cov_grad_sp  # - dm.huv_rest * k_p_rest + dm.huv_ref * k_p_ref

    elif item == "drift_eval":
        usp_ref = (ki.ref_drift.dipsPoints_ui_ai * ki.ref_drift.dipsPoints_ui_aj).sum(axis=-1)
        ug = (ki.ori_drift.dips_ug_ai * ki.ori_drift.dips_ug_aj).sum(axis=-1)
        drift = (usp_ref + ug) * (ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1)).sum(-1)
        return drift

    elif item == "drift_ug":
        total_ug = _get_universal_gradient_terms(ki, options)

        return total_ug

    elif item == "drift_usp":
        usp_d2 = _get_universal_sp_terms(ki, options)

        return usp_d2

    elif item == "drift":
        usp_ref = (ki.ref_drift.dipsPoints_ui_ai * ki.ref_drift.dipsPoints_ui_aj).sum(axis=-1)
        ug = (ki.ori_drift.dips_ug_ai * ki.ori_drift.dips_ug_aj).sum(axis=-1)
        drift = (usp_ref + ug) * (ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1)).sum(-1)
        return drift

    elif item == "cov":

        cov = get_covariance(c_o, dm, k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest, ki, options)
        return cov

    elif item == "sigma_0_sp":
        return c_o * (k_rest_rest - k_ref_ref)  # These are right terms

    elif item == "sigma_0_grad_sp":
        return c_o * (dm.hu_ref * k_p_ref)  # These are right terms

    elif item == "sigma_0_u_sp":
        usp_ref = (ki.ref_drift.dipsPoints_ui_ai * ki.ref_drift.dipsPoints_ui_aj).sum(axis=-1)

        usp_ref_d2b = (ki.ref_drift.dipsPoints_ui_bi1 * ki.ref_drift.dipsPoints_ui_bj1).sum(axis=-1)
        usp_ref_d2c = (ki.ref_drift.dipsPoints_ui_bi2 * ki.ref_drift.dipsPoints_ui_bj2).sum(axis=-1)
        usp_ref_d2 = usp_ref_d2b * usp_ref_d2c

        selector = (ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1)).sum(-1)

        return selector * (options.gi_res * usp_ref + options.i_res * usp_ref_d2)
