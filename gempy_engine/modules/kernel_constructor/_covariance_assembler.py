from dataclasses import dataclass

from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.kernel_classes.kernel_functions import AvailableKernelFunctions, KernelFunction
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.kernel_constructor._structs import KernelInput

import numpy as np

tfnp = BackendTensor.tfnp
tensor_types = BackendTensor.tensor_types

# TODO: Move this to its right place
euclidean_distances = True


def create_cov_kernel(ki: KernelInput, options: InterpolationOptions) -> tensor_types:
    kernel_f = options.kernel_function.value
    a = options.range
    c_o = options.c_o

    # Calculate euclidean or square distances depending on the function kernel
    global euclidean_distances
    if options.kernel_function == AvailableKernelFunctions.exponential:
        euclidean_distances = False
    else:
        euclidean_distances = True

    dm = _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices)

    k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest = \
        _compute_all_kernel_terms(a, kernel_f, dm.r_ref_ref, dm.r_ref_rest, dm.r_rest_ref, dm.r_rest_rest)

    cov_grad = dm.hu * dm.hv / (dm.r_ref_ref ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * dm.perp_matrix  # C
    cov_sp = k_rest_rest - k_rest_ref - k_ref_rest + k_ref_ref  # It is expanding towards cross
    cov_grad_sp = - dm.huv_rest * k_p_rest + dm.huv_ref * k_p_ref  # C

    # TODO: This Universal term seems buggy. It should also have a rest component!
    usp = (ki.ref_drift.dipsPoints_ui_ai * ki.ref_drift.dipsPoints_ui_aj).sum(axis=-1)
    ug = (ki.ori_drift.dips_ug_ai * ki.ori_drift.dips_ug_aj).sum(axis=-1)
    #drift = (usp + ug) * (ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1)).sum(-1)
    cov = c_o * (cov_grad + cov_sp + cov_grad_sp)# + drift)

    return cov

def create_scalar_kernel(ki: KernelInput, options: InterpolationOptions) -> tensor_types:
    kernel_f = options.kernel_function.value
    a = options.range
    c_o = options.c_o

    # Calculate euclidean or square distances depending on the function kernel
    global euclidean_distances
    if options.kernel_function == AvailableKernelFunctions.exponential:
        euclidean_distances = False
    else:
        euclidean_distances = True

    dm = _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices)

    k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest = \
        _compute_all_kernel_terms(a, kernel_f, dm.r_ref_ref, dm.r_ref_rest, dm.r_rest_ref, dm.r_rest_rest)

    sigma_0_sp = k_rest_rest - k_ref_ref # This are right terms
    sigma_0_grad_sp = dm.huv_ref * k_p_ref # this are the right terms

    return c_o * (- sigma_0_sp + sigma_0_grad_sp) # TODO: + drift


def create_grad_kernel(ki: KernelInput, options: InterpolationOptions) -> tensor_types:
    kernel_f = options.kernel_function.value
    a = options.range
    c_o = options.c_o

    # Calculate euclidean or square distances depending on the function kernel
    global euclidean_distances
    if options.kernel_function == AvailableKernelFunctions.exponential:
        euclidean_distances = False
    else:
        euclidean_distances = True

    dm = _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices)

    k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest = \
        _compute_all_kernel_terms(a, kernel_f, dm.r_ref_ref, dm.r_ref_rest, dm.r_rest_ref, dm.r_rest_rest)

    sigma_0_grad = (+1) * dm.hu * dm.hv / (dm.r_ref_ref ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * dm.perp_matrix
    sigma_0_sp_grad =  dm.huv_ref * k_p_ref -  dm.huv_rest * k_p_rest
    return c_o * (sigma_0_grad + sigma_0_sp_grad)

@dataclass
class InternalDistancesMatrices:
    dif_ref_ref: np.ndarray
    dif_rest_rest: np.ndarray
    hu: np.ndarray
    hv: np.ndarray
    huv_ref: np.ndarray
    huv_rest: np.ndarray
    perp_matrix: np.ndarray
    r_ref_ref: np.ndarray
    r_ref_rest: np.ndarray
    r_rest_ref: np.ndarray
    r_rest_rest: np.ndarray


def _compute_all_kernel_terms(a: int, kernel_f: KernelFunction, r_ref_ref, r_ref_rest, r_rest_ref, r_rest_rest):

    k_rest_rest = kernel_f.base_function(r_rest_rest, a)
    k_ref_ref = kernel_f.base_function(r_ref_ref, a)
    k_ref_rest = kernel_f.base_function(r_ref_rest, a)
    k_rest_ref = kernel_f.base_function(r_rest_ref, a)
    k_p_ref = kernel_f.derivative_div_r(r_ref_ref, a)  # First derivative DIVIDED BY r C'(r)/r
    k_p_rest = kernel_f.derivative_div_r(r_rest_rest, a)  # First derivative DIVIDED BY r C'(r)/r
    k_a = kernel_f.second_derivative(r_ref_ref, a)  # Second derivative of the kernel
    return k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest


def _compute_all_distance_matrices(cs, ori_sp_matrices) -> InternalDistancesMatrices:
    dif_ref_ref = ori_sp_matrices.dip_ref_i - ori_sp_matrices.dip_ref_j

    dif_rest_rest = ori_sp_matrices.diprest_i - ori_sp_matrices.diprest_j
    hu = (dif_ref_ref * (cs.hu_sel_i * cs.hu_sel_j)).sum(axis=-1)  # C
    hv = -(dif_ref_ref * (cs.hv_sel_i * cs.hv_sel_j)).sum(axis=-1)  # C

    hu_ref = dif_ref_ref * (cs.hu_sel_i * cs.h_sel_ref_j)
    hv_ref = dif_ref_ref * (cs.h_sel_ref_i * cs.hv_sel_j)
    huv_ref =  hu_ref.sum(axis=-1) - hv_ref.sum(axis=-1)  # C

    hu_rest = dif_rest_rest * (cs.hu_sel_i * cs.h_sel_rest_j)
    hv_rest = dif_rest_rest * (cs.h_sel_rest_i * cs.hv_sel_j)
    huv_rest = hu_rest.sum(axis=-1) - hv_rest.sum(axis=-1)  # C

    perp_matrix = (cs.hu_sel_i * cs.hv_sel_j).sum(axis=-1)
    if BackendTensor.pykeops_enabled is True:

        r_ref_ref = dif_ref_ref.sqdist(dif_ref_ref)
        r_rest_rest = dif_rest_rest.sqdist(dif_rest_rest)
        r_ref_rest = ori_sp_matrices.dip_ref_i.sqdist(ori_sp_matrices.diprest_j)
        r_rest_ref = ori_sp_matrices.diprest_j.sqdist(ori_sp_matrices.dip_ref_j)

    else:
        r_ref_ref = (dif_ref_ref ** 2).sum(-1)
        r_rest_rest = (dif_rest_rest ** 2).sum(-1)
        r_ref_rest = ((ori_sp_matrices.dip_ref_i - ori_sp_matrices.diprest_j) ** 2).sum(-1)
        r_rest_ref = ((ori_sp_matrices.diprest_i - ori_sp_matrices.dip_ref_j) ** 2).sum(-1)

        if euclidean_distances:
            r_ref_ref = tfnp.sqrt(r_ref_ref)
            r_rest_rest = tfnp.sqrt(r_rest_rest)
            r_ref_rest = tfnp.sqrt(r_ref_rest)
            r_rest_ref = tfnp.sqrt(r_rest_ref)

    return InternalDistancesMatrices(dif_ref_ref, dif_rest_rest, hu, hv, huv_ref, huv_rest, perp_matrix,
                                     r_ref_ref, r_ref_rest, r_rest_ref, r_rest_rest)



def _test_covariance_items(ki: KernelInput, options: InterpolationOptions, item):
    """This method is not to be executed in production. Just for sanity check
    """
    kernel_f = options.kernel_function.value
    a = options.range
    c_o = options.c_o

    dm = _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices)

    # with open('distance_matrices.pickle', 'wb') as handle:
    #     import pickle
    #     pickle.dump(dm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest = \
        _compute_all_kernel_terms(a, kernel_f, dm.r_ref_ref, dm.r_ref_rest, dm.r_rest_ref, dm.r_rest_rest)

    if item == "cov_grad":
        cov_grad = dm.hu * dm.hv / (dm.r_ref_ref ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * dm.perp_matrix  # C
        return cov_grad

    elif item == "cov_sp":
        return k_rest_rest - k_rest_ref - k_ref_rest + k_ref_ref

    elif item == "cov_grad_sp":
        return - dm.huv_rest * k_p_rest + dm.huv_ref * k_p_ref

    elif item == "drift":
        usp = (ki.ref_drift.dipsPoints_ui_ai * ki.ref_drift.dipsPoints_ui_aj).sum(axis=-1)
        ug = (ki.ori_drift.dips_ug_ai * ki.ori_drift.dips_ug_aj).sum(axis=-1)
        drift = (usp + ug) * (ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1)).sum(-1)
        return drift
    elif  item == "cov":
        cov_grad = dm.hu * dm.hv / (dm.r_ref_ref ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * dm.perp_matrix  # C
        cov_sp = k_rest_rest - k_rest_ref - k_ref_rest + k_ref_ref  # It is expanding towards cross
        cov_grad_sp = - dm.huv_rest * k_p_rest + dm.huv_ref * k_p_ref  # C

        # TODO: This Universal term seems buggy. It should also have a rest component!
        usp = (ki.ref_drift.dipsPoints_ui_ai * ki.ref_drift.dipsPoints_ui_aj).sum(axis=-1)
        ug = (ki.ori_drift.dips_ug_ai * ki.ori_drift.dips_ug_aj).sum(axis=-1)
        drift = (usp + ug) * (ki.drift_matrix_selector.sel_ui * (ki.drift_matrix_selector.sel_vj + 1)).sum(-1)
        cov =  (cov_grad + cov_sp + cov_grad_sp + drift)

        return cov

    elif item =="sigma_0_sp":
        return c_o * (k_rest_rest - k_ref_ref) # This are right terms

    elif item =="sigma_0_grad_sp":
        return c_o * ( dm.huv_ref * k_p_ref) # This are right terms
