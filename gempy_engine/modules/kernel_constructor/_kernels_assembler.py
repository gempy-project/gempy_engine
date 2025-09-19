from . import _structs
from ._covariance_assembler import get_covariance
from ._internalDistancesMatrices import InternalDistancesMatrices
from ._structs import KernelInput, CartesianSelector, OrientationSurfacePointsCoords
from ...core.backend_tensor import BackendTensor as bt
from ...core.data.kernel_classes.kernel_functions import KernelFunction
from ...core.data.options import KernelOptions

tensor_types = bt.tensor_types


def create_cov_kernel(ki: KernelInput, options: KernelOptions) -> tensor_types:
    kernel_f: KernelFunction = options.kernel_function.value

    distances_matrices = _compute_all_distance_matrices(
        cs=ki.cartesian_selector,
        ori_sp_matrices=ki.ori_sp_matrices,
        square_distance=kernel_f.consume_sq_distance,
        is_gradient=False
    )

    kernels: tuple = _compute_all_kernel_terms(
        range_=options.range,
        kernel_functions=kernel_f,
        r_ref_ref=distances_matrices.r_ref_ref,
        r_ref_rest=distances_matrices.r_ref_rest,
        r_rest_ref=distances_matrices.r_rest_ref,
        r_rest_rest=distances_matrices.r_rest_rest
    )

    cov = get_covariance(options.c_o, distances_matrices, *kernels, ki, options)

    return cov


# noinspection DuplicatedCode
def create_scalar_kernel(ki: KernelInput, options: KernelOptions) -> tensor_types:
    kernel_f = options.kernel_function.value
    a = options.range
    c_o = options.c_o

    # region distances
    dm = _compute_all_distance_matrices(
        cs=ki.cartesian_selector,
        ori_sp_matrices=ki.ori_sp_matrices,
        square_distance=kernel_f.consume_sq_distance,
        is_gradient=False
    )

    k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest = _compute_all_kernel_terms(
        range_=a,
        kernel_functions=kernel_f,
        r_ref_ref=dm.r_ref_ref,
        r_ref_rest=dm.r_ref_rest,
        r_rest_ref=dm.r_rest_ref,
        r_rest_rest=dm.r_rest_rest
    )

    # endregion

    # region sp and grad-sp
    sigma_0_sp = - options.i_res * (k_rest_rest - k_ref_ref)
    sigma_0_grad_sp = options.gi_res * (dm.hu_ref * k_p_ref)
    # endregion

    # region universal_sp
    usp_ref = bt.t.sum(ki.ref_drift.dipsPoints_ui_ai * ki.ref_drift.dipsPoints_ui_aj, axis=-1)
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
            drift_start_post_y=j_size
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

    dm = _compute_all_distance_matrices(ki.cartesian_selector, ki.ori_sp_matrices,
                                        square_distance=kernel_f.consume_sq_distance, is_gradient=True)

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


def _compute_all_kernel_terms(range_: int, kernel_functions: KernelFunction, r_ref_ref, r_ref_rest, r_rest_ref, r_rest_rest):
    # ? This could also be cached as the DistancesBuffer but let's wait until it becomes a bottle neck

    k_rest_rest = kernel_functions.base_function(r_rest_rest, range_)
    k_ref_ref = kernel_functions.base_function(r_ref_ref, range_)
    k_ref_rest = kernel_functions.base_function(r_ref_rest, range_)
    k_rest_ref = kernel_functions.base_function(r_rest_ref, range_)
    k_p_ref = kernel_functions.derivative_div_r(r_ref_ref, range_)  # First derivative DIVIDED BY r C'(r)/r
    k_p_rest = kernel_functions.derivative_div_r(r_rest_rest, range_)  # First derivative DIVIDED BY r C'(r)/r
    k_a = kernel_functions.second_derivative(r_ref_ref, range_)  # Second derivative of the kernel
    return k_a, k_p_ref, k_p_rest, k_ref_ref, k_ref_rest, k_rest_ref, k_rest_rest


class DistancesBuffer:
    last_internal_distances_matrices: InternalDistancesMatrices = None


# noinspection DuplicatedCode
def _compute_all_distance_matrices(cs: CartesianSelector, ori_sp_matrices: OrientationSurfacePointsCoords,
                                   square_distance: bool, is_gradient: bool, is_testing: bool = False) -> InternalDistancesMatrices:
    # ! For the DistanceBuffer optimization we are assuming that we are always computing the scalar kernel first
    # ! and then the gradient kernel. This is because the gradient kernel needs the scalar kernel distances

    is_cached_matrices = DistancesBuffer.last_internal_distances_matrices is not None
    if is_gradient and is_cached_matrices and is_testing is False:
        distance_matrices: InternalDistancesMatrices = _compute_distances_using_cache(
            cs=cs,
            last_internal_distances_matrices=DistancesBuffer.last_internal_distances_matrices
        )
    else:
        distance_matrices: InternalDistancesMatrices = _compute_distances_new(
            cs=cs,
            ori_sp_matrices=ori_sp_matrices,
            square_distance=square_distance
        )

    if develeping_distances_buffer := False and is_gradient:  # This is for developing
        _check_which_items_are_the_same_between_calls(distance_matrices)
    
    DistancesBuffer.last_internal_distances_matrices = distance_matrices  # * Save common values for next call
    return distance_matrices


def _compute_distances_using_cache(cs, last_internal_distances_matrices: InternalDistancesMatrices) -> InternalDistancesMatrices:
    dif_ref_ref = last_internal_distances_matrices.dif_ref_ref  # Can be cached
    dif_rest_rest = last_internal_distances_matrices.dif_rest_rest  # Can be cached

    hu = last_internal_distances_matrices.hu  # Can be cached
    hv = -bt.t.sum(dif_ref_ref * (cs.hv_sel_i * cs.hv_sel_j), axis=-1)  # Axis dependent

    hu_ref = last_internal_distances_matrices.hu_ref  # Can be cached
    hv_ref = bt.t.sum(dif_ref_ref * (cs.h_sel_ref_i * cs.hv_sel_j), axis=-1)  # Axis dependent
    huv_ref = hu_ref - hv_ref  # Axis dependent

    hu_rest = last_internal_distances_matrices.hu_rest  # Can be cached
    hv_rest = bt.t.sum(dif_rest_rest * (cs.h_sel_rest_i * cs.hv_sel_j), axis=-1)  # Axis dependent
    huv_rest = hu_rest - hv_rest  # Axis dependent

    perp_matrix = bt.t.sum(cs.hu_sel_i * cs.hv_sel_j, axis=-1, dtype="int8")  # Axis dependent

    # region: distance r
    r_ref_ref = last_internal_distances_matrices.r_ref_ref  # Can be cached
    r_rest_rest = last_internal_distances_matrices.r_rest_rest  # Can be cached
    r_ref_rest = last_internal_distances_matrices.r_ref_rest  # Can be cached
    r_rest_ref = last_internal_distances_matrices.r_rest_ref  # Can be cached

    # region: For gradients
    hu_ref_grad = bt.t.sum(dif_ref_ref * (cs.h_sel_ref_i * cs.hu_sel_j), axis=-1)  # Axis dependent
    hu_rest_grad = bt.t.sum(dif_rest_rest * (cs.h_sel_ref_i * cs.hu_sel_j), axis=-1)  # Axis dependent
    # endregion

    new_distance_matrices = InternalDistancesMatrices(
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

    return new_distance_matrices


def _compute_distances_new(cs: CartesianSelector, ori_sp_matrices, square_distance) -> InternalDistancesMatrices:
    dif_ref_ref = ori_sp_matrices.dip_ref_i - ori_sp_matrices.dip_ref_j  # Can be cached
    dif_rest_rest = ori_sp_matrices.diprest_i - ori_sp_matrices.diprest_j  # Can be cached

    hu = bt.t.sum(dif_ref_ref * (cs.hu_sel_i * cs.hu_sel_j), axis=-1)  # Can be cached
    hv = -bt.t.sum(dif_ref_ref * (cs.hv_sel_i * cs.hv_sel_j), axis=-1)  # Axis dependent

    hu_ref = bt.t.sum(dif_ref_ref * (cs.hu_sel_i * cs.h_sel_ref_j), axis=-1)  # Can be cached
    hv_ref = bt.t.sum(dif_ref_ref * (cs.h_sel_ref_i * cs.hv_sel_j), axis=-1)  # Axis dependent
    huv_ref = hu_ref - hv_ref  # Axis dependent

    hu_rest = bt.t.sum(dif_rest_rest * (cs.hu_sel_i * cs.h_sel_rest_j), axis=-1)  # Can be cached
    hv_rest = bt.t.sum(dif_rest_rest * (cs.h_sel_rest_i * cs.hv_sel_j), axis=-1)  # Axis dependent
    huv_rest = hu_rest - hv_rest  # Axis dependent

    perp_matrix = bt.t.sum(cs.hu_sel_i * cs.hv_sel_j, axis=-1, dtype='int8')  # * dtype arg only works for pure numpy. Axis dependent

    # region: distance r
    r_ref_ref = bt.t.sum(dif_ref_ref ** 2, axis=-1)  # Can be cached
    r_rest_rest = bt.t.sum(dif_rest_rest ** 2, axis=-1)  # Can be cached
    r_ref_rest = bt.t.sum((ori_sp_matrices.dip_ref_i - ori_sp_matrices.diprest_j) ** 2, axis=-1)  # Can be cached
    r_rest_ref = bt.t.sum((ori_sp_matrices.diprest_i - ori_sp_matrices.dip_ref_j) ** 2, axis=-1)  # Can be cached

    if square_distance is False:
        # @off
        epsilon = 1e-10  # Add small regularization term to avoid numerical errors
        r_ref_ref = bt.t.sqrt(r_ref_ref + epsilon)
        r_rest_rest = bt.t.sqrt(r_rest_rest + epsilon)
        r_ref_rest = bt.t.sqrt(r_ref_rest + epsilon)
        r_rest_ref = bt.t.sqrt(r_rest_ref + epsilon)
    # endregion

    new_distance_matrices = InternalDistancesMatrices(
        dif_ref_ref=dif_ref_ref,
        dif_rest_rest=dif_rest_rest,
        hu=hu, # used for grads
        hv=hv, # used for grads
        huv_ref=huv_ref,
        huv_rest=huv_rest,
        perp_matrix=perp_matrix,
        r_ref_ref=r_ref_ref,  # used
        r_ref_rest=r_ref_rest, # used
        r_rest_ref=r_rest_ref, # used
        r_rest_rest=r_rest_rest, # used
        hu_ref=hu_ref, # used for normal scalar
        hu_rest=hu_rest,
        hu_ref_grad=None,  # * These are set on the compute_from_cache because we assume that we have already computed the rest at least once
        hu_rest_grad=None,
    )
    return new_distance_matrices


def _check_which_items_are_the_same_between_calls(distance_matrices):
    import numpy as np
    print(f"Checking distances matrices. Shape: {distance_matrices.dif_ref_ref.shape}")
    for k, v in DistancesBuffer.last_internal_distances_matrices.__dict__.items():
        if not np.allclose(v, distance_matrices.__dict__[k]):
            print("Not allclose", k)
