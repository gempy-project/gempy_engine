from typing import Callable, Optional

import numpy as np

from gempy_engine.data_structures.private_structures import SurfacePointsInternals, OrientationsInternals, KernelInput
from gempy_engine.data_structures.public_structures import OrientationsInput, InterpolationOptions
from gempy_engine.systems.generators import squared_euclidean_distances, tensor_transpose
from gempy_engine.config import BackendConfig
from gempy_engine.systems.kernel.aux_functions import b_scalar_assembly

from gempy_engine.config import BackendConfig, AvailableBackends

tfnp = BackendConfig.t
tensor_types = BackendConfig.tensor_types


def kernel_solver(sp_internals: SurfacePointsInternals,
                  ori_internals: OrientationsInternals,
                  interpolation_options: InterpolationOptions,
                  kernel_type='cubic'):
    cov_size = ori_internals.n_orientations_tiled + \
               sp_internals.n_points + \
               interpolation_options.n_uni_eq

    kernel_input = vectors_preparation(sp_internals, ori_internals,
                                       interpolation_options)
    kernels = get_kernels(kernel_type)
    b = b_scalar_assembly(ori_internals, cov_size)

    w = kernel_reduction(
        kernel_input,
        b,
        interpolation_options.range,
        interpolation_options.c_o,
        *kernels
    )
    return w


def get_kernels(typeof: str):
    if typeof == 'cubic':
        from .kernel_functions import cubic_function, cubic_function_p_div_r, cubic_function_a
        return cubic_function, cubic_function_p_div_r, cubic_function_a
    elif typeof == 'gaussian' or typeof == 'exponential':
        from .kernel_functions import exp_function, exp_function_p_div_r, exp_function_a
        return exp_function, exp_function_p_div_r, exp_function_a


def create_covariance(ki: KernelInput, a: float, c_o: float,
                      kernel: Callable, kernel_1st: Callable,
                      kernel_2nd: Callable):
    dif_ref_ref = ki.dip_ref_i - ki.dip_ref_j
    dif_rest_rest = ki.diprest_i - ki.diprest_j

    hu = (dif_ref_ref * (ki.hu_sel_i * ki.hu_sel_j)).sum(axis=-1)  # C
    hv = -(dif_ref_ref * (ki.hv_sel_i * ki.hv_sel_j)).sum(axis=-1)  # C
    perp_matrix = (ki.hu_sel_i * ki.hv_sel_j).sum(axis=-1)

    if BackendConfig.pykeops_enabled is True:
        r_ref_ref = ((dif_ref_ref ** 2).sum(-1)).sqrt()
        r_rest_rest = ((dif_rest_rest ** 2).sum(-1)).sqrt()
        r_ref_rest = (((ki.dip_ref_i - ki.diprest_j) ** 2).sum(-1)).sqrt()
        r_rest_ref = (((ki.diprest_i - ki.dip_ref_j) ** 2).sum(-1)).sqrt()
    else:
        r_ref_ref = tfnp.sqrt((dif_ref_ref ** 2).sum(-1))
        r_rest_rest = tfnp.sqrt((dif_rest_rest ** 2).sum(-1))
        r_ref_rest = tfnp.sqrt(((ki.dip_ref_i - ki.diprest_j) ** 2).sum(-1))
        r_rest_ref = tfnp.sqrt(((ki.diprest_i - ki.dip_ref_j) ** 2).sum(-1))

    k_rest_rest = kernel(r_rest_rest, a)
    k_ref_ref = kernel(r_ref_ref, a)
    k_ref_rest = kernel(r_ref_rest, a)
    k_rest_ref = kernel(r_rest_ref, a)
    k_p_ref = kernel_1st(r_ref_ref, a)  # First derivative DIVIDED BY r C'(r)/r
    k_p_rest = kernel_1st(r_rest_rest, a)  # First derivative DIVIDED BY r C'(r)/r
    k_a = kernel_2nd(r_ref_ref, a)  # Second derivative of the kernel

    hu_ref = dif_ref_ref * (ki.hu_sel_i * ki.hu_sel_points_j)
    hv_ref = dif_ref_ref * (ki.hv_sel_points_i * ki.hv_sel_j)
    huv_ref = hu_ref.sum(axis=-1) - hv_ref.sum(axis=-1)  # C

    hu_rest = dif_rest_rest * (ki.hu_sel_i * ki.hu_sel_points_j)
    hv_rest = dif_rest_rest * (ki.hv_sel_points_i * ki.hv_sel_j)
    huv_rest = hu_rest.sum(axis=-1) - hv_rest.sum(axis=-1)  # C

    c_g = hu * hv / (r_ref_ref ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * perp_matrix  # C
    c_sp = k_rest_rest - k_rest_ref - k_ref_rest + k_ref_ref  # It is expanding towards cross
    c_gsp = - huv_rest * k_p_rest + huv_ref * k_p_ref  # C

    usp = (ki.dipsref_ui_ai * ki.dipsref_ui_aj).sum(axis=-1)
    ug = (ki.dips_ug_ai * ki.dips_ug_aj).sum(axis=-1)
    drift = (usp + ug) * (ki.sel_ui * (ki.sel_vj + 1)).sum(-1)
    cov = c_o * (c_g + c_sp + c_gsp + drift)

    return cov


def kernel_reduction(ki: KernelInput, b, range, c_o, kernel, kernel_1st,
                     kernel_2nd, smooth=.0001):
    cov = create_covariance(ki, range, c_o, kernel, kernel_1st, kernel_2nd)

    if BackendConfig.pykeops_enabled is True and BackendConfig.engine_backend is not AvailableBackends.tensorflow:
        w = cov.solve(np.asarray(b).astype('float32'),
                      alpha=smooth,
                      dtype_acc='float32')
    elif BackendConfig.pykeops_enabled is True and BackendConfig.engine_backend is AvailableBackends.tensorflow:
        w = cov.solve(b.numpy().astype('float32'), alpha=smooth, dtype_acc='float32')
    elif BackendConfig.pykeops_enabled is False and BackendConfig.engine_backend is AvailableBackends.tensorflow:
        w = tfnp.linalg.solve(cov, b)
    elif BackendConfig.pykeops_enabled is False and BackendConfig.engine_backend is not AvailableBackends.tensorflow:
        w = tfnp.linalg.solve(cov, b[:, 0])
    else:
        raise AttributeError('There is a weird combination of libraries?')
    return w


def vectors_preparation(sp_internals: SurfacePointsInternals,
                        ori_internals: OrientationsInternals,
                        interpolation_options: InterpolationOptions,
                        cov_size: Optional[int] = None,
                        backend=None):

    if cov_size is None:
        cov_size = ori_internals.n_orientations_tiled + \
                   sp_internals.n_points + \
                   interpolation_options.n_uni_eq

    dipref_i, dipref_j = input_dips_points(ori_internals.dip_positions_tiled,
                                           sp_internals.ref_surface_points,
                                           interpolation_options)
    diprest_i, diprest_j = input_dips_points(ori_internals.dip_positions_tiled,
                                             sp_internals.rest_surface_points,
                                             interpolation_options)
    hu_sel_i, hu_sel_j, hv_sel_i, hv_sel_j, hu_points_j, hv_points_i = hu_hv_sel(
        ori_internals.n_orientations,
        interpolation_options.number_dimensions,
        sp_internals.n_points,
        cov_size
    )

    dips_ug_ai, dips_ug_aj, dips_ug_bi, dips_ug_bj = input_ug(
        ori_internals,
        sp_internals.n_points,
        interpolation_options
    )

    # Universal
    dipsref_ui_ai, dipsref_ui_aj, dipsref_ui_bi1, \
    dips_uiref_bj1, dipsref_ui_bi2, dips_uiref_bj2 = input_usp(
        sp_internals.ref_surface_points,
        ori_internals.n_orientations_tiled,
        interpolation_options
    )

    dipsrest_ui_ai, dipsrest_ui_aj, dipsrest_ui_bi1, \
    dips_uirest_bj1, dipsrest_ui_bi2, dips_uirest_bj2 = input_usp(
        sp_internals.rest_surface_points,
        ori_internals.n_orientations_tiled,
        interpolation_options
    )

    sel_ui, sel_uj, sel_vi, sel_vj = drift_sel(cov_size,
                                               interpolation_options.n_uni_eq)

    # Prepare Return
    if backend == 'numpy':
        ki = KernelInput(dipref_i, dipref_j, diprest_i, diprest_j, hu_sel_i, hu_sel_j,
                         hv_sel_i, hv_sel_j, hu_points_j, hv_points_i,
                         dips_ug_ai, dips_ug_aj, dips_ug_bi,
                         dips_ug_bj, dipsref_ui_ai, dipsref_ui_aj, dipsref_ui_bi1,
                         dips_uiref_bj1, dipsref_ui_bi2, dips_uiref_bj2, dipsrest_ui_ai,
                         dipsrest_ui_aj, dipsrest_ui_bi1, dips_uirest_bj1,
                         dipsrest_ui_bi2, dips_uirest_bj2,
                         sel_ui, sel_uj, sel_vi, sel_vj)

    elif backend == 'pykeops':
        from pykeops.numpy import LazyTensor
        # Convert to LazyTensors
        args = [dipref_i, dipref_j, diprest_i, diprest_j, hu_sel_i, hu_sel_j,
                hv_sel_i, hv_sel_j, hu_points_j, hv_points_i,
                dips_ug_ai, dips_ug_aj, dips_ug_bi,
                dips_ug_bj, dipsref_ui_ai, dipsref_ui_aj, dipsref_ui_bi1,
                dips_uiref_bj1, dipsref_ui_bi2, dips_uiref_bj2, dipsrest_ui_ai,
                dipsrest_ui_aj, dipsrest_ui_bi1, dips_uirest_bj1,
                dipsrest_ui_bi2, dips_uirest_bj2, sel_ui, sel_uj, sel_vi, sel_vj]
        if BackendConfig.engine_backend is AvailableBackends.tensorflow:
            # TODO: Possibly eventually I have to add i.numpy() to convert
            # the eager tensors to numpy
            ki_args = [LazyTensor(i.astype('float32')) for i in args]
        else:
            ki_args = [LazyTensor(i.astype('float32')) for i in args]
        ki = KernelInput(*ki_args)
    elif backend == 'tf':
        raise NotImplementedError

    else:
        raise AttributeError('backend must benumpy, pykeops or tf.')

    return ki


def input_usp(surface_points, ori_size, interpolation_options: InterpolationOptions):
    n_dim = interpolation_options.number_dimensions
    z = np.zeros((ori_size,
                  interpolation_options.number_dimensions))

    z2 = np.zeros((interpolation_options.n_uni_eq,
                   interpolation_options.number_dimensions))

    zb = z2.copy()
    zc = z2.copy()

    if interpolation_options.uni_degree != 0:
        for i in range(interpolation_options.number_dimensions):
            z2[i, i] = 1

    # Degree 1

    # z2
    # uni_a = np.array([[1, 0],
    #                   [0, 1],
    #                   [0, 0],
    #                   [0, 0],
    #                   [0, 0],
    #                   [0, 0]])

    points_a = np.vstack((z, surface_points, z2))
    points_ia = points_a[:, None, :]
    points_ja = points_a[None, :, :]
    # ui_a = (points_0a * points_1a).sum(axis=-1)
    # print('ui_a: \n', ui_a)

    # Degree 2
    if interpolation_options.uni_degree == 2:
        for i in range(n_dim):
            zb[n_dim + i, i] = 1
        zb[n_dim * 2:, 0] = 1

        for i in range(n_dim):
            zc[n_dim + i, i] = 1
        zc[n_dim * 2:, 1] = 1

    # uni_b1 = np.array([[0, 0],
    #                    [0, 0],
    #                    [1, 0],
    #                    [0, 1],
    #                    [1, 0],
    #                    [1, 0]])

    points_b1 = np.vstack((z, surface_points, zb))
    points_ib1 = points_b1[:, None, :]
    points_jb1 = points_b1[None, :, :]
    # ui_b1 = (points_0b1 * points_1b1).sum(axis=-1)

    points_b2 = np.vstack((z, surface_points, zc))
    points_ib2 = points_b2[:, None, :]
    points_jb2 = points_b2[None, :, :]
    return points_ia, points_ja, points_ib1, points_jb1, points_ib2, points_jb2


def input_ug(ori_internals: OrientationsInternals,
             sp_size: int,
             interpolation_options: InterpolationOptions):
    n_ori = ori_internals.n_orientations
    n_dim = interpolation_options.number_dimensions

    z = np.zeros((n_ori * n_dim + sp_size + interpolation_options.n_uni_eq,
                  interpolation_options.number_dimensions))
    # z2 = z.copy()
    #
    # z[:n_ori, 0] = 1
    # z[n_ori:n_ori * n_dim, 1] = 1
    if interpolation_options.uni_degree != 0:
        # Degree 1
        for i in range(interpolation_options.number_dimensions):
            z[n_ori*i:n_ori * (i+1)] = 1
            z[-interpolation_options.n_uni_eq + i, i] = 1

    # dips_a = np.vstack((ori_internals.dip_positions_tiled, z))
    dips_a = z
    dips_ia = dips_a[:, None, :]
    dips_ja = dips_a[None, :, :]

    # Degree 2
    dips_b_aux = ori_internals.dip_positions_tiled
    z2 = np.zeros((interpolation_options.n_uni_eq,
                   interpolation_options.number_dimensions))

    if interpolation_options.uni_degree == 2:

        for i in range(interpolation_options.number_dimensions):
            dips_b_aux[n_ori * i:n_ori * (i + 1), i] = 0
            z2[sp_size + n_dim + i, i] = 2
            z2[sp_size + n_dim * 2 + i] = 1
            z2[sp_size + n_dim * 2 + i, i] = 0

        #
        # uni_b = np.array([[0, 0, 0],
        #                   [0, 0, 0],
        #                   [0, 0, 0],
        #                   [2, 0, 0],
        #                   [0, 2, 0],
        #                   [0, 0, 2],
        #                   [0, 1, 1],
        #                   [1, 0, 1],
        #                   [1, 1, 0]])

    dips_b = np.vstack((dips_b_aux, z2))
    dips_ib = dips_b[:, None, :]
    dips_jb = dips_b[None, :, :]
    return dips_ia, dips_ja, dips_ib, dips_jb


def hu_hv_sel(n_dips, n_dim, n_points, cov_size):
    sel_0 = np.zeros((cov_size, n_dim))
    sel_1 = np.zeros((cov_size, n_dim))
    sel_2 = np.zeros((cov_size, n_dim))
    for i in range(n_dim):
        sel_0[n_dips * i:n_dips * (i + 1), i] = 1
        sel_1[:n_dips * n_dim, :] = 1
    sel_2[n_dips * n_dim: n_dips * n_dim + n_points, :] = 1

    sel_hui = sel_0[:, None, :]
    sel_huj = sel_1[None, :, :]

    sel_hvi = sel_1[:, None, :]
    sel_hvj = sel_0[None, :, :]

    sel_hu_points_j = sel_2[None, :, :]
    sel_hv_points_i = sel_2[:, None, :]

    return sel_hui, sel_huj, sel_hvi, sel_hvj, sel_hu_points_j, sel_hv_points_i


def drift_sel(cov_size, n_drift_eq):
    sel_0 = np.zeros((cov_size, 2))
    sel_1 = np.zeros((cov_size, 2))

    sel_0[:-n_drift_eq, 0] = 1
    sel_0[-n_drift_eq:, 1] = 1
    sel_1[-n_drift_eq:, :] = 1

    sel_ui = sel_0[:, None, :]
    sel_uj = sel_1[None, :, :]

    sel_vi = -sel_1[:, None, :]
    sel_vj = -sel_0[None, :, :]
    a = sel_ui * (sel_vj + 1)
    print(a)
    return sel_ui, sel_uj, sel_vi, sel_vj
    #
    # n_dim = 2
    # uni_terms = 2
    # dipsref, dipsrest, n_dips = input
    #
    # #perp_cgi = np.zeros((dipsref.shape[0], 2))
    # #perp_cgi[:n_dips * 2, 1] = 1
    # #perp_cgi[n_dips * 2:, 0] = 0
    #
    # # sel_0 = perp_cgi[:, None, :]
    # # sel_1 = perp_cgi[None, :, :]
    # # sel = (sel_0 - sel_1).sum(axis=-1)
    # #sel = (sel_0 * sel_1 - 1).prod(axis=-1)
    # #print(sel)
    # dipsref_0 = dipsref[:, None, :].copy()
    # dipsref_1 = dipsref[None, :, :].copy()
    #
    #
    # sel_0 = np.zeros_like(dipsref)
    # sel_0[:-uni_terms, 0] = 1
    # sel_0[-uni_terms:, 1] = 1
    #
    # sel_1 = np.zeros_like(dipsref)
    # sel_1[-uni_terms:, :] = 1
    #
    # sel_01 = sel_0[:, None, :].copy()
    # sel_11 = sel_1[None, :, :].copy()
    #
    # sel_02 = sel_1[:, None, :].copy()
    # sel_12 = sel_0[None, :, :].copy()
    #
    # hu_ref = (sel_01 * sel_11)
    # hv_ref = (sel_02 * sel_12)
    #
    # print(hu_ref.sum(axis=-1) - hv_ref.sum(axis=-1))


def input_dips_points(dip_pos, points_pos, interpolations_options):
    z = np.zeros((interpolations_options.n_uni_eq,
                  interpolations_options.number_dimensions))

    dipspoints = np.vstack((dip_pos, points_pos, z))
    dips_points0 = dipspoints[:, None, :]
    dips_points1 = dipspoints[None, :, :]

    return dips_points0, dips_points1
