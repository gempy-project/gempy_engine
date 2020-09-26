from typing import Callable

import numpy as np

from gempy_engine.data_structures.private_structures import SurfacePointsInternals, OrientationsInternals, KernelInput
from gempy_engine.data_structures.public_structures import OrientationsInput, InterpolationOptions
from gempy_engine.systems.generators import squared_euclidean_distances, tensor_transpose


def kernel_solver(sp_internals: SurfacePointsInternals,
                  ori_internals: OrientationsInternals,
                  kriging_parameters: InterpolationOptions,
                  ):
    kernel_input = vectors_preparation(sp_internals, ori_internals,
                                       kriging_parameters)
    # Either the weights or directly the scalar field
    r = kernel_reduction(kernel_input)



def create_covariance(ki: KernelInput, kernel: Callable, kernel_1st: Callable,
                      kernel_2nd: Callable):
    dif_ref_ref = ki.dip_ref_i - ki.dip_ref_j
    dif_rest_rest = ki.diprest_i - ki.diprest_j

    hu = (dif_ref_ref * (ki.hu_sel_i * ki.hu_sel_j)).sum(axis=-1)
    hv = (dif_ref_ref * (ki.hv_sel_i * ki.hv_sel_j)).sum(axis=-1)

    r_ref_ref = np.sqrt((dif_ref_ref ** 2).sum(-1))
    r_rest_rest = np.sqrt((dif_rest_rest ** 2).sum(-1))
    r_ref_rest = np.sqrt(((ki.dip_ref_i - ki.diprest_j) ** 2).sum(-1))
    k_rest_rest = kernel(r_rest_rest)
    k_ref_ref = kernel(r_ref_ref)
    k_ref_rest = kernel(r_ref_rest)
    k_p_ref = kernel_1st(r_ref_ref)  # First derivative DIVIDED BY r C'(r)/r
    k_p_rest = kernel_1st(r_rest_rest)  # First derivative DIVIDED BY r C'(r)/r
    k_a = kernel_2nd(r_ref_ref)  # Second derivative of the kernel

    hu_ref = dif_ref_ref * (ki.hu_sel_i * ki.hu_sel_points_j)
    hv_ref = dif_ref_ref * (ki.hv_sel_points_i * ki.hv_sel_j)
    huv_ref = hu_ref.sum(axis=-1) - hv_ref.sum(axis=-1)

    hu_rest = dif_rest_rest * (ki.hu_sel_i * ki.hu_sel_points_j)
    hv_rest = dif_rest_rest * (ki.hv_sel_points_i * ki.hv_sel_j)
    huv_rest = hu_rest.sum(axis=-1) - hv_rest.sum(axis=-1)

    cov = hu * hv / r_ref_ref * (- k_p_ref + k_a) - k_p_ref + \
          k_rest_rest - 2 * k_ref_rest + k_ref_ref + \
          - huv_rest * k_p_rest + huv_ref * k_p_ref

    return cov


def kernel_reduction(ki: KernelInput):
    cov = create_covariance(ki)


def vectors_preparation(sp_internals: SurfacePointsInternals,
                        ori_internals: OrientationsInternals,
                        interpolation_options: InterpolationOptions
                        ):
    cov_size = ori_internals.n_orientations + sp_internals.n_points + interpolation_options.n_uni_eq
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
    return KernelInput(dipref_i, dipref_j, diprest_i, diprest_j, hu_sel_i, hu_sel_j,
                       hv_sel_i, hv_sel_j, hu_points_j, hv_points_i,
                       dips_ug_ai, dips_ug_aj, dips_ug_bi,
                       dips_ug_bj, dipsref_ui_ai, dipsref_ui_aj, dipsref_ui_bi1,
                       dips_uiref_bj1, dipsref_ui_bi2, dips_uiref_bj2, dipsrest_ui_ai,
                       dipsrest_ui_aj, dipsrest_ui_bi1, dips_uirest_bj1,
                       dipsrest_ui_bi2, dips_uirest_bj2)


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
    points_ia = points_a[:, None, :].copy()
    points_ja = points_a[None, :, :].copy()
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
    points_ib1 = points_b1[:, None, :].copy()
    points_jb1 = points_b1[None, :, :].copy()
    # ui_b1 = (points_0b1 * points_1b1).sum(axis=-1)

    points_b2 = np.vstack((z, surface_points, zc))
    points_ib2 = points_b2[:, None, :].copy()
    points_jb2 = points_b2[None, :, :].copy()
    return points_ia, points_ja, points_ib1, points_jb1, points_ib2, points_jb2


def input_ug(ori_internals: OrientationsInternals,
             sp_size: int,
             interpolation_options: InterpolationOptions):
    n_ori = ori_internals.n_orientations
    n_dim = interpolation_options.number_dimensions

    z = np.zeros((sp_size + interpolation_options.n_uni_eq,
                  interpolation_options.number_dimensions))
    z2 = z.copy()
    if interpolation_options.uni_degree != 0:
        # Degree 1
        for i in range(interpolation_options.number_dimensions):
            z[sp_size + i, i] = 1

    dips_a = np.vstack((ori_internals.dip_positions_tiled, z))

    dips_ia = dips_a[:, None, :].copy()
    dips_ja = dips_a[None, :, :].copy()

    dips_b_aux = ori_internals.dip_positions_tiled.copy()
    if interpolation_options.uni_degree == 2:
        # Degree 2

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
    dips_ib = dips_b[:, None, :].copy()
    dips_jb = dips_b[None, :, :].copy()
    return dips_ia, dips_ja, dips_ib, dips_jb


def hu_hv_sel(n_dips, n_dim, n_points, cov_size):
    sel_0 = np.zeros((cov_size, n_dim))
    sel_1 = np.zeros((cov_size, n_dim))
    sel_2 = np.zeros((cov_size, n_dim))
    for i in range(n_dim):
        sel_0[n_dips * i:n_dips * (i + 1), i] = 1
        sel_1[:n_dips * n_dim, :] = 1
    sel_2[n_dips * n_dim: n_dips * n_dim + n_points, :] = 1

    sel_hui = sel_0[:, None, :].copy()
    sel_huj = sel_1[None, :, :].copy()

    sel_hvi = sel_1[:, None, :].copy()
    sel_hvj = sel_0[None, :, :].copy()

    sel_hu_points_j = sel_2[None, :, :].copy()
    sel_hv_points_i = sel_2[:, None, :].copy()

    return sel_hui, sel_huj, sel_hvi, sel_hvj, sel_hu_points_j, sel_hv_points_i


def input_dips_points(dip_pos, points_pos, interpolations_options):
    z = np.zeros((interpolations_options.n_uni_eq,
                  interpolations_options.number_dimensions))

    dipspoints = np.vstack((dip_pos, points_pos, z))
    dips_points0 = dipspoints[:, None, :].copy()
    dips_points1 = dipspoints[None, :, :].copy()

    return dips_points0, dips_points1
