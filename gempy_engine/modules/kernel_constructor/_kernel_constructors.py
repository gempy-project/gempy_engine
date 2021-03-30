from typing import Tuple

import numpy as np

from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.kernel_constructor._structs import SurfacePointsInternals, OrientationSurfacePointsCoords, \
    CartesianSelector, OrientationsInternals, OrientationsDrift, PointsDrift, DriftMatrixSelector


def assembly_dips_surface_points_coord_matrix(dips_coord: np.ndarray, sp_coord: SurfacePointsInternals,
                                              options: InterpolationOptions) -> OrientationSurfacePointsCoords:
    dips_points0a, dips_points1a = _assembly(dips_coord, options, sp_coord.ref_surface_points)
    dips_points0b, dips_points1b = _assembly(dips_coord, options, sp_coord.rest_surface_points)

    return OrientationSurfacePointsCoords(dips_points0a, dips_points1a, dips_points0b, dips_points1b)


def hu_hv_sel(n_dips, n_dim, n_points, cov_size) -> CartesianSelector:
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

    return CartesianSelector(sel_hui, sel_huj, sel_hvi, sel_hvj, sel_hu_points_j, sel_hv_points_i)


def input_ug(ori_internals: OrientationsInternals, sp_size: int, interpolation_options: InterpolationOptions) \
        -> OrientationsDrift:
    n_ori = ori_internals.n_orientations
    n_dim = interpolation_options.number_dimensions

    z = np.zeros((n_ori * n_dim + sp_size + interpolation_options.n_uni_eq,
                  interpolation_options.number_dimensions))

    if interpolation_options.uni_degree != 0:
        # Degree 1
        for i in range(interpolation_options.number_dimensions):
            z[n_ori * i:n_ori * (i + 1)] = 1
            z[-interpolation_options.n_uni_eq + i, i] = 1

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

    dips_b = np.vstack((dips_b_aux, z2))
    dips_ib = dips_b[:, None, :]
    dips_jb = dips_b[None, :, :]
    return OrientationsDrift(dips_ia, dips_ja, dips_ib, dips_jb)


def input_usp(surface_points: SurfacePointsInternals, ori_size: int,
              interpolation_options: InterpolationOptions) -> PointsDrift:
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

    points_a = np.vstack((z, surface_points, z2))
    points_ia = points_a[:, None, :]
    points_ja = points_a[None, :, :]

    # Degree 2
    if interpolation_options.uni_degree == 2:
        for i in range(n_dim):
            zb[n_dim + i, i] = 1
        zb[n_dim * 2:, 0] = 1

        for i in range(n_dim):
            zc[n_dim + i, i] = 1
        zc[n_dim * 2:, 1] = 1

    points_b1 = np.vstack((z, surface_points, zb))
    points_ib1 = points_b1[:, None, :]
    points_jb1 = points_b1[None, :, :]

    points_b2 = np.vstack((z, surface_points, zc))
    points_ib2 = points_b2[:, None, :]
    points_jb2 = points_b2[None, :, :]
    return PointsDrift(points_ia, points_ja, points_ib1, points_jb1, points_ib2, points_jb2)


def drift_selector(cov_size: int, n_drift_eq: int) -> DriftMatrixSelector:
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
    return DriftMatrixSelector(sel_ui, sel_uj, sel_vi, sel_vj)


def _assembly(dips_coord: np.ndarray, options: InterpolationOptions, sp_coord: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    z = np.zeros((options.n_uni_eq, options.number_dimensions))
    dipspoints = np.vstack((dips_coord, sp_coord, z))
    dips_points0 = dipspoints[:, None, :]
    dips_points1 = dipspoints[None, :, :]
    return dips_points0, dips_points1
