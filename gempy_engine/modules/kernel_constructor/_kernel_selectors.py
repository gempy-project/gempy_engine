import numpy as np

from gempy_engine.modules.kernel_constructor._structs import DriftMatrixSelector, CartesianSelector


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