import numpy as np

from gempy_engine.modules.kernel_constructor._structs import DriftMatrixSelector, CartesianSelector


def drift_selector(x_size: int, y_size: int, n_drift_eq: int) -> DriftMatrixSelector:
    sel_i = np.zeros((x_size, 2))
    sel_j = np.zeros((y_size, 2))

    sel_i[:-n_drift_eq, 0] = 1
    sel_i[-n_drift_eq:, 1] = 1
    sel_j[-n_drift_eq:, :] = 1

    sel_ui = sel_i[:, None, :]
    sel_uj = sel_j[None, :, :]

    sel_vi = -sel_j[:, None, :]
    sel_vj = -sel_i[None, :, :]
    a = sel_ui * (sel_vj + 1)
    return DriftMatrixSelector(sel_ui, sel_uj, sel_vi, sel_vj)


def hu_hv_sel(x_sel_hu, y_sel_hu, x_sel_hv, y_sel_hv, x_sel_hv_points, y_sel_hv_points) -> CartesianSelector:

    sel_hui = x_sel_hu[:, None, :]
    sel_huj = y_sel_hu[None, :, :]

    sel_hvi = x_sel_hv[:, None, :]
    sel_hvj = y_sel_hv[None, :, :]

    sel_hu_points_i = x_sel_hv_points[:, None, :]
    sel_hu_points_j = y_sel_hv_points[None, :, :]

    return CartesianSelector(sel_hui, sel_huj, sel_hvi, sel_hvj, sel_hu_points_i, sel_hu_points_j)


def dips_sp_cartesian_selector(vector_size, n_dim, n_dips, n_points):
    sel_hu = np.zeros((vector_size, n_dim))
    sel_hv = np.zeros((vector_size, n_dim))
    sel_huv = np.zeros((vector_size, n_dim))
    for i in range(n_dim):
        sel_hu[n_dips * i:n_dips * (i + 1), i] = 1
        sel_hv[:n_dips * n_dim, :] = 1

    sel_huv[n_dips * n_dim: n_dips * n_dim + n_points, :] = 1

    return sel_hu, sel_hv, sel_huv

def grid_cartesian_selector(vector_size, n_dim):
    sel_hu = np.ones((vector_size, n_dim))
    sel_hv = np.ones((vector_size, n_dim))
    sel_huv = np.ones((vector_size, n_dim))

    return sel_hu, sel_hv, sel_huv