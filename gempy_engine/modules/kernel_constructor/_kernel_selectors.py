import numpy as np

from gempy_engine.config import DEFAULT_DTYPE


def dips_sp_cartesian_selector(vector_size, n_dim, n_dips, n_points, axis: int = None):
    sel_hu = np.zeros((vector_size, n_dim), dtype=DEFAULT_DTYPE)
    sel_hv = np.zeros((vector_size, n_dim), dtype=DEFAULT_DTYPE)
    sel_huv = np.zeros((vector_size, n_dim), dtype=DEFAULT_DTYPE)
    for i in range(n_dim):
        sel_hu[n_dips * i:n_dips * (i + 1), i] = 1

    sel_hv[:n_dips * n_dim, :] = 1

    if axis is None:
        sel_huv[n_dips * n_dim: n_dips * n_dim + n_points, :] = 1
    else:
        sel_huv[n_dips * n_dim: n_dips * n_dim + n_points, axis] = 1

    return sel_hu, sel_hv, sel_huv


def grid_cartesian_selector(vector_size, n_dim, axis: int = None):
    sel_hu = np.ones((vector_size, n_dim), dtype=DEFAULT_DTYPE)
    sel_huv = np.ones((vector_size, n_dim), dtype=DEFAULT_DTYPE)

    if axis is None:
        sel_hv = np.ones((vector_size, n_dim), dtype=DEFAULT_DTYPE)
    else:
        sel_hv = np.zeros((vector_size, n_dim), dtype=DEFAULT_DTYPE)
        sel_hv[:, axis] = 1

    return sel_hu, sel_hv, sel_huv
