import numpy as np

from ...core.data.matrices_sizes import MatricesSizes


def dips_sp_cartesian_selector(matrices_sizes: MatricesSizes, axis: int = None):
    vector_size = matrices_sizes.cov_size
    n_dim = matrices_sizes.dim
    n_dips = matrices_sizes.n_dips
    n_points = matrices_sizes.sp_size

    sel_hu = np.zeros((vector_size, n_dim), dtype="int8")
    sel_hv = np.zeros((vector_size, n_dim), dtype="int8")
    sel_huv = np.zeros((vector_size, n_dim), dtype="int8")
    
    for i in range(n_dim): # ! sel_hu has to be the same also for all axis 
        sel_hu[n_dips * i:n_dips * (i + 1), i] = 1

    sel_hv[:n_dips * n_dim, :] = 1

    if axis is None:
        sel_huv[n_dips * n_dim: n_dips * n_dim + n_points, :] = 1
    else:
        sel_huv[n_dips * n_dim: n_dips * n_dim + n_points, axis] = 1
    
    return sel_hu, sel_hv, sel_huv


def grid_cartesian_selector(matrices_sizes: MatricesSizes, axis: int = None):
    vector_size = matrices_sizes.grid_size
    n_dim = matrices_sizes.dim
    
    sel_hu = np.ones((vector_size, n_dim), dtype="int8")
    sel_huv = np.ones((vector_size, n_dim), dtype="int8")

    if axis is None:
        sel_hv = np.ones((vector_size, n_dim), dtype="int8")
    else:
        sel_hv = np.zeros((vector_size, n_dim), dtype="int8")
        sel_hv[:, axis] = 1

    return sel_hu, sel_hv, sel_huv
