from typing import Tuple

import numpy as np

from ...core.backend_tensor import BackendTensor
from ...core.data.matrices_sizes import MatricesSizes
from ...core.data.kernel_classes.orientations import OrientationsInternals
from ...core.data.options import KernelOptions
from ...core.backend_tensor import BackendTensor as bt
from ...core.host_arrays import upload


def assembly_dips_points_tensor(dips_coord: np.ndarray, sp_coord: np.ndarray, matrices_size: MatricesSizes):
    n_dim = matrices_size.dim
    drift_size = matrices_size.drifts_size
    z = bt.t.zeros((drift_size, n_dim), dtype=BackendTensor.dtype_obj)
    
    dipspoints = bt.t.vstack((dips_coord, sp_coord, z))
    return dipspoints


def assembly_dips_ug_coords(ori_internals: OrientationsInternals, interpolation_options: KernelOptions,
                            matrices_size: MatricesSizes) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_dim = matrices_size.dim
    full_cov_size = matrices_size.cov_size
    degree = interpolation_options.uni_degree

    # The 0/1/2 patterns depend only on sizes: built on the host and uploaded at once. Rows of the
    # second- and third-term masks take the dip positions instead.
    patterns = upload(ug_host_patterns(ori_internals.n_orientations, matrices_size, degree))
    dips_a, dips_b, dips_c, uni_second_degree_selector = patterns[0], patterns[1], patterns[3], patterns[5]

    if degree == 2:
        dips_b_aux = ori_internals.dip_positions_tiled
        padding = BackendTensor.t.zeros((full_cov_size - dips_b_aux.shape[0], n_dim), dtype=BackendTensor.dtype_obj)
        positions = BackendTensor.t.vstack((dips_b_aux, padding))  # rows aligned with the covariance rows
        dips_b = BackendTensor.t.where(patterns[2] > 0, positions, dips_b)
        dips_c = BackendTensor.t.where(patterns[4] > 0, positions, dips_c)

    return dips_a, dips_b, dips_c, uni_second_degree_selector


def ug_host_patterns(n_ori: int, matrices_size: MatricesSizes, degree: int) -> np.ndarray:
    """(6, cov_size, dim): degree-1 term, second term and its position mask, third term and its position
    mask, second-degree selector of ``assembly_dips_ug_coords``."""
    n_dim = matrices_size.dim
    shift = matrices_size.sp_size + matrices_size.ori_size
    a, b, b_mask, c, c_mask, selector = patterns = np.zeros((6, matrices_size.cov_size, n_dim), dtype=BackendTensor.dtype)
    if degree != 0:
        # Degree 1
        for i in range(n_dim):
            a[n_ori * i:n_ori * (i + 1), i] = 1
            a[shift + i, i] = 1
    if degree == 2:
        for i in range(n_dim):
            rows = slice(n_ori * i, n_ori * (i + 1))
            # Second term
            b_mask[rows, i] = 1
            b[shift + n_dim + i, i] = 2
            # Third term
            c_mask[rows, :] = 1
            c_mask[rows, i] = 0
            selector[rows, :] = 1
            selector[rows, i] = 0
            c[shift + n_dim * 2 + i] = 1
            c[shift + n_dim * 2 + i, n_dim - i - 1] = 0
            selector[shift + n_dim * 2 + i] = 1
            selector[shift + n_dim * 2 + i, n_dim - i - 1] = 0
    return patterns


def assembly_dips_points_coords(surface_points: np.ndarray, matrices_sizes: MatricesSizes,
                                interpolation_options: KernelOptions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_dim = interpolation_options.number_dimensions
    ori_size = matrices_sizes.ori_size
    drifts_size = matrices_sizes.drifts_size

    z = bt.t.zeros((ori_size, n_dim), dtype=BackendTensor.dtype_obj)  # * Orientations area
    universal = upload(universal_host_rows(interpolation_options, drifts_size))

    points_degree_1 = bt.t.vstack((z, surface_points, universal[0]))
    points_degree_2a = bt.t.vstack((z, surface_points, universal[1]))
    points_degree_2b = bt.t.vstack((z, surface_points, universal[2]))

    return points_degree_1, points_degree_2a, points_degree_2b


def universal_host_rows(interpolation_options: KernelOptions, drifts_size: int) -> np.ndarray:
    """(3, drifts_size, dim): the drift rows of ``assembly_dips_points_coords``, degree 1 (z2) and the two
    degree-2 factors (zb, zc)."""
    n_dim = interpolation_options.number_dimensions
    # * Universal area: degree 1 (z2) and the two degree-2 factors (zb, zc), built on the host
    z2, zb, zc = rows = np.zeros((3, drifts_size, n_dim), dtype=BackendTensor.dtype)
    if interpolation_options.uni_degree != 0:
        for i in range(n_dim):
            z2[i, i] = 1

    if interpolation_options.uni_degree == 2:
        for i in range(n_dim):
            zb[n_dim + i, i] = 1

        zb[n_dim * 2, 0] = 1
        zb[n_dim * 2 + 1, 0] = 1
        zb[n_dim * 2 + 2, 1] = 1

        for i in range(n_dim):
            zc[n_dim + i, i] = 1
        zc[n_dim * 2, 1] = 1
        zc[n_dim * 2 + 1, 2] = 1
        zc[n_dim * 2 + 2, 2] = 1
    return rows
