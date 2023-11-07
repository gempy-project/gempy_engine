from typing import Tuple

import numpy as np

from ...core.backend_tensor import BackendTensor
from ...core.data.matrices_sizes import MatricesSizes
from ...core.data.kernel_classes.orientations import OrientationsInternals
from ...core.data.options import KernelOptions
from ...core.backend_tensor import BackendTensor as bt


def assembly_dips_points_tensor(dips_coord: np.ndarray, sp_coord: np.ndarray, matrices_size: MatricesSizes):
    n_dim = matrices_size.dim
    drift_size = matrices_size.drifts_size
    z = bt.t.zeros((drift_size, n_dim), dtype=BackendTensor.dtype_obj)
    
    dipspoints = bt.t.vstack((dips_coord, sp_coord, z))
    return dipspoints


def assembly_dips_ug_coords(ori_internals: OrientationsInternals, interpolation_options: KernelOptions,
                            matrices_size: MatricesSizes) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_ori = ori_internals.n_orientations
    n_dim = matrices_size.dim
    full_cov_size = matrices_size.cov_size
    shift = matrices_size.sp_size + matrices_size.ori_size

    z = np.zeros((full_cov_size, n_dim), dtype=BackendTensor.dtype)

    # Assembly vectors for degree 1
    if interpolation_options.uni_degree != 0:
        # Degree 1
        for i in range(n_dim):
            z[n_ori * i:n_ori * (i + 1), i] = 1
            z[shift + i, i] = 1

    dips_a = z

    # region Degree 2
    # region Second term:
    dips_b_aux = ori_internals.dip_positions_tiled

    z2 = np.zeros((full_cov_size, n_dim), dtype=BackendTensor.dtype)

    if interpolation_options.uni_degree == 2:
        for i in range(interpolation_options.number_dimensions):
            z2[n_ori * i:n_ori * (i + 1), i] = dips_b_aux[n_ori * i:n_ori * (i + 1), i]
            z2[shift + n_dim + i, i] = 2

    dips_b = z2
    # endregion

    # region Third term:
    z3 = np.zeros((full_cov_size, interpolation_options.number_dimensions), dtype=BackendTensor.dtype)
    uni_second_degree_selector = np.zeros_like(z3, dtype=BackendTensor.dtype)

    if interpolation_options.uni_degree == 2:
        for i in range(interpolation_options.number_dimensions):
            z3[n_ori * i:n_ori * (i + 1), :] = dips_b_aux[n_ori * i:n_ori * (i + 1), :]
            z3[n_ori * i:n_ori * (i + 1), i] = 0

            uni_second_degree_selector[n_ori * i:n_ori * (i + 1), :] = 1
            uni_second_degree_selector[n_ori * i:n_ori * (i + 1), i] = 0

            z3[shift + n_dim * 2 + i] = 1
            z3[shift + n_dim * 2 + i, n_dim - i - 1] = 0

            uni_second_degree_selector[shift + n_dim * 2 + i] = 1
            uni_second_degree_selector[shift + n_dim * 2 + i, n_dim - i - 1] = 0

    dips_c = z3

    # endregion
    # endregion

    return dips_a, dips_b, dips_c, uni_second_degree_selector


def assembly_dips_points_coords(surface_points: np.ndarray, matrices_sizes: MatricesSizes,
                                interpolation_options: KernelOptions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_dim = interpolation_options.number_dimensions
    ori_size = matrices_sizes.ori_size
    drifts_size = matrices_sizes.drifts_size
    
    z = bt.t.zeros((ori_size, n_dim), dtype=BackendTensor.dtype_obj)  # * Orientations area
    z2 = bt.t.zeros((drifts_size, n_dim), dtype=BackendTensor.dtype_obj)  # * Universal area

    zb = bt.t.copy(z2)  # ! This block has to be here because it has to be before we modify z2
    zc = bt.t.copy(z2)

    if interpolation_options.uni_degree != 0:
        for i in range(n_dim):
            z2[i, i] = 1

    # Degree 1
    points_degree_1 = bt.t.vstack((z, surface_points, z2))

    # Degree 2
    # TODO: Substitute vstack
    if interpolation_options.uni_degree == 2:

        for i in range(n_dim):
            zb[n_dim + i, i] = 1

        zb[n_dim * 2, 0] = 1
        zb[n_dim * 2 + 1, 0] = 1
        zb[n_dim * 2 + 2, 1] = 1

        for i in range(n_dim):
            zc[n_dim + i, i] = 1
        # zc[n_dim * 2:, 1] = 1
        zc[n_dim * 2, 1] = 1
        zc[n_dim * 2 + 1, 2] = 1
        zc[n_dim * 2 + 2, 2] = 1

    points_degree_2a = bt.t.vstack((z, surface_points, zb))
    points_degree_2b = bt.t.vstack((z, surface_points, zc))

    return points_degree_1, points_degree_2a, points_degree_2b
