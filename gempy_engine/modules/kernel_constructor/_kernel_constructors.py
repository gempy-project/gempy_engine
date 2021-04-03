from typing import Tuple

import numpy as np

from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.kernel_classes.orientations import OrientationsInternals


def assembly_dips_points_tensor(dips_coord: np.ndarray, sp_coord: np.ndarray, options: InterpolationOptions):
    z = np.zeros((options.n_uni_eq, options.number_dimensions))
    dipspoints = np.vstack((dips_coord, sp_coord, z))
    return dipspoints


def assembly_dips_ug_coords(ori_internals: OrientationsInternals, sp_size: int,
                            interpolation_options: InterpolationOptions) \
        -> Tuple[np.ndarray, np.ndarray]:
    n_ori = ori_internals.n_orientations
    n_dim = interpolation_options.number_dimensions

    full_cov_size = n_ori * n_dim + sp_size + interpolation_options.n_uni_eq
    z = np.zeros((full_cov_size, interpolation_options.number_dimensions))

    # Assembly vector for degree 1
    if interpolation_options.uni_degree != 0:
        # Degree 1
        for i in range(interpolation_options.number_dimensions):
            z[n_ori * i:n_ori * (i + 1)] = 1
            z[-interpolation_options.n_uni_eq + i, i] = 1

    dips_a = z

    # Degree 2
    dips_b_aux = ori_internals.dip_positions_tiled
    z2 = np.zeros((interpolation_options.n_uni_eq, interpolation_options.number_dimensions))

    if interpolation_options.uni_degree == 2:
        for i in range(interpolation_options.number_dimensions):
            dips_b_aux[n_ori * i:n_ori * (i + 1), i] = 0
            z2[sp_size + n_dim + i, i] = 2
            z2[sp_size + n_dim * 2 + i] = 1
            z2[sp_size + n_dim * 2 + i, i] = 0

    dips_b = np.vstack((dips_b_aux, z2))

    return dips_a, dips_b


def assembly_dips_points_coords(surface_points: np.ndarray, ori_size: int,
                                interpolation_options: InterpolationOptions) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_dim = interpolation_options.number_dimensions

    z = np.zeros((ori_size, n_dim))
    z2 = np.zeros((interpolation_options.n_uni_eq, n_dim))

    zb = z2.copy()
    zc = z2.copy()

    if interpolation_options.uni_degree != 0:
        for i in range(interpolation_options.number_dimensions):
            z2[i, i] = 1

    # Degree 1
    points_degree_1 = np.vstack((z, surface_points, z2))

    # Degree 2
    if interpolation_options.uni_degree == 2:
        for i in range(n_dim):
            zb[n_dim + i, i] = 1
        zb[n_dim * 2:, 0] = 1

        for i in range(n_dim):
            zc[n_dim + i, i] = 1
        zc[n_dim * 2:, 1] = 1

    points_degree_2a = np.vstack((z, surface_points, zb))
    points_degree_2b = np.vstack((z, surface_points, zc))

    return points_degree_1, points_degree_2a, points_degree_2b
