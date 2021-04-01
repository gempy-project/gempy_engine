from typing import Tuple

import numpy as np

from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.modules.kernel_constructor._structs import OrientationSurfacePointsCoords, \
    OrientationsDrift, PointsDrift
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePointsInternals
from gempy_engine.core.data.kernel_classes.orientations import OrientationsInternals


def assembly_core_tensor(x_ref: np.ndarray, y_ref: np.ndarray, x_rest: np.ndarray, y_rest: np.ndarray) -> \
        OrientationSurfacePointsCoords:
    def _assembly(x, y) -> Tuple[np.ndarray, np.ndarray]:
        dips_points0 = x[:, None, :]  # i
        dips_points1 = y[None, :, :]  # j
        return dips_points0, dips_points1

    dips_points_ref_i, dips_points_ref_j = _assembly(x_ref, y_ref)
    dips_points_rest_i, dips_points_rest_j = _assembly(x_rest, y_rest)

    return OrientationSurfacePointsCoords(dips_points_ref_i, dips_points_ref_j, dips_points_rest_i, dips_points_rest_j)


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


def assembly_ug_tensor(x_degree_1: np.ndarray, y_degree_1: np.ndarray, x_degree_2: np.ndarray, y_degree_2: np.ndarray):
    return OrientationsDrift(
        x_degree_1[:, None, :],
        y_degree_1[None, :, :],
        x_degree_2[:, None, :],
        y_degree_2[None, :, :]
    )


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


def assembly_usp_tensor(x_degree_1: np.ndarray, y_degree_1: np.ndarray, x_degree_2a: np.ndarray,
                        y_degree_2a: np.ndarray, x_degree_2b: np.ndarray, y_degree_2b: np.ndarray):
    return PointsDrift(
        x_degree_1[:, None, :],
        y_degree_1[None, :, :],
        x_degree_2a[:, None, :],
        y_degree_2a[None, :, :],
        x_degree_2b[:, None, :],
        y_degree_2b[None, :, :],
    )
