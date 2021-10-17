from typing import Tuple

from ...config import DEFAULT_DTYPE
from ...core.backend_tensor import BackendTensor
from ...core.data.options import InterpolationOptions
from ...core.data.kernel_classes.orientations import OrientationsInternals
import numpy as np

tfnp = BackendTensor.tfnp
tensor_types = BackendTensor.tensor_types

def assembly_dips_points_tensor(dips_coord: tensor_types, sp_coord: tensor_types, options: InterpolationOptions):
    z = tfnp.zeros((options.n_uni_eq, options.number_dimensions))

    dipspoints = tfnp.concat((dips_coord, sp_coord, z), axis=0)
    return dipspoints


def assembly_dips_ug_coords(ori_internals: OrientationsInternals, sp_size: int,
                            interpolation_options: InterpolationOptions) \
        -> Tuple[tensor_types, tensor_types,tensor_types,tensor_types]:
    n_ori = ori_internals.n_orientations
    n_dim = interpolation_options.number_dimensions

    full_cov_size = n_ori * n_dim + sp_size + interpolation_options.n_uni_eq
    z = np.zeros((full_cov_size, interpolation_options.number_dimensions), dtype=DEFAULT_DTYPE)

    # Assembly vector for degree 1
    if interpolation_options.uni_degree != 0:
        # Degree 1
        for i in range(interpolation_options.number_dimensions):
            z[n_ori * i:n_ori * (i + 1), i] = 1
            z[-interpolation_options.n_uni_eq + i, i] = 1

    dips_a = z

    # Degree 2
    # TODO: Refactor degree 2
        # Second term:
    dips_b_aux = ori_internals.dip_positions_tiled

    # TODO [X]: Now we have that vstack below. We have to create the proper array here
    z2 = np.zeros((full_cov_size, interpolation_options.number_dimensions), dtype=DEFAULT_DTYPE)

    shift = n_ori * n_dim +  sp_size
    if interpolation_options.uni_degree == 2:
        for i in range(interpolation_options.number_dimensions):
            z2[n_ori * i:n_ori * (i + 1), i] = dips_b_aux[n_ori * i:n_ori * (i + 1), i]
            #z2[n_ori * i:n_ori * (i + 1), :] = dips_b_aux[n_ori * i:n_ori * (i + 1), :]

            #dips_b_aux[n_ori * i:n_ori * (i + 1), i] = 0
            z2[shift + n_dim + i, i] = 2
            # z2[shift + n_dim * 2 + i] = 1
            # z2[shift + n_dim * 2 + i, n_dim - i - 1] = 0

    dips_b = z2 #np.vstack((dips_b_aux, z2))


        # Third term:
    z3 = np.zeros((full_cov_size, interpolation_options.number_dimensions), dtype=DEFAULT_DTYPE)
    uni_second_degree_selector = np.zeros_like(z3, dtype=DEFAULT_DTYPE)

    if interpolation_options.uni_degree == 2:
        for i in range(interpolation_options.number_dimensions):
            #z3[n_ori * i:n_ori * (i + 1), i] = dips_b_aux[n_ori * i:n_ori * (i + 1), i]
            z3[n_ori * i:n_ori * (i + 1), :] = dips_b_aux[n_ori * i:n_ori * (i + 1), :]
            z3[n_ori * i:n_ori * (i + 1), i] = 0

            #uni_second_degree_selector[n_ori * i :n_ori * (i + 1), i] = 1
            uni_second_degree_selector[n_ori * i:n_ori * (i + 1), :] = 1
            uni_second_degree_selector[n_ori * i:n_ori * (i + 1), i] = 0

            #z3[shift + n_dim + i, i] = 2
            z3[shift + n_dim * 2 + i] = 1
            z3[shift + n_dim * 2 + i, n_dim - i - 1] = 0

            uni_second_degree_selector[shift + n_dim * 2 + i] = 1
            uni_second_degree_selector[shift + n_dim * 2 + i, n_dim - i - 1] = 0


    dips_c = z3 #np.vstack((dips_b_aux, z2))

    return dips_a, dips_b, dips_c, uni_second_degree_selector


def assembly_dips_points_coords(surface_points: tensor_types, ori_size: int,
                                interpolation_options: InterpolationOptions) \
        -> Tuple[tensor_types,tensor_types,tensor_types]:
    n_dim = interpolation_options.number_dimensions

    z = np.zeros((ori_size, n_dim), dtype=DEFAULT_DTYPE)
    z2 = np.zeros((interpolation_options.n_uni_eq, n_dim), dtype=DEFAULT_DTYPE)

    zb = np.zeros((interpolation_options.n_uni_eq, n_dim), dtype=DEFAULT_DTYPE)#z2.copy()
    zc = np.zeros((interpolation_options.n_uni_eq, n_dim), dtype=DEFAULT_DTYPE)#z2.copy()

    if interpolation_options.uni_degree != 0:
        for i in range(interpolation_options.number_dimensions):
            z2[i, i] = 1

    # Degree 1
    points_degree_1 = tfnp.concat((z, surface_points, z2), axis=0)
    #points_degree_1 = tfnp.vstack((z, surface_points, z2))

    # Degree 2
    # TODO: Substitute vstack

    if interpolation_options.uni_degree == 2:
        for i in range(n_dim):
            zb[n_dim + i, i] = 1


        zb[n_dim * 2, 0] = 1
        zb[n_dim * 2 +1, 0] = 1
        zb[n_dim * 2 + 2, 1] = 1


        for i in range(n_dim):
            zc[n_dim + i, i] = 1

        zc[n_dim * 2, 1] = 1
        zc[n_dim * 2 + 1, 2] = 1
        zc[n_dim * 2 + 2, 2] = 1


    # points_degree_2a = tfnp.vstack((z, surface_points, zb))
    # points_degree_2b = tfnp.vstack((z, surface_points, zc))
    points_degree_2a = tfnp.concat((z, surface_points, zb), axis=0)
    points_degree_2b = tfnp.concat((z, surface_points, zc), axis=0)

    return points_degree_1, points_degree_2a, points_degree_2b
