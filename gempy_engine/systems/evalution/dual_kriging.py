from gempy_engine.config import tfnp, tensorflow_imported, tensor_types, pykeops_imported
import numpy as np

from gempy_engine.data_structures.private_structures import OrientationsInternals, SurfacePointsInternals, ExportInput
from gempy_engine.data_structures.public_structures import InterpolationOptions
from gempy_engine.systems.kernel.kernel import get_kernels, input_usp


def export(sp_internals: SurfacePointsInternals,
           ori_internals: OrientationsInternals,
           interpolations_options: InterpolationOptions, grid):
    # TODO this function can be split into two

    cov_size = sp_internals.n_points + \
               ori_internals.n_orientations_tiled + \
               interpolations_options.n_uni_eq
    direction_val = get_direction_val('x')

    ei = vector_preparation_export(sp_internals, ori_internals,
                                   interpolations_options, grid, cov_size)

    hu = hu_x0(ori_internals, interpolations_options, grid, cov_size)
    hv = hv_x0(ori_internals, interpolations_options, grid,
               direction_val, cov_size)

    k, k1, k2 = get_kernels('cubic')
    perp_v = compute_perep_v(
        ori_internals.n_orientations,
        cov_size,
        direction_val
    )
    hu_rest = hu_points(sp_internals.rest_surface_points, grid, direction_val)
    hu_ref = hu_points(sp_internals.ref_surface_points, grid, direction_val)

    # TODO maybe create a class for these
    drift_sp = prepare_usp(grid, cov_size, interpolations_options)
    drift_g = prepare_ug(grid, cov_size, direction_val,
                         interpolations_options)

    z_0 = export_scalar(ei, hu, drift_sp, k1, interpolations_options)
    dz_dx = export_gradients(ei, hu, hv, hu_ref, hu_rest, drift_g,
                             k1, k2, perp_v,
                             interpolations_options.range)

    return z_0, dz_dx


def export_scalar(ei: ExportInput, hu_x0, drift, kernel_1st, interpolation_options,
                  ):

    a = interpolation_options.range

    if pykeops_imported is True:
        r_dip_x0 = (((ei.dips_i - ei.grid_j) ** 2).sum(-1)).sqrt()
    else:
        r_dip_x0 = tfnp.sqrt(((ei.dips_i - ei.grid_j) ** 2).sum(-1))
    k_p_grad = kernel_1st(r_dip_x0, a)

    sigma_0_grad_interface = hu_x0 * k_p_grad

    if pykeops_imported is True:
        r_ref_grid = (((ei.ref_i - ei.grid_j) ** 2).sum(-1)).sqrt()
        r_rest_grid = (((ei.rest_i - ei.grid_j) ** 2).sum(-1)).sqrt()

    else:
        r_ref_grid = tfnp.sqrt(((ei.ref_i - ei.grid_j) ** 2).sum(-1))
        r_rest_grid = tfnp.sqrt(((ei.rest_i - ei.grid_j) ** 2).sum(-1))

    k_ref_x0 = kernel_1st(r_ref_grid, a)
    k_rest_x0 = kernel_1st(r_rest_grid, a)

    sigma_0_interf = k_rest_x0 - k_ref_x0

    return sigma_0_grad_interface + sigma_0_interf + drift


def export_gradients(ei: ExportInput, hu, hv, hu_ref, hu_rest, drift_g,
                     kernel_1st, kernel_2nd, perp_v, a):
    if pykeops_imported is True:
        r_dip_x0 = (((ei.dips_i - ei.grid_j) ** 2).sum(-1)).sqrt()
        r_ref_x0 = (((ei.ref_i - ei.grid_j) ** 2).sum(-1)).sqrt()
        r_rest_x0 = (((ei.rest_i - ei.grid_j) ** 2).sum(-1)).sqrt()
    else:
        r_dip_x0 = tfnp.sqrt(((ei.dips_i - ei.grid_j) ** 2).sum(-1))
        r_ref_x0 = tfnp.sqrt(((ei.ref_i - ei.grid_j) ** 2).sum(-1))
        r_rest_x0 = tfnp.sqrt(((ei.rest_i - ei.grid_j) ** 2).sum(-1))

    k_p_dip_x0 = kernel_1st(r_dip_x0, a)
    k_a_dip_x0 = kernel_2nd(r_dip_x0, a)

    sigma_0_grad = hu * hv / (r_dip_x0 ** 2 + 1e-5) * (- k_p_dip_x0 + k_a_dip_x0) - \
                   k_p_dip_x0 * perp_v

    # ===
    # Euclidian distances
    k_p_ref = kernel_1st(r_ref_x0)
    k_p_rest = kernel_1st(r_rest_x0)

    sigma_0_sp_grad = hu_rest * k_p_rest - hu_ref * k_p_ref

    return sigma_0_grad + sigma_0_sp_grad + drift_g


def vector_preparation_export(sp_internals: SurfacePointsInternals,
                              ori_internals: OrientationsInternals,
                              interpolations_options: InterpolationOptions, grid,
                              cov_size):
    dips_init = tfnp.zeros((cov_size, interpolations_options.number_dimensions))
    ref_init = tfnp.zeros((cov_size, interpolations_options.number_dimensions))
    rest_init = tfnp.zeros((cov_size, interpolations_options.number_dimensions))
    dips = ori_internals.dip_positions_tiled
    dips_init[:ori_internals.n_orientations_tiled] = dips

    dips_i = dips_init[:, None, :]
    grid_j = grid[None, :, :]

    s1 = ori_internals.n_orientations_tiled
    s2 = s1 + sp_internals.n_points

    ref_init[s1:s2] = sp_internals.ref_surface_points
    rest_init[s1:s2] = sp_internals.rest_surface_points

    ref_i = ref_init[:, None, :]
    rest_i = rest_init[:, None, :]

    ei = ExportInput(dips_i, grid_j, ref_i, rest_i)
    return ei


def hu_x0(ori_internals: OrientationsInternals,
          interpolations_options,
          grid, cov_size):
    n_dim = interpolations_options.number_dimensions
    n_dips = ori_internals.n_orientations

    dips = ori_internals.ori_input.dip_positions.reshape((-1, 1), order="F")
    dips_x0 = tfnp.zeros((cov_size, 1), dtype='float64')

    sel_0 = np.zeros((cov_size, n_dim))
    for i in range(n_dim):
        sel_0[n_dips * i:n_dips * (i + 1), i] = 1
    sel_hui = sel_0[:, None, :]

    dips_x0[:ori_internals.n_orientations_tiled] = dips
    dips_i = dips_x0[:, None, :]

    grid_j = grid[None, :, :]
    hu_x0 = ((dips_i - grid_j) * sel_hui).sum(axis=-1)

    return hu_x0


def hv_x0(ori_internals: OrientationsInternals,
          interpolations_options: InterpolationOptions, grid,
          direction_val, cov_size):
    n_dim = interpolations_options.number_dimensions
    n_dips = ori_internals.n_orientations

    dips = tfnp.tile(
        ori_internals.ori_input.dip_positions[:, direction_val],
        interpolations_options.number_dimensions
    ).reshape((-1, 1))

    dips_x0 = tfnp.zeros((cov_size, 1), dtype='float64')
    grid_ = grid[:, direction_val].reshape((-1, 1))

    sel_0 = np.zeros((cov_size, n_dim))
    for i in range(n_dim):
        sel_0[n_dips * i:n_dips * (i + 1), i] = 1
    sel_hui = sel_0[:, None, :]

    dips_x0[:ori_internals.n_orientations_tiled] = dips
    dips_i = dips_x0[:, None, :]

    grid_j = grid_[None, :, :]
    hv_x0 = ((dips_i - grid_j) * sel_hui).sum(axis=-1)
    return hv_x0


def hu_points(points, grid, dir_val):
    return points[:, dir_val] + grid[:, dir_val].reshape((grid[:, dir_val].shape[0], 1))


def get_direction_val(direction: str):
    if direction == 'x':
        direction_val = 0
    elif direction == 'y':
        direction_val = 1
    elif direction == 'z':
        direction_val = 2
    else:
        raise AttributeError('direction must be x, y, z')
    return direction_val


def prepare_ug(grid, cov_size, direction_val: int,
               interpolation_options: InterpolationOptions):

    n_dim = interpolation_options.number_dimensions
    n_eq = interpolation_options.n_uni_eq
    i = direction_val
    z = np.zeros((cov_size,
                  interpolation_options.number_dimensions))

    z2 = np.zeros((cov_size,
                  interpolation_options.number_dimensions))

    if interpolation_options.uni_degree != 0:
        # Degree 1
        z[-n_eq + i, i] = 1

    drift_i = z[:, None, :]
    grid_j = grid[None, :, :]

    drift_1 = (drift_i * grid_j).sum(-1)
    print(drift_1)

    if interpolation_options.uni_degree == 2:
        z2[-n_eq + n_dim + i, i] = 2
        z2[-n_eq + n_dim * 2 + i] = 1
        z2[-n_eq + n_dim * 2 + i, i] = 0

    drift_2_i = z2[:, None, :]

    drift_2 = (drift_2_i * grid_j).sum(-1)
    print(drift_2)

    return drift_1 + drift_2


def prepare_usp(grid, cov_size, interpolation_options: InterpolationOptions):
    drift_init = tfnp.zeros((cov_size, interpolation_options.number_dimensions))
    u_eq = interpolation_options.n_uni_eq
    n_dim = interpolation_options.number_dimensions

    if interpolation_options.uni_degree != 0:
        for i in range(interpolation_options.number_dimensions):
            drift_init[-u_eq + i, i] = 1
    drift_i = drift_init[:, None, :]
    grid_j = grid[None, :, :]

    drift_1 = (drift_i * grid_j).sum(-1)

    # Degree 2
    zb = tfnp.zeros((cov_size, interpolation_options.number_dimensions))
    zc = tfnp.zeros((cov_size, interpolation_options.number_dimensions))
    if interpolation_options.uni_degree == 2:

        for i in range(n_dim):
            zb[-u_eq + n_dim + i, i] = 1
        zb[-u_eq + n_dim * 2:, 0] = 1

        for i in range(n_dim):
            zc[-u_eq + n_dim + i, i] = 1
        zc[-u_eq + n_dim * 2:, 1] = 1

    drift_i_2a = zb[:, None, :]
    drift_i_2b = zc[:, None, :]

    drift_2 = (drift_i_2a * grid_j).sum(-1) * (drift_i_2b * grid_j).sum(-1)

    return drift_1 + drift_2


def compute_perep_v(n_ori: int, cov_size: int, direction_val: int):
    perp_v = tfnp.zeros((cov_size, 1))
    perp_v[n_ori * direction_val:n_ori * (direction_val + 1)] = 1
    return perp_v
