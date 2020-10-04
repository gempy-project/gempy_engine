from gempy_engine.config import tfnp, tensorflow_imported, tensor_types, pykeops_imported
import numpy as np

from gempy_engine.data_structures.private_structures import OrientationsInternals, SurfacePointsInternals, ExportInput
from gempy_engine.data_structures.public_structures import InterpolationOptions
from gempy_engine.systems.kernel.kernel import get_kernels


def hu_x0(ori_internals: OrientationsInternals,
          grid, cov_size):

    dips = ori_internals.ori_input.dip_positions.reshape((-1, 1), order='F')
    dips_x0 = tfnp.zeros((cov_size, 1), dtype='float64')

    sel_x0 = tfnp.zeros((cov_size, 1), dtype='float64')
    sel_x0[:ori_internals.n_orientations_tiled] = 1
    dips_x0[:ori_internals.n_orientations_tiled] = dips
    dips_i = dips_x0[:, None, :]
    sel_x0 = sel_x0[:, None, :]
    grid_j = grid.reshape((-1, 1), order='F')[None, :, :]
    hu_x0 = ((dips_i - grid_j) * sel_x0).sum(axis=-1)
    return hu_x0


def get_direction_val(direction: str):
    if direction == 'x':
        direction_val = 0
    if direction == 'y':
        direction_val = 1
    if direction == 'z':
        direction_val = 2
    return direction_val


def hv_x0(ori_internals: OrientationsInternals,
          interpolations_options: InterpolationOptions, grid,
          direction_val, cov_size):

    dips = tfnp.tile(
        ori_internals.ori_input.dip_positions[:, direction_val],
        interpolations_options.number_dimensions
    ).reshape((-1, 1))

    dips_x0 = tfnp.zeros((cov_size, 1), dtype='float64')
    grid_ = tfnp.tile(
        grid[:, direction_val],
        interpolations_options.number_dimensions
    ).reshape((-1, 1))

    sel_x0 = tfnp.zeros((cov_size, 1), dtype='float64')
    sel_x0[:ori_internals.n_orientations_tiled] = 1
    dips_x0[:ori_internals.n_orientations_tiled] = dips
    dips_i = dips_x0[:, None, :]
    sel_x0 = sel_x0[:, None, :]
    grid_j = grid_[None, :, :]
    hv_x0 = ((dips_i - grid_j) * sel_x0).sum(axis=-1)
    return hv_x0


def export(sp_internals: SurfacePointsInternals,
           ori_internals: OrientationsInternals,
           interpolations_options: InterpolationOptions, grid):
    cov_size = sp_internals.n_points + \
               ori_internals.n_orientations_tiled + \
               interpolations_options.n_uni_eq
    direction_val = get_direction_val('x')

    ei = vector_preparation_export(sp_internals, ori_internals, interpolations_options)
    hu = hu_x0(ori_internals, grid, cov_size)
    hv = hv_x0(ori_internals, interpolations_options, direction_val, cov_size)
    k, k1, k2 = get_kernels('cubic')
    perp_v = compute_perep_v(
        ori_internals.n_orientations,
        cov_size,
        direction_val
    )
    z_0 = export_scalar(ei, hu, k1, interpolations_options.range)
    dz_dx =  export_gradients(ei, hu, hv, k1, k2, perp_v)

    return z_0, dz_dx


def compute_perep_v(n_ori: int, cov_size:int, direction_val: int):
    perp_v = tfnp.zeros_like(cov_size)
    perp_v[n_ori * direction_val:n_ori * (direction_val + 1)] = 1
    return perp_v


def vector_preparation_export(sp_internals: SurfacePointsInternals,
                              ori_internals: OrientationsInternals,
                              interpolations_options: InterpolationOptions, grid):
    dips = ori_internals.ori_input.dip_positions

    dips_i = dips[:, None, :]
    grid_j = grid[None, :, :]

    z = np.zeros((interpolations_options.n_uni_eq,
                  interpolations_options.number_dimensions))
    z2 = np.zeros_like(ori_internals.dip_positions_tiled)

    ref_x0 = np.vstack((z2, sp_internals.ref_surface_points, z))
    rest_x0 = np.vstack((z2, sp_internals.rest_surface_points, z))
    ref_i = ref_x0[:, None, :]
    rest_i = rest_x0[:, None, :]

    ei = ExportInput(dips_i, grid_j, ref_i, rest_i)
    return ei


def export_scalar(ei: ExportInput, hu_x0, kernel_1st, a):

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

    return sigma_0_grad_interface + sigma_0_interf


def export_gradients(ei: ExportInput, hu, hv,
                     kernel_1st, kernel_2nd, perp_v):
    if pykeops_imported is True:
        r_dip_x0 = (((ei.dips_i - ei.grid_j) ** 2).sum(-1)).sqrt()
    else:
        r_dip_x0 = tfnp.sqrt(((ei.dips_i - ei.grid_j) ** 2).sum(-1))

    k_p_dip_x0 = kernel_1st(r_dip_x0)
    k_a_dip_x0 = kernel_2nd(r_dip_x0)

    sigma_0_grad = hu * hv / (r_dip_x0 ** 2 + 1e-5) * (- k_p_dip_x0 + k_a_dip_x0) - \
                   k_p_dip_x0 * perp_v
    return


# hu_x0[:ori_internals.n_orientations_tiled] = (dips_i - grid_j)
#
# tfnp.concat([hu_x0[:, :, 0],
#              tfnp.zeros(cov_size - g_s, dtype='float64')],
#              #                     -1))


#
#
#

def export_gradients(export_grad):
    "Contribution gradients and interfaces-gradients"

    if direction == 'x':
        direction_val = 0
    if direction == 'y':
        direction_val = 1
    if direction == 'z':
        direction_val = 2
    # self.gi_reescale = theano.shared(1)
    #
    # if weights is None:
    #     weights = self.extend_dual_kriging()
    # if grid_val is None:
    #     grid_val = self.x_to_interpolate()
    #
    # length_of_CG = self.matrices_shapes()[0]

    # Cartesian distances between the point to simulate and the dips
    # TODO optimize to compute this only once?
    # Euclidean distances
    sed_dips_SimPoint = self.squared_euclidean_distances(grid_val, self.dips_position_tiled).T

    # Cartesian distances between dips positions
    h_u = T.tile(
        self.dips_position[:, direction_val] -
        grid_val[:, direction_val].reshape((grid_val[:, direction_val].shape[0], 1)), 3)
    h_v = T.horizontal_stack(
        T.tile(self.dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1)),
               1),
        T.tile(self.dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1)),
               1),
        T.tile(self.dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1)),
               1))

    perpendicularity_vector = T.zeros(T.stack(length_of_CG))
    perpendicularity_vector = T.set_subtensor(
        perpendicularity_vector[
        self.dips_position.shape[0] * direction_val:self.dips_position.shape[0] * (direction_val + 1)], 1)

    sigma_0_grad = hu * hv / (r_ref_x0 ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * perp_matrix

    return

#
# def contribution_gradient(self, direction='x', grid_val=None, weights=None):
#     if direction == 'x':
#         direction_val = 0
#     if direction == 'y':
#         direction_val = 1
#     if direction == 'z':
#         direction_val = 2
#     # self.gi_reescale = theano.shared(1)
#     #
#     # if weights is None:
#     #     weights = self.extend_dual_kriging()
#     # if grid_val is None:
#     #     grid_val = self.x_to_interpolate()
#     #
#     # length_of_CG = self.matrices_shapes()[0]
#
#     # Cartesian distances between the point to simulate and the dips
#     # TODO optimize to compute this only once?
#     # Euclidean distances
#     sed_dips_SimPoint = self.squared_euclidean_distances(grid_val, self.dips_position_tiled).T
#
#     if 'sed_dips_SimPoint' in self.verbose:
#         sed_dips_SimPoint = theano.printing.Print('sed_dips_SimPoint')(sed_dips_SimPoint)
#
#     # Cartesian distances between dips positions
#     # h_u = T.tile(self.dips_position[:, direction_val] - grid_val[:, direction_val].reshape(
#     #     (grid_val[:, direction_val].shape[0], 1)), 3)
#     # h_v = T.horizontal_stack(
#     #     T.tile(self.dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1)),
#     #            1),
#     #     T.tile(self.dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1)),
#     #            1),
#     #     T.tile(self.dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1)),
#     #            1))
#     #
#     # perpendicularity_vector = T.zeros(T.stack(length_of_CG))
#     # perpendicularity_vector = T.set_subtensor(
#     #     perpendicularity_vector[
#     #     self.dips_position.shape[0] * direction_val:self.dips_position.shape[0] * (direction_val + 1)], 1)
#
#
#     sigma_0_grad = hu * hv / (r_ref_x0 ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * perp_matrix
#     return sigma_0_grad
#
#
# def contribution_gradient_interface(self, grid_val=None, weights=None):
#     """
#     Computation of the contribution of the foliations at every point to interpolate
#     Returns:
#         theano.tensor.vector: Contribution of all foliations (input) at every point to interpolate
#     """
#
#     length_of_CG = self.matrices_shapes()[0]
#
#     # Cartesian distances between the point to simulate and the dips
#     hu_SimPoint = T.vertical_stack(
#         (self.dips_position[:, 0] - grid_val[:, 0].reshape((grid_val[:, 0].shape[0], 1))).T,
#         (self.dips_position[:, 1] - grid_val[:, 1].reshape((grid_val[:, 1].shape[0], 1))).T,
#         (self.dips_position[:, 2] - grid_val[:, 2].reshape((grid_val[:, 2].shape[0], 1))).T
#     )
#
#     # Euclidian distances
#     sed_dips_SimPoint = self.squared_euclidean_distances(self.dips_position_tiled, grid_val)
#     # Gradient contribution
#
#     sigma_0_grad = hu_SimPoint * k_p_ref
#     return sigma_0_grad
#
#
# def contribution_interface_gradient(self, direction='x', grid_val=None, weights=None):
#     """
#     Computation of the contribution of the foliations at every point to interpolate
#     Returns:
#         theano.tensor.vector: Contribution of all foliations (input) at every point to interpolate
#     """
#     #
#     # if direction == 'x':
#     #     dir_val = 0
#     # if direction == 'y':
#     #     dir_val = 1
#     # if direction == 'z':
#     #     dir_val = 2
#     #
#     # if weights is None:
#     #     weights = self.extend_dual_kriging()
#     # if grid_val is None:
#     #     grid_val = self.x_to_interpolate()
#     #
#     # length_of_CG, length_of_CGI = self.matrices_shapes()[:2]
#     #
#     # # Cartesian distances between the point to simulate and the dips
#     # hu_rest = (- self.rest_layer_points[:, dir_val] + grid_val[:, dir_val].reshape((grid_val[:, dir_val].shape[0], 1)))
#     # hu_ref = (- self.ref_layer_points[:, dir_val] + grid_val[:, dir_val].reshape((grid_val[:, dir_val].shape[0], 1)))
#     #
#     # # Euclidian distances
#     #
#     # sed_grid_rest = self.squared_euclidean_distances(grid_val, self.rest_layer_points)
#     # sed_grid_ref = self.squared_euclidean_distances(grid_val, self.ref_layer_points)
#     #
#     # # Gradient contribution
#     # self.gi_reescale = 2
#     #
#     # sigma_0_grad = T.sum(
#     #     (weights[length_of_CG:length_of_CG + length_of_CGI] *
#     #      self.gi_reescale * (
#     #              (hu_rest *
#     #               (sed_grid_rest < self.a_T) *  # first derivative
#     #               (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_grid_rest / self.a_T ** 3 -
#     #                                35 / 2 * sed_grid_rest ** 3 / self.a_T ** 5 +
#     #                                21 / 4 * sed_grid_rest ** 5 / self.a_T ** 7))) -
#     #              (hu_ref *
#     #               (sed_grid_ref < self.a_T) *  # first derivative
#     #               (- self.c_o_T * ((-14 / self.a_T ** 2) + 105 / 4 * sed_grid_ref / self.a_T ** 3 -
#     #                                35 / 2 * sed_grid_ref ** 3 / self.a_T ** 5 +
#     #                                21 / 4 * sed_grid_ref ** 5 / self.a_T ** 7)))).T),
#     #     axis=0)
#
#     sigma_0_grad = hu_rest * k_p_rest - hu_ref * k_p_ref
#
#     return sigma_0_grad
