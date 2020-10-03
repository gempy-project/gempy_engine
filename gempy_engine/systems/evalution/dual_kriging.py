from gempy_engine.config import tfnp, tensorflow_imported, tensor_types, pykeops_imported
import numpy as np

from gempy_engine.data_structures.private_structures import OrientationsInternals


def export_scalar(sp_internals, ori_internals: OrientationsInternals,
                  interpolations_options, grid):

    z = np.zeros((interpolations_options.n_uni_eq,
                  interpolations_options.number_dimensions))

    #dipspoints = np.vstack((dip_pos, points_pos, z))

    #dips_points0 = dipspoints[:, None, :]
    #dips_points1 = dipspoints[None, :, :]
    dips = ori_internals.ori_input.dip_positions.reshape((-1, 1), order='F')

    dips_i = dips[:, None, :]
    grid_j = grid.reshape((-1, 1), order='F')[None, :, :]
    hu_x0 = (dips_i - grid_j)

    print(hu_x0)
#
# def exp():
#     g = tfnp.concat([dip,
#                      tfnp.zeros(cov_size - g_s, dtype='float64')],
#                     -1))
#     "Contribution interfaces and gradients-interfaces"
#
#     z = np.zeros((interpolations_options.n_uni_eq,
#                   interpolations_options.number_dimensions))
#
#     dipspoints = np.vstack((dip_pos, points_pos, z))
#     dips_points0 = dipspoints[:, None, :]
#     dips_points1 = dipspoints[None, :, :]
#
#     return dips_points0, dips_points1
#
#     sed_rest_SimPoint = self.squared_euclidean_distances(self.rest_layer_points, grid_val)
#     sed_ref_SimPoint = self.squared_euclidean_distances(self.ref_layer_points, grid_val)
#
#     sigma_0_interf = k_rest_x0 - k_ref_x0
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
#     sigma_0_grad_interface = hu_SimPoint * k_p_ref
#
#     return
#
#
#
#
# def export_gradients(export_grad):
#     "Contribution gradients and interfaces-gradients"
#
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
#     sigma_0_grad = hu * hv / (r_ref_x0 ** 2 + 1e-5) * (- k_p_ref + k_a) - k_p_ref * perp_matrix
#     sigma_0_grad = hu_rest * k_p_rest - hu_ref * k_p_ref
#     return
#
#
# def contribution_interface(self, grid_val=None, weights=None):
#     """
#       Computation of the contribution of the interfaces at every point to interpolate
#       Returns:
#           theano.tensor.vector: Contribution of all interfaces (input) at every point to interpolate
#       """
#
#
#     # Euclidian distances
#     sed_rest_SimPoint = self.squared_euclidean_distances(self.rest_layer_points, grid_val)
#     sed_ref_SimPoint = self.squared_euclidean_distances(self.ref_layer_points, grid_val)
#
#     sigma_0_interf = k_rest_x0 - k_ref_x0
#
#     return sigma_0_interf
#
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
