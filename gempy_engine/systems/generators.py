from typing import Union
from gempy_engine.config import use_tf
from pykeops.numpy import LazyTensor as LazyTensor_np
import numpy as np

# TODO Check in the configuration script
from gempy_engine.data_structures.public_structures import SurfacePointsInput

try:
    import tensorflow as tf

    # if we import it we check in config
    tensorflow_imported = use_tf
except ImportError:
    tensorflow_imported = False

# There are 3 possibles cases for the numpy-TF compatibility
# 1) signature and args are the same -> Nothing needs to be changed
# 2) signature is different but args are the same -> We need to override the
#    name of the function
# 3) signature and args are different -> We need an if statement

# Case 1)
tfnp = tf if tensorflow_imported else np
tensor_types = Union[np.ndarray, tf.Tensor, tf.Variable]

# Case 2)
if tensorflow_imported is False:
    tfnp.reduce_sum = tfnp.sum
    tfnp.concat = tfnp.concatenate


def tile_dip_positions(dip_positions, n_dimensions):
    return tfnp.tile(dip_positions, n_dimensions)


def get_ref_rest(sp_input: SurfacePointsInput,
                 number_of_points_per_surface: np.ndarray):
    sp = sp_input.sp_positions
    nugget_effect = sp_input.nugget_effect_scalar

    # reference point: every first point of each layer
    ref_positions = tfnp.cumsum(
        tf.concat([[0], number_of_points_per_surface[:-1] + 1], axis=0)
    )

    if tensorflow_imported:
        one_hot_ = tfnp.one_hot(
            ref_positions, tfnp.reduce_sum(number_of_points_per_surface + 1),
            dtype=tf.int32
        )
        # reference:1 rest: 0
        partitions = tfnp.reduce_sum(one_hot_, axis=0)

        # selecting surface points as rest set and reference set
        rest_points, ref_points = tfnp.dynamic_partition(sp, partitions, 2)
        rest_nugget, ref_nugget = tfnp.dynamic_partition(nugget_effect, partitions, 2)

    else:
        def get_one_hot(targets, nb_classes):
            res = np.eye(nb_classes, dtype='int32')[np.array(targets).reshape(-1)]
            return res.reshape(list(targets.shape) + [nb_classes])

        one_hot_ = get_one_hot(ref_positions,
                               tfnp.reduce_sum(number_of_points_per_surface + 1))

        partitions = tfnp.reduce_sum(one_hot_, axis=0)
        partitions_bool = partitions.astype(bool)

        ref_points = sp[partitions_bool]
        rest_points = sp[~partitions_bool]
        ref_nugget = sp[partitions_bool]
        rest_nugget = sp[~partitions_bool]

    # repeat the reference points (the number of persurface -1)  times
    ref_points_repeated = tfnp.repeat(
        ref_points, number_of_points_per_surface, 0)
    ref_nugget_repeated = tfnp.repeat(
        ref_nugget, number_of_points_per_surface, 0)

    return ref_points_repeated, rest_points, ref_nugget_repeated, rest_nugget


def squared_euclidean_distances(x_1: tensor_types,
                                x_2: tensor_types):
    """
    Compute the euclidian distances in 3D between all the points in x_1 and x_2

    Args:
        x_1 (theano.tensor.matrix): shape n_points x number dimension
        x_2 (theano.tensor.matrix): shape n_points x number dimension

    Returns:
        theano.tensor.matrix: Distancse matrix. shape n_points x n_points
    """

    # T.maximum avoid negative numbers increasing stability
    # x_1 = LazyTensor_np(x_1[:, None, :])
    # x_2 = LazyTensor_np(x_2[None, :, :])
    #
    # t1 = tfnp.reduce_sum(x_1 ** 2, axis=1)
    # t2 = tfnp.reduce_sum(x_2 ** 2, axis=1)
    #
    # t3 = tfnp.reshape(t1, (x_1.shape[0], 1))
    # t4 = tfnp.reshape(t2, (1, x_2.shape[0]))
    #
    # sqd = tfnp.sqrt(tfnp.maximum(
    #     t3 +
    #     t4 -
    #     2 * tfnp.tensordot(x_1, tfnp.transpose(x_2), axes=1)
    #     , 1e-12
    # ))
    keops = False
    if keops is False:
        x_1 = x_1[:, None, :]
        x_2 = x_2[None, :, :]
    else:
        x_1 = LazyTensor_np(x_1[:, None, :])
        x_2 = LazyTensor_np(x_2[None, :, :])
    sqd = tfnp.reduce_sum(((x_1 - x_2) ** 2), -1)

    return sqd


def cartesian_distances(xzy_0: tensor_types,
                        xzy_1: tensor_types,
                        n_dim: int):
    # Cartesian distances between dips positions
    h = tfnp.concat((
        tfnp.tile(
            xzy_0[:, 0] - xzy_1[:, 0].reshape(
                (
                    xzy_0[:, 0].shape[0], 1
                )
            ),
            (n_dim, 1)
        ),
        tfnp.tile(
            xzy_0[:, 1] - xzy_1[:, 1].reshape(
                (
                    xzy_0[:, 0].shape[0], 1
                )
            ),
            (n_dim, 1)
        ),
        tfnp.tile(
            xzy_0[:, 2] - xzy_1[:, 2].reshape(
                (
                    xzy_0[:, 0].shape[0], 1
                )
            ),
            (n_dim, 1)
        )),
        axis=1
    )
    return h


def compute_perpendicular_matrix(dips_size, n_dimensions=3):
    # Perpendicularity matrix. Boolean matrix to separate cross-covariance and
    # every gradient direction covariance (block diagonal)
    perpendicularity_matrix_ = np.zeros((dips_size * n_dimensions,
                                         dips_size * n_dimensions))

    perpendicularity_matrix_[0:dips_size, 0:dips_size] += 1
    perpendicularity_matrix_[dips_size:dips_size * 2, dips_size:dips_size * 2] += 1
    perpendicularity_matrix_[dips_size * 2:dips_size * 3, dips_size * 2:dips_size * 3] += 1

    if tensorflow_imported is True:
        perpendicularity_matrix = tfnp.constant(perpendicularity_matrix_)
    else:
        perpendicularity_matrix = perpendicularity_matrix_
    return perpendicularity_matrix


def compute_cov_gradients(sed_dips_dips, h_u, h_v, perpendicularity_matrix,
                          kriging_parameters, nugget_effect_grad):
    range_ = kriging_parameters.range
    c_o = kriging_parameters.c_o

    # Covariance matrix for gradients at every xyz direction and their cross-covariances

    if tensorflow_imported:
        t1 = tf.math.divide_no_nan(h_u * h_v, sed_dips_dips ** 2)
    else:
        t1 = (h_u * h_v / sed_dips_dips ** 2)
    c_g = t1 * ((
                        -c_o * ((-14 / range_ ** 2) + 105 / 4 * sed_dips_dips / range_ ** 3 -
                                35 / 2 * sed_dips_dips ** 3 / range_ ** 5 +
                                21 / 4 * sed_dips_dips ** 5 / range_ ** 7)
                ) + (
                        c_o * 7 * (9 * sed_dips_dips ** 5 - 20 * range_ ** 2 * sed_dips_dips ** 3 +
                                   15 * range_ ** 4 * sed_dips_dips - 4 * range_ ** 5) / (2 * range_ ** 7)
                )) - (
                  perpendicularity_matrix *
                  c_o * ((-14 / range_ ** 2) + 105 / 4 * sed_dips_dips / range_ ** 3 -
                         35 / 2 * sed_dips_dips ** 3 / range_ ** 5 + 21 / 4 * sed_dips_dips ** 5 / range_ ** 7)
          )

    # Setting nugget effect of the gradients
    c_g = c_g + tfnp.eye(c_g.shape[0], dtype='float64') * nugget_effect_grad

    return c_g


def tensor_transpose(tensor):
    return tfnp.transpose(tensor)
