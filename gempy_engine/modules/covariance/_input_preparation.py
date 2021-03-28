import numpy as np

from gempy_engine.config import BackendConf, AvailableBackends
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints

from gempy_engine.modules.covariance._structs import OrientationsInternals, SurfacePointsInternals
from gempy_engine.modules.covariance._utils import tfnp


def orientations_preprocess(orientations: Orientations):
    tiled_positions = tfnp.tile(orientations.dip_positions, (orientations.n_dimensions, 1))
    return OrientationsInternals(orientations, tiled_positions)


def surface_points_preprocess(sp_input: SurfacePoints, number_of_points_per_surface: np.ndarray) -> SurfacePointsInternals:
    sp = sp_input.sp_coords
    nugget_effect = sp_input.nugget_effect_scalar

    # reference point: every first point of each layer
    ref_positions = tfnp.cumsum(tfnp.concat([np.array([0]), number_of_points_per_surface[:-1] + 1], axis=0))

    if BackendConf.engine_backend is AvailableBackends.tensorflow:
        ref_nugget, ref_points, rest_nugget, rest_points = \
            _compute_rest_ref_in_tf(nugget_effect, number_of_points_per_surface, ref_positions, sp)

    else:
        ref_nugget, ref_points, rest_nugget, rest_points = \
            _compute_rest_ref_in_numpy(nugget_effect, number_of_points_per_surface, ref_positions, sp)

    # repeat the reference points (the number of persurface -1)  times
    ref_points_repeated = tfnp.repeat(ref_points, number_of_points_per_surface, 0)
    ref_nugget_repeated = tfnp.repeat(ref_nugget, number_of_points_per_surface, 0)

    nugget_effect_ref_rest = rest_nugget + ref_nugget_repeated

    return SurfacePointsInternals(ref_points_repeated, rest_points, nugget_effect_ref_rest)


def _compute_rest_ref_in_numpy(nugget_effect, number_of_points_per_surface, ref_positions, sp):
    def get_one_hot(targets, nb_classes):
        res = np.eye(nb_classes, dtype='int32')[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])

    one_hot_ = get_one_hot(ref_positions,
                           tfnp.reduce_sum(number_of_points_per_surface + 1))
    partitions = tfnp.reduce_sum(one_hot_, axis=0)
    partitions_bool = partitions.astype(bool)
    ref_points = sp[partitions_bool]
    rest_points = sp[~partitions_bool]
    ref_nugget = nugget_effect[partitions_bool]
    rest_nugget = nugget_effect[~partitions_bool]
    return ref_nugget, ref_points, rest_nugget, rest_points


def _compute_rest_ref_in_tf(nugget_effect, number_of_points_per_surface, ref_positions, sp):
    one_hot_ = tfnp.one_hot(ref_positions, tfnp.reduce_sum(number_of_points_per_surface + 1), dtype=tfnp.int32)
    # reference:1 rest: 0
    partitions = tfnp.reduce_sum(one_hot_, axis=0)
    # selecting surface points as rest set and reference set
    rest_points, ref_points = tfnp.dynamic_partition(sp, partitions, 2)
    rest_nugget, ref_nugget = tfnp.dynamic_partition(nugget_effect, partitions, 2)
    return ref_nugget, ref_points, rest_nugget, rest_points