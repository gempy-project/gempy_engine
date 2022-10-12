import numpy as np

from gempy_engine.core.backend_tensor import BackendTensor, BackendTensor as b, AvailableBackends
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.kernel_classes.orientations import Orientations, OrientationsInternals
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints, SurfacePointsInternals


def orientations_preprocess(orientations: Orientations):
    tiled_positions = b.tfnp.tile(orientations.dip_positions, (orientations.n_dimensions, 1))
    tiled_gradients = b.tfnp.tile(orientations.dip_gradients, (orientations.n_dimensions, 1))
    return OrientationsInternals(orientations, tiled_positions, tiled_gradients)


def surface_points_preprocess(sp_input: SurfacePoints, tensors_structure: TensorsStructure) -> SurfacePointsInternals:

    partitions_bool = tensors_structure.partitions_bool

    if False:#BackendTensor.engine_backend is AvailableBackends.tensorflow:
        ref_nugget, ref_points, rest_nugget, rest_points = _compute_rest_ref_in_tf(partitions_bool, sp_input)
    else:
        ref_nugget, ref_points, rest_nugget, rest_points = _compute_rest_ref_in_numpy(partitions_bool, sp_input)

    # repeat the reference points (the number of persurface -1)  times
    number_repetitions = tensors_structure.number_of_points_per_surface - 1
    ref_points_repeated = b.tfnp.repeat(ref_points, number_repetitions, 0)
    ref_nugget_repeated = b.tfnp.repeat(ref_nugget, number_repetitions, 0)

    nugget_effect_ref_rest = rest_nugget + ref_nugget_repeated

    return SurfacePointsInternals(ref_points_repeated, rest_points, nugget_effect_ref_rest)


def _compute_rest_ref_in_numpy(partitions_bool: np.ndarray, sp: SurfacePoints):
    # reference point: every first point of each layer
    nugget_effect = sp.nugget_effect_scalar
    sp_coords = sp.sp_coords

    ref_points = sp_coords[partitions_bool]
    rest_points = sp_coords[~partitions_bool]

    ref_nugget = nugget_effect[partitions_bool]
    rest_nugget = nugget_effect[~partitions_bool]
    return ref_nugget, ref_points, rest_nugget, rest_points


def method_name(tensors_structure: TensorsStructure):
    def get_one_hot(targets, nb_classes):
        res = np.eye(nb_classes, dtype='int32')[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape) + [nb_classes])
    
    ref_positions = tensors_structure.reference_sp_position

    one_hot_ = get_one_hot(ref_positions, tensors_structure.total_number_sp)
    partitions = b.tfnp.reduce_sum(one_hot_, axis=0)
    partitions_bool = partitions.astype(bool)
    
    return partitions_bool


def _compute_rest_ref_in_tf(partitions: np.ndarray, sp: SurfacePoints):
    nugget_effect = sp.nugget_effect_scalar
    sp_coords = sp.sp_coords

    # reference point: every first point of each layer
    # TODO: Missing testing (May 2022)
    # ref_positions = b.tfnp.cumsum(
    #     b.tfnp.concat([np.array([0], dtype="int32"), number_of_points_per_surface[:-1]], axis=0))
    # ref_positions = tensors_structure.reference_sp_position
    # 
    # one_hot_ = b.tfnp.one_hot(ref_positions, tensors_structure.total_number_sp, dtype=b.t.int32)
    # # reference:1 rest: 0
    # partitions = b.tfnp.reduce_sum(one_hot_, axis=0)

    # selecting surface points as rest set and reference set
    rest_points, ref_points = b.tfnp.dynamic_partition(sp_coords, partitions, 2)
    rest_nugget, ref_nugget = b.tfnp.dynamic_partition(nugget_effect, partitions, 2)
    return ref_nugget, ref_points, rest_nugget, rest_points
