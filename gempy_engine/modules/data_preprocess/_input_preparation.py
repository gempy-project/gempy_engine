from gempy_engine.core.backend_tensor import BackendTensor, BackendTensor as b
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.kernel_classes.orientations import Orientations, OrientationsInternals
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints, SurfacePointsInternals


def orientations_preprocess(orientations: Orientations):
    tiled_positions = b.tfnp.tile(orientations.dip_positions, (orientations.n_dimensions, 1))
    tiled_gradients = b.tfnp.tile(orientations.dip_gradients, (orientations.n_dimensions, 1))
    tiled_nugget = b.tfnp.tile(orientations.nugget_effect_grad, (1, orientations.n_dimensions))
    return OrientationsInternals(
        orientations=orientations,
        dip_positions_tiled=tiled_positions,
        gradients_tiled=tiled_gradients,
        nugget_effect_grad=tiled_nugget
    )


def surface_points_preprocess(sp_input: SurfacePoints, tensors_structure: TensorsStructure) -> SurfacePointsInternals:

    partitions_bool = tensors_structure.partitions_bool

    # reference point: every first point of each layer
    nugget_effect = sp_input.nugget_effect_scalar
    sp_coords = sp_input.sp_coords
    
    points = sp_coords[partitions_bool]
    points1 = sp_coords[~partitions_bool]
    nugget = nugget_effect[partitions_bool]
    nugget1 = nugget_effect[~partitions_bool]
    result = nugget, points, nugget1, points1
    ref_nugget, ref_points, rest_nugget, rest_points = result

    # repeat the reference points (the number of persurface -1)  times
    number_repetitions = tensors_structure.number_of_points_per_surface - 1
    number_repetitions = BackendTensor.t.array(number_repetitions)
    
    ref_points_repeated = b.t.repeat(ref_points, number_repetitions, 0)  # ref_points shape: (1, 3)
    # ref_nugget_repeated = b.t.repeat(ref_nugget, number_repetitions, 0)  # ref_nugget shape: (1)

    # ? (miguel April 24) I need to decide what to do with this -- nugget_effect_ref_rest = (rest_nugget + ref_nugget_repeated)/2
    nugget_effect_ref_rest = rest_nugget

    return SurfacePointsInternals(ref_points_repeated, rest_points, nugget_effect_ref_rest)


import torch


def surface_points_preprocess_(sp_input: SurfacePoints, tensors_structure: TensorsStructure) -> SurfacePointsInternals:
    # 1. Ensure inputs are PyTorch tensors on the correct device (GPU)
    # If they are currently NumPy arrays, convert them ONCE before this function is called.
    partitions_bool = tensors_structure.partitions_bool
    sp_coords = sp_input.sp_coords
    nugget_effect = sp_input.nugget_effect_scalar

    # 2. Perform boolean masking directly on the GPU
    # PyTorch handles this natively and parallelizes it perfectly.
    ref_points = sp_coords[partitions_bool]
    rest_points = sp_coords[~partitions_bool]

    # ref_nugget = nugget_effect[partitions_bool] # Only uncomment if you need it later
    rest_nugget = nugget_effect[~partitions_bool]

    # 3. Get the repetitions tensor
    number_repetitions = tensors_structure.number_of_points_per_surface - 1

    # Ensure number_repetitions is a tensor on the same device as ref_points
    if not isinstance(number_repetitions, torch.Tensor):
        number_repetitions = torch.tensor(number_repetitions, device=ref_points.device)

    # 4. Use repeat_interleave instead of repeat
    ref_points_repeated = torch.repeat_interleave(ref_points, number_repetitions, dim=0)

    # nugget_effect_ref_rest = (rest_nugget + ref_nugget_repeated)/2
    nugget_effect_ref_rest = rest_nugget

    return SurfacePointsInternals(ref_points_repeated, rest_points, nugget_effect_ref_rest)


