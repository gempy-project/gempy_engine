import numpy as np

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor, BackendTensor as b
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.kernel_classes.orientations import Orientations, OrientationsInternals
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints, SurfacePointsInternals
from gempy_engine.core.host_arrays import upload


def orientations_preprocess(orientations: Orientations):
    tiled_positions = b.tfnp.tile(orientations.dip_positions, (orientations.n_dimensions, 1))
    tiled_gradients = b.tfnp.tile(orientations.dip_gradients, (orientations.n_dimensions, 1))
    tiled_nugget = b.tfnp.tile(orientations.nugget_effect_grad, orientations.n_dimensions)
    return OrientationsInternals(
        orientations=orientations,
        dip_positions_tiled=tiled_positions,
        gradients_tiled=tiled_gradients,
        nugget_effect_grad=tiled_nugget
    )


def chunk_preprocess(surface_points: list[SurfacePoints], orientations: list[Orientations],
                     tensors_structures: list[TensorsStructure]) -> list[tuple] | None:
    """``(surface_points_preprocess, orientations_preprocess)`` of several stacks, from one index upload and
    one gather per array over the stacks' arrays concatenated; each stack gets views of the results.

    The rows are the ones the per-stack functions select (reference, rest and repeated-reference points;
    orientations tiled once per dimension), so the values are the same. Each structure also keeps its
    stack-local split (``TensorsStructure.host_split``) from the same upload, for its fault values. None
    on the numpy backend, without host counts, or with arrays of mixed dtypes.
    """
    counts = [getattr(ts, "host_counts", None) for ts in tensors_structures]
    if BackendTensor.engine_backend is not AvailableBackends.PYTORCH or any(not isinstance(c, np.ndarray) for c in counts):
        return None
    sources = {"coords": [sp.sp_coords for sp in surface_points], "nuggets": [sp.nugget_effect_scalar for sp in surface_points],
               "positions": [o.dip_positions for o in orientations], "gradients": [o.dip_gradients for o in orientations],
               "ori_nuggets": [o.nugget_effect_grad for o in orientations]}
    if (any(len({a.dtype for a in arrays}) != 1 for arrays in sources.values())
            or any(int(c.sum()) != sp.sp_coords.shape[0] for c, sp in zip(counts, surface_points))):
        return None

    names = ("ref", "rest", "repeated", "tiled", "ref_local", "rest_local", "repeated_local", "surface")
    parts, sp_at, ori_at = {name: [] for name in names}, 0, 0
    for c, o in zip(counts, orientations):
        c = c.astype(np.int64)
        starts = np.concatenate(([0], np.cumsum(c)[:-1]))
        rest = np.setdiff1d(np.arange(c.sum()), starts)  # ascending, as a boolean mask selects
        repeated = np.repeat(starts, c - 1)
        for name, local in (("ref", starts), ("rest", rest), ("repeated", repeated)):
            parts[name].append(local + sp_at)
            parts[name + "_local"].append(local)
        parts["tiled"].append(np.tile(np.arange(o.dip_positions.shape[0]), o.n_dimensions) + ori_at)
        parts["surface"].append(np.repeat(np.arange(len(c)), c - 1))
        sp_at, ori_at = sp_at + int(c.sum()), ori_at + o.dip_positions.shape[0]
    pieces = [np.concatenate(parts[name]).astype(np.int64) for name in names]
    index = dict(zip(names, BackendTensor.t.split(upload(np.concatenate(pieces)), [len(p) for p in pieces])))

    def per_stack(values, name):
        return BackendTensor.t.split(values, [len(p) for p in parts[name]])

    coords, nuggets, positions, gradients, ori_nuggets = (BackendTensor.t.cat(arrays) for arrays in sources.values())
    rest_nuggets = nuggets[index["rest"]]
    sp_fields = zip(per_stack(coords[index["repeated"]], "repeated"), per_stack(coords[index["rest"]], "rest"),
                    per_stack(rest_nuggets, "rest"), per_stack(nuggets[index["repeated"]], "repeated"),
                    per_stack(nuggets[index["ref"]], "ref"), per_stack(index["surface"].to(rest_nuggets.dtype), "surface"))
    ori_fields = zip(orientations, per_stack(positions[index["tiled"]], "tiled"),
                     per_stack(gradients[index["tiled"]], "tiled"), per_stack(ori_nuggets[index["tiled"]], "tiled"))
    local = zip(*(per_stack(index[name], name) for name in ("ref_local", "rest_local", "repeated_local", "surface")))
    internals = []
    for structure, sp_values, ori_values, split in zip(tensors_structures, sp_fields, ori_fields, local):
        structure._host_split = (str(BackendTensor.device), split)
        o, tiled_positions, tiled_gradients, tiled_nuggets = ori_values
        internals.append((SurfacePointsInternals(*sp_values),
                          OrientationsInternals(orientations=o, dip_positions_tiled=tiled_positions,
                                                gradients_tiled=tiled_gradients, nugget_effect_grad=tiled_nuggets)))
    return internals


def surface_points_preprocess(sp_input: SurfacePoints, tensors_structure: TensorsStructure) -> SurfacePointsInternals:
    split = tensors_structure.host_split()
    if split is not None:  # the same rows, gathered by host-built indices
        ref_index, rest_index, repeated_ref_index, surface_index = split
        nugget_effect, sp_coords = sp_input.nugget_effect_scalar, sp_input.sp_coords
        rest_nugget = nugget_effect[rest_index]
        return SurfacePointsInternals(
            sp_coords[repeated_ref_index],
            sp_coords[rest_index],
            rest_nugget,
            nugget_effect[repeated_ref_index],
            nugget_effect[ref_index],
            surface_index.to(rest_nugget.dtype),
        )

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
    ref_nugget_repeated = b.t.repeat(ref_nugget, number_repetitions, 0)
    surface_ids = b.t.repeat(
        b.arange(tensors_structure.n_surfaces, dtype=rest_nugget.dtype),
        number_repetitions,
        0,
    )

    return SurfacePointsInternals(
        ref_points_repeated,
        rest_points,
        rest_nugget,
        ref_nugget_repeated,
        ref_nugget,
        surface_ids,
    )




def surface_points_preprocess_(sp_input: SurfacePoints, tensors_structure: TensorsStructure) -> SurfacePointsInternals:
    import torch
    # 1. Ensure inputs are PyTorch tensors on the correct device (GPU)
    # If they are currently NumPy arrays, convert them ONCE before this function is called.
    partitions_bool = tensors_structure.partitions_bool
    sp_coords = sp_input.sp_coords
    nugget_effect = sp_input.nugget_effect_scalar

    # 2. Perform boolean masking directly on the GPU
    # PyTorch handles this natively and parallelizes it perfectly.
    ref_points = sp_coords[partitions_bool]
    rest_points = sp_coords[~partitions_bool]

    ref_nugget = nugget_effect[partitions_bool]
    rest_nugget = nugget_effect[~partitions_bool]

    # 3. Get the repetitions tensor
    number_repetitions = tensors_structure.number_of_points_per_surface - 1

    # Ensure number_repetitions is a tensor on the same device as ref_points
    if not isinstance(number_repetitions, torch.Tensor):
        number_repetitions = torch.tensor(number_repetitions, device=ref_points.device)

    # 4. Use repeat_interleave instead of repeat
    ref_points_repeated = torch.repeat_interleave(ref_points, number_repetitions, dim=0)
    ref_nugget_repeated = torch.repeat_interleave(ref_nugget, number_repetitions, dim=0)
    surface_ids = torch.repeat_interleave(
        torch.arange(tensors_structure.n_surfaces, device=ref_points.device, dtype=ref_points.dtype),
        number_repetitions,
    )

    return SurfacePointsInternals(
        ref_points_repeated,
        rest_points,
        rest_nugget,
        ref_nugget_repeated,
        ref_nugget,
        surface_ids,
    )
