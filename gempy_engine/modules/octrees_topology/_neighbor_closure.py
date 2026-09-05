"""One-step sparse lattice support; never seeds a second halo from support cells."""
from itertools import product
import math

from ...core.backend_tensor import BackendTensor
from ...core.data.options.evaluation_options import OctreeRefinementMode


def close_refinement_mask(coordinates, primary_mask, domain_shape, mode):
    """Return support-only mask and number of unique absent in-domain cells."""
    t = BackendTensor.tfnp
    mode = OctreeRefinementMode(mode)
    support = t.zeros(primary_mask.shape, dtype=bool)
    if mode == OctreeRefinementMode.FAST or not t.any(primary_mask):
        return support, 0
    shape = tuple(int(n) for n in domain_shape)
    if math.prod(shape) > 2 ** 63 - 1:
        raise OverflowError("Octree domain exceeds signed int64 coordinate-key capacity")
    bounds = t.array(shape, dtype='int64')
    factors = t.array([shape[1] * shape[2], shape[2], 1], dtype='int64')
    keys = (coordinates * factors).sum(axis=1)
    order = t.argsort(keys)
    sorted_keys = keys[order]
    seeds = coordinates[primary_mask]
    missing = []
    for offset in product((-1, 0, 1), repeat=3):
        distance = sum(abs(v) for v in offset)
        if distance == 0 or (mode == OctreeRefinementMode.BALANCED and distance != 1):
            continue
        neighbors = seeds + t.array(offset, dtype='int64')
        inside = ((neighbors >= 0) & (neighbors < bounds)).all(axis=1)
        requested = (neighbors[inside] * factors).sum(axis=1)
        positions = t.searchsorted(sorted_keys, requested)
        safe = t.clip(positions, 0, len(sorted_keys) - 1)
        found = (positions < len(sorted_keys)) & (sorted_keys[safe] == requested)
        support[order[safe[found]]] = True
        missing.append(requested[~found])
    missing_count = int(t.unique(t.concatenate(missing)).shape[0])
    return support & ~primary_mask, missing_count
