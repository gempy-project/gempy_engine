"""One-step sparse lattice support; never seeds a second halo from support cells."""
from itertools import product
import math

import numpy as np

from ...core.backend_tensor import BackendTensor
from ...core.data.options.evaluation_options import OctreeRefinementMode
from ...core.host_arrays import upload


def close_refinement_mask(coordinates, primary_mask, domain_shape, mode):
    """Return support-only mask and number of unique absent in-domain cells.

    Neighbors are looked up offset by offset without host reads: out-of-domain and found neighbors
    are masked instead of filtered, and the absent keys are counted with one ``unique``.
    """
    t = BackendTensor.tfnp
    mode = OctreeRefinementMode(mode)
    support = t.zeros(primary_mask.shape, dtype=bool)
    if mode == OctreeRefinementMode.FAST or not t.any(primary_mask):
        return support, 0
    shape = tuple(int(n) for n in domain_shape)
    if math.prod(shape) > 2 ** 63 - 1:
        raise OverflowError("Octree domain exceeds signed int64 coordinate-key capacity")
    offsets = [offset for offset in product((-1, 0, 1), repeat=3)
               if (distance := sum(abs(v) for v in offset)) != 0
               and not (mode == OctreeRefinementMode.BALANCED and distance != 1)]
    # Host-built constants, one upload: bounds, key factors, then the offsets.
    constants = upload(np.array([shape, [shape[1] * shape[2], shape[2], 1], *offsets], dtype=np.int64))
    bounds, factors = constants[0], constants[1]
    keys = (coordinates * factors).sum(axis=1)
    order = t.argsort(keys)
    sorted_keys = keys[order]
    last = len(sorted_keys) - 1
    seeds = coordinates[primary_mask]
    hits = t.zeros(primary_mask.shape, dtype='int64')
    missing = []
    for k in range(len(offsets)):
        neighbors = seeds + constants[2 + k]
        inside = ((neighbors >= 0) & (neighbors < bounds)).all(axis=1)
        requested = (neighbors * factors).sum(axis=1)
        positions = t.searchsorted(sorted_keys, requested)
        safe = t.clip(positions, 0, last)
        found = inside & (positions <= last) & (sorted_keys[safe] == requested)
        _add_at(hits, order[safe], found)
        # Keys are non-negative inside the domain: -1 marks "not missing".
        missing.append(t.where(inside & ~found, requested, -1))
    absent = t.unique(t.concatenate(missing))
    missing_count = int((absent >= 0).sum())
    return (hits > 0) & ~primary_mask, missing_count


def _add_at(counts, index, flags):
    if hasattr(counts, "index_add_"):
        counts.index_add_(0, index, flags.to(counts.dtype))
    else:
        np.add.at(counts, index, flags.astype(counts.dtype))
