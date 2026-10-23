"""2:1 face/edge balancing of adaptive joint leaves, before the cell complex is built.

The production octree only refines cells whose corners differ, so a leaf can be
two or more levels coarser than a face or edge neighbour. The cell complex
requires at most one level across faces and edges (corner-only differences are
allowed), so coarse leaves are split here until that holds. Existing leaves keep
their production corner samples; a split leaf's children reuse every corner
value already known at that lattice node and only new nodes are queried.
NumPy or Torch inputs; Torch work stays on the inputs' device.
"""

from itertools import product

import numpy as np

from .joint_cell_branches import CORNERS
from .joint_lattice import leaf_corner_nodes
from .joint_xp import host, namespace

# Face and edge neighbours; corner-only neighbours may differ by any depth.
_DIRECTIONS = np.array([d for d in product((-1, 0, 1), repeat=3) if 1 <= sum(map(abs, d)) <= 2],
                       dtype=np.int64)


def balance_leaves(origins, spans, domain_shape):
    """Split leaves until face/edge neighbours differ by at most one level.

    Inputs are aligned dyadic leaves covering ``[0, domain_shape)`` in
    finest-lattice units. Returns ``(origins, spans, source, split_count)``:
    ``source`` is each leaf's input row, or -1 for children of split leaves.
    Unsplit leaves keep their order and come first, children are appended.
    """
    xp = namespace(origins, spans)
    origins = xp.asarray(origins, dtype=np.int64).reshape(-1, 3)
    spans = xp.asarray(spans, dtype=np.int64).reshape(-1)
    source = xp.arange(len(spans))
    domain = tuple(int(d) for d in host(domain_shape))
    corners = xp.asarray(CORNERS, dtype=np.int64)
    split_count = 0
    while True:
        coarse = _coarse_neighbors(xp, origins, spans, domain)
        if not len(coarse):
            return origins, spans, source, split_count
        split_count += len(coarse)
        keep = xp.ones(len(spans), dtype=bool)
        keep[coarse] = False
        half = spans[coarse] // 2
        children = (origins[coarse][:, None, :] + half[:, None, None] * corners).reshape(-1, 3)
        origins = xp.concatenate((origins[keep], children))
        spans = xp.concatenate((spans[keep], xp.repeat(half, 8)))
        source = xp.concatenate((source[keep], xp.full(len(children), -1, dtype=np.int64)))


def _coarse_neighbors(xp, origins, spans, domain):
    """Leaves more than twice as wide as a face/edge neighbour, probed from the finer side.

    A neighbour wider than a leaf covers that leaf's whole adjacent face or edge
    (dyadic nesting), so one finest-voxel probe per direction finds it.
    """
    directions = xp.asarray(_DIRECTIONS, dtype=np.int64)
    probes = origins[:, None, :] + xp.where(directions > 0, spans[:, None, None],
                                            xp.where(directions < 0, -1, 0))
    probes = probes.reshape(-1, 3)
    widths = xp.repeat(spans, len(_DIRECTIONS))
    inside = ((probes >= 0) & (probes < xp.asarray(domain, dtype=np.int64))).all(axis=1)
    found = _locate(xp, probes[inside], origins, spans, domain)
    if bool((found < 0).any()):
        raise ValueError('incomplete_coverage: no leaf contains a balance probe')
    too_coarse = spans[found] > 2 * widths[inside]
    return xp.unique(found[too_coarse])


# Lattices up to this many finest voxels locate probes in a dense owner grid.
_DENSE_VOXELS = 2**24


def _locate(xp, points, origins, spans, domain):
    """Index of the leaf containing each finest voxel (-1 if none)."""
    if domain[0]*domain[1]*domain[2] <= _DENSE_VOXELS:
        owner = _owner_grid(xp, origins, spans, domain)
        return xp.astype(owner[points[:, 0], points[:, 1], points[:, 2]], np.int64)
    weights = xp.asarray([domain[1] * domain[2], domain[2], 1], dtype=np.int64)
    levels = xp.unique(spans)
    level_of = xp.searchsorted(levels, spans)
    keys = (origins * weights).sum(axis=1) * len(levels) + level_of
    order = xp.argsort(keys)
    sorted_keys = keys[order]
    result = xp.full(len(points), -1, dtype=np.int64)
    for level, width in enumerate(levels.tolist()):
        candidate = ((points // width) * width * weights).sum(axis=1) * len(levels) + level
        position = xp.minimum(xp.searchsorted(sorted_keys, candidate), len(sorted_keys) - 1)
        hit = sorted_keys[position] == candidate
        result[hit] = order[position[hit]]
    return result


def _owner_grid(xp, origins, spans, domain):
    """(domain) int32 grid of the leaf covering each finest voxel (-1 where none).

    Per span, leaves are marked on that span's coarse grid and upsampled, so
    the work is proportional to the lattice volume per level.
    """
    owner = xp.full(domain, -1, dtype=np.int32)
    for width in xp.unique(spans).tolist():
        rows = xp.flatnonzero(spans == width)
        coarse = xp.full(tuple(-(-d//width) for d in domain), -1, dtype=np.int32)
        cells = origins[rows]//width
        coarse[cells[:, 0], cells[:, 1], cells[:, 2]] = xp.astype(rows, np.int32)
        fine = xp.repeat(xp.repeat(xp.repeat(coarse, width, axis=0), width, axis=1), width, axis=2)
        fine = fine[:domain[0], :domain[1], :domain[2]]
        owner = xp.where(fine >= 0, fine, owner)
    return owner


def balance_joint_leaves(origins, spans, domain_shape, samples, sample_nodes):
    """Balanced ``(origins, spans, samples, report)`` for joint extraction.

    ``samples`` (S, N, 8) are raw corner values per exported surface;
    ``sample_nodes(nodes)`` returns raw values (S, M) at host integer lattice
    nodes (M, 3) and is called at most once, only for nodes no leaf already
    has. An already balanced input is returned unchanged (samples as float64).
    """
    xp = namespace(origins, spans, samples)
    samples = xp.asarray(samples, dtype=np.float64)
    balanced_origins, balanced_spans, unsplit, split_count = balance_leaves(origins, spans, domain_shape)
    report = dict(split_leaves=split_count, added_leaves=len(balanced_spans) - len(spans), queried_nodes=0)
    if not split_count:
        return origins, spans, samples, report
    original = xp.asarray(origins, dtype=np.int64)
    spans = xp.asarray(spans, dtype=np.int64)
    stride = [int(d) + 1 for d in host(domain_shape)]
    weights = xp.asarray([stride[1] * stride[2], stride[2], 1], dtype=np.int64)
    corners = xp.asarray(CORNERS, dtype=np.int64)
    known_codes, first = xp.unique((leaf_corner_nodes(original, spans, corners) * weights).sum(-1).reshape(-1),
                                   return_index=True)
    known_values = samples.reshape(len(samples), -1)[:, first]
    new = xp.flatnonzero(unsplit < 0)
    new_nodes = leaf_corner_nodes(balanced_origins[new], balanced_spans[new], corners).reshape(-1, 3)
    new_codes = (new_nodes * weights).sum(-1)
    position = xp.minimum(xp.searchsorted(known_codes, new_codes), len(known_codes) - 1)
    known = known_codes[position] == new_codes
    values = xp.empty((len(samples), len(new_codes)))
    values[:, known] = known_values[:, position[known]]
    if not bool(known.all()):
        query_codes, query_first, query_inverse = xp.unique(new_codes[~known], return_index=True,
                                                            return_inverse=True)
        queried = np.asarray(sample_nodes(host(new_nodes[~known][query_first])), dtype=np.float64)
        if queried.shape != (len(samples), len(query_codes)) or not np.isfinite(queried).all():
            raise ValueError('invalid_balance_samples: finite (S, M) raw values required')
        values[:, ~known] = xp.asarray(queried)[:, query_inverse]
        report['queried_nodes'] = len(query_codes)
    balanced_samples = xp.empty((len(samples), len(balanced_spans), 8))
    balanced_samples[:, unsplit >= 0] = samples[:, unsplit[unsplit >= 0]]
    balanced_samples[:, new] = values.reshape(len(samples), len(new), 8)
    return balanced_origins, balanced_spans, balanced_samples, report
