"""2:1 face/edge balancing of adaptive joint leaves, before the cell complex is built.

The production octree only refines cells whose corners differ, so a leaf can be
two or more levels coarser than a face or edge neighbour. The cell complex
requires at most one level across faces and edges (corner-only differences are
allowed), so coarse leaves are split here until that holds. Existing leaves keep
their production corner samples; a split leaf's children reuse every corner
value already known at that lattice node and only new nodes are queried.
"""

from itertools import product

import numpy as np

from .joint_cell_branches import CORNERS
from .joint_lattice import leaf_corner_nodes

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
    origins = np.asarray(origins, dtype=np.int64).reshape(-1, 3)
    spans = np.asarray(spans, dtype=np.int64).reshape(-1)
    source = np.arange(len(spans), dtype=np.int64)
    domain = np.asarray(domain_shape, dtype=np.int64)
    split_count = 0
    while True:
        coarse = _coarse_neighbors(origins, spans, domain)
        if not len(coarse):
            return origins, spans, source, split_count
        split_count += len(coarse)
        keep = np.ones(len(spans), dtype=bool)
        keep[coarse] = False
        half = spans[coarse] // 2
        children = (origins[coarse][:, None, :] + half[:, None, None] * CORNERS).reshape(-1, 3)
        origins = np.concatenate((origins[keep], children))
        spans = np.concatenate((spans[keep], np.repeat(half, 8)))
        source = np.concatenate((source[keep], np.full(len(children), -1, dtype=np.int64)))


def _coarse_neighbors(origins, spans, domain):
    """Leaves more than twice as wide as a face/edge neighbour, probed from the finer side.

    A neighbour wider than a leaf covers that leaf's whole adjacent face or edge
    (dyadic nesting), so one finest-voxel probe per direction finds it.
    """
    probes = origins[:, None, :] + np.where(_DIRECTIONS > 0, spans[:, None, None],
                                            np.where(_DIRECTIONS < 0, -1, 0))
    probes = probes.reshape(-1, 3)
    widths = np.repeat(spans, len(_DIRECTIONS))
    inside = np.all((probes >= 0) & (probes < domain), axis=1)
    found = _locate(probes[inside], origins, spans, domain)
    if np.any(found < 0):
        raise ValueError('incomplete_coverage: no leaf contains a balance probe')
    too_coarse = spans[found] > 2 * widths[inside]
    return np.unique(found[too_coarse])


def _locate(points, origins, spans, domain):
    """Index of the leaf containing each finest voxel (-1 if none)."""
    weights = np.array([domain[1] * domain[2], domain[2], 1], dtype=np.int64)
    levels = np.unique(spans)
    level_of = np.searchsorted(levels, spans)
    keys = (origins * weights).sum(axis=1) * len(levels) + level_of
    order = np.argsort(keys)
    sorted_keys = keys[order]
    result = np.full(len(points), -1, dtype=np.int64)
    for level, width in enumerate(levels):
        candidate = ((points // width) * width * weights).sum(axis=1) * len(levels) + level
        position = np.minimum(np.searchsorted(sorted_keys, candidate), len(sorted_keys) - 1)
        hit = sorted_keys[position] == candidate
        result[hit] = order[position[hit]]
    return result


def balance_joint_leaves(origins, spans, domain_shape, samples, sample_nodes):
    """Balanced ``(origins, spans, samples, report)`` for joint extraction.

    ``samples`` (S, N, 8) are raw corner values per exported surface;
    ``sample_nodes(nodes)`` returns raw values (S, M) at integer lattice nodes
    (M, 3) and is called at most once, only for nodes no leaf already has.
    An already balanced input is returned unchanged.
    """
    samples = np.asarray(samples, dtype=np.float64)
    balanced_origins, balanced_spans, unsplit, split_count = balance_leaves(origins, spans, domain_shape)
    report = dict(split_leaves=split_count, added_leaves=len(balanced_spans) - len(spans), queried_nodes=0)
    if not split_count:
        return origins, spans, samples, report
    original = np.asarray(origins, dtype=np.int64)
    stride = np.asarray(domain_shape, dtype=np.int64) + 1
    weights = np.array([stride[1] * stride[2], stride[2], 1], dtype=np.int64)
    known_codes, first = np.unique((leaf_corner_nodes(original, spans, CORNERS) * weights).sum(-1).reshape(-1),
                                   return_index=True)
    known_values = samples.reshape(len(samples), -1)[:, first]
    new = np.flatnonzero(unsplit < 0)
    new_nodes = leaf_corner_nodes(balanced_origins[new], balanced_spans[new], CORNERS).reshape(-1, 3)
    new_codes = (new_nodes * weights).sum(-1)
    position = np.minimum(np.searchsorted(known_codes, new_codes), len(known_codes) - 1)
    known = known_codes[position] == new_codes
    values = np.empty((len(samples), len(new_codes)))
    values[:, known] = known_values[:, position[known]]
    if not known.all():
        query_codes, query_first, query_inverse = np.unique(new_codes[~known], return_index=True,
                                                            return_inverse=True)
        queried = np.asarray(sample_nodes(new_nodes[~known][query_first]), dtype=np.float64)
        if queried.shape != (len(samples), len(query_codes)) or not np.isfinite(queried).all():
            raise ValueError('invalid_balance_samples: finite (S, M) raw values required')
        values[:, ~known] = queried[:, query_inverse.reshape(-1)]
        report['queried_nodes'] = len(query_codes)
    balanced_samples = np.empty((len(samples), len(balanced_spans), 8))
    balanced_samples[:, unsplit >= 0] = samples[:, unsplit[unsplit >= 0]]
    balanced_samples[:, new] = values.reshape(len(samples), len(new), 8)
    return balanced_origins, balanced_spans, balanced_samples, report

