"""Sparse, array-only incidence for a fully covered, face/edge-balanced octree."""

import contextlib
import gc
from itertools import product
from typing import NamedTuple

import numpy as np

from .joint_cell_branches import CORNERS, EDGE_END, EDGE_START
from .joint_xp import INT64_MAX, NUMPY, host, is_tensor, namespace


# Same cyclic quadrant order as joint_cell_branches.INCIDENT_OFFSETS.
_INCIDENT_OFFSETS = (
    ((0, -1, -1), (0, -1, 0), (0, 0, 0), (0, 0, -1)),
    ((-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1)),
    ((-1, -1, 0), (-1, 0, 0), (0, 0, 0), (0, -1, 0)),
)


def _build_adaptive_complex_reference(origins, spans, domain_shape) -> dict:
    """Build canonical face tiles and minimal original-edge segments.

    Inputs are integer arrays of shapes (N, 3), (N,), and (3,). Leaves
    are aligned dyadic cubes in [0, domain_shape), in finest-lattice units.
    Cell IDs are input row indices; inputs are not modified. The domain may
    contain multiple roots and need not itself be a cube or power of two.

    Returns exactly ``{'faces': [...], 'edges': [...]}``. Each record has
    ``axis`` (0/1/2), ``origin`` (integer tuple of length 3), ``span`` (int),
    and ``cells``. Face cells are (negative-side, positive-side) for interior
    tiles and (owner, None) for boundaries. Edge cells are a four-entry cyclic
    quadrant ring matching joint_cell_branches.INCIDENT_OFFSETS; exterior
    quadrants are None and coarse cells may occur twice. Both lists are sorted
    by (axis, origin, span), independent of input order except for cell IDs.

    Faces split only at actual shared leaf faces. Edges partition the union
    of ORIGINAL leaf edges at all collinear endpoints, including fine edges
    hanging inside coarse faces. No virtual leaves, resampling, or triangles
    are introduced. Face/edge depth differences above one raise ValueError
    with ``unsupported_unbalanced_octree``; corner-only differences are allowed.
    Storage is proportional to leaves and incidence, never domain volume.

    Pure-Python reference: the oracle for the vectorised ``build_adaptive_complex``
    and its exact fallback when lattice codes could overflow int64.
    """
    arrays = [np.asarray(value) for value in (origins, spans, domain_shape)]
    origins, spans, domain_shape = arrays
    if (origins.ndim != 2 or origins.shape[1:] != (3,) or
            spans.shape != (len(origins),) or domain_shape.shape != (3,)):
        raise ValueError('invalid_shape: expected origins (N,3), spans (N,), domain_shape (3,)')
    if any(array.dtype.kind not in 'iu' for array in arrays):
        raise ValueError('invalid_integer_input: lattice coordinates and spans must be integers')
    # Python integers keep volume and quarter-lattice arithmetic overflow-free.
    points = [tuple(map(int, row)) for row in origins]
    widths = list(map(int, spans))
    domain = tuple(map(int, domain_shape))
    if not points or any(size <= 0 for size in domain):
        raise ValueError('invalid_domain: require nonempty leaves and positive domain_shape')
    leaves = {}
    for cell, (origin, width) in enumerate(zip(points, widths)):
        if width <= 0 or width & (width - 1):
            raise ValueError('invalid_span: spans must be positive powers of two')
        if any(value % width for value in origin):
            raise ValueError('unaligned_origin: origins must be multiples of their spans')
        if any(value < 0 or value + width > size for value, size in zip(origin, domain)):
            raise ValueError('outside_domain: leaf exceeds domain_shape')
        key = (width, origin)
        if key in leaves:
            raise ValueError('overlapping_leaves: duplicate leaf')
        leaves[key] = cell
    largest = max(widths)
    for origin, width in zip(points, widths):
        ancestor = width * 2
        while ancestor <= largest:
            key = (ancestor, tuple(value // ancestor * ancestor for value in origin))
            if key in leaves:
                raise ValueError('overlapping_leaves: leaf and ancestor both present')
            ancestor *= 2
    if sum(width ** 3 for width in widths) != domain[0] * domain[1] * domain[2]:
        raise ValueError('incomplete_coverage: leaves must cover the full rectangular domain')
    levels = sorted(set(widths))
    quarter = min(widths)

    def containing(point4):
        # Exact quarter-lattice probes avoid floating-point ambiguity at large coordinates.
        if any(value < 0 or value >= 4 * size for value, size in zip(point4, domain)):
            return None
        for width in levels:
            origin = tuple(value // (4 * width) * width for value in point4)
            cell = leaves.get((width, origin))
            if cell is not None:
                return cell
        raise ValueError('incomplete_coverage: no leaf contains an interior probe')

    faces = {}
    lines = {}
    for cell, (origin, width) in enumerate(zip(points, widths)):
        for axis in range(3):
            other = [a for a in range(3) if a != axis]
            for side in (0, 1):
                lower = list(origin)
                lower[axis] += side * width
                pending = [(tuple(lower), width)]
                while pending:
                    tile, size = pending.pop()
                    probe = [4 * value for value in tile]
                    probe[axis] += quarter if side else -quarter
                    for a in other:
                        probe[a] += 2 * size
                    neighbor = containing(probe)
                    if neighbor is not None:
                        neighbor_width = widths[neighbor]
                        if max(width, neighbor_width) > 2 * min(width, neighbor_width):
                            raise ValueError('unsupported_unbalanced_octree: face depth difference exceeds one')
                        if neighbor_width < size:
                            half = size // 2
                            for offsets in product((0, half), repeat=2):
                                child = list(tile)
                                for a, offset in zip(other, offsets):
                                    child[a] += offset
                                pending.append((tuple(child), half))
                            continue
                    cells = ((cell, None) if neighbor is None else
                             (cell, neighbor) if side else (neighbor, cell))
                    key = (axis, tile, size)
                    if key in faces and faces[key] != cells:
                        raise ValueError('inconsistent_face_incidence: adjacent references disagree')
                    faces[key] = cells
            for offsets in product((0, width), repeat=2):
                lower = list(origin)
                for a, offset in zip(other, offsets):
                    lower[a] += offset
                line = (axis, tuple(lower[a] for a in other))
                events = lines.setdefault(line, {})
                start, end = origin[axis], origin[axis] + width
                events[start] = events.get(start, 0) + 1
                events[end] = events.get(end, 0) - 1

    edges = []
    for (axis, transverse), events in sorted(lines.items()):
        other = [a for a in range(3) if a != axis]
        endpoints = sorted(events)
        active = 0
        for start, end in zip(endpoints, endpoints[1:]):
            active += events[start]
            if not active:
                continue
            lower = [0, 0, 0]
            lower[axis] = start
            for a, value in zip(other, transverse):
                lower[a] = value
            midpoint = [4 * value for value in lower]
            midpoint[axis] += 2 * (end - start)
            ring = []
            for offsets in _INCIDENT_OFFSETS[axis]:
                probe = list(midpoint)
                for a in other:
                    probe[a] += quarter if offsets[a] == 0 else -quarter
                ring.append(containing(probe))
            incident_widths = [widths[cell] for cell in ring if cell is not None]
            if max(incident_widths) > 2 * min(incident_widths):
                raise ValueError('unsupported_unbalanced_octree: edge depth difference exceeds one')
            edges.append(dict(axis=axis, origin=tuple(lower), span=end - start, cells=tuple(ring)))
    edges.sort(key=lambda edge: (edge['axis'], edge['origin'], edge['span']))
    return {
        'faces': [dict(axis=axis, origin=origin, span=span, cells=faces[key])
                  for key in sorted(faces) for axis, origin, span in (key,)],
        'edges': edges,
    }


_INT64_SAFE = 2 ** 62


class AdaptiveComplex(NamedTuple):
    """Array form of the ``build_adaptive_complex`` records, in the same order.

    Faces: ``face_axis`` (F,), ``face_origin`` (F, 3), ``face_span`` (F,) and
    ``face_cells`` (F, 2) (negative side, positive side; -1 outside). Edges:
    ``edge_axis`` (E,), ``edge_origin`` (E, 3), ``edge_span`` (E,) and the cyclic
    leaf ring ``edge_cells`` (E, 4) (-1 outside). All int64.
    """
    face_axis: np.ndarray
    face_origin: np.ndarray
    face_span: np.ndarray
    face_cells: np.ndarray
    edge_axis: np.ndarray
    edge_origin: np.ndarray
    edge_span: np.ndarray
    edge_cells: np.ndarray


def build_adaptive_complex(origins, spans, domain_shape) -> dict:
    """Vectorised ``_build_adaptive_complex_reference``: same records, order and errors."""
    return complex_records(adaptive_complex_arrays(origins, spans, domain_shape))


def complex_records(complex_):
    """``{'faces': [...], 'edges': [...]}`` records of an ``AdaptiveComplex``."""
    complex_ = AdaptiveComplex(*(host(a) for a in complex_))
    face_cells = complex_.face_cells.astype(object)
    face_cells[complex_.face_cells < 0] = None
    ring_cells = complex_.edge_cells.astype(object)
    ring_cells[complex_.edge_cells < 0] = None
    with _paused_gc():
        return {
            'faces': [{'axis': a, 'origin': (x, y, z), 'span': size, 'cells': pair}
                      for a, x, y, z, size, pair in zip(complex_.face_axis.tolist(), *complex_.face_origin.T.tolist(),
                                                        complex_.face_span.tolist(), map(tuple, face_cells.tolist()))],
            'edges': [{'axis': a, 'origin': (x, y, z), 'span': size, 'cells': ring}
                      for a, x, y, z, size, ring in zip(complex_.edge_axis.tolist(), *complex_.edge_origin.T.tolist(),
                                                        complex_.edge_span.tolist(), map(tuple, ring_cells.tolist()))],
        }


def complex_arrays(records):
    """``AdaptiveComplex`` of ``{'faces', 'edges'}`` records."""
    def column(rows, name, width=None):
        values = [r[name] for r in rows]
        if name == 'cells':
            values = [[-1 if c is None else c for c in cells] for cells in values]
        return np.asarray(values, dtype=np.int64).reshape((len(rows),) + (() if width is None else (width,)))

    faces, edges = records['faces'], records['edges']
    return AdaptiveComplex(column(faces, 'axis'), column(faces, 'origin', 3), column(faces, 'span'),
                           column(faces, 'cells', 2), column(edges, 'axis'), column(edges, 'origin', 3),
                           column(edges, 'span'), column(edges, 'cells', 4))


def adaptive_complex_arrays(origins, spans, domain_shape) -> AdaptiveComplex:
    """Array form of ``_build_adaptive_complex_reference``: same rows, order and errors.

    Point location uses per-level mixed-radix lattice codes and ``searchsorted``
    instead of per-probe dictionary lookups. If any code or quarter-lattice probe
    could exceed int64, the exact pure-Python reference is used instead. Torch
    inputs give Torch outputs on their device; any input the device path cannot
    order exactly is resolved (and rejected) on the host.
    """
    xp = namespace(origins, spans)
    if xp.torch:
        try:
            return _adaptive_complex_arrays(xp, origins, spans, domain_shape)
        except _HostFallback:
            result = _adaptive_complex_arrays(NUMPY, host(origins), host(spans), domain_shape)
            return AdaptiveComplex(*(xp.asarray(a) for a in result))
    return _adaptive_complex_arrays(NUMPY, *(np.asarray(value) for value in (origins, spans)), domain_shape)


def partial_complex_arrays(origins, spans, domain_shape, face_leaves=None, leaf_edges=None) -> AdaptiveComplex:
    """Part of the full complex, in its order: the tiles of leaves ``face_leaves`` and the given leaf edges.

    For extraction, which needs only what its surfaces cross. ``face_leaves`` are leaf rows;
    ``leaf_edges`` are rows ``(leaves, edges)``, ``edges`` in 0..11 of ``EDGE_START``. Of those, only
    the ones that are minimal edges are kept. Each minimal edge is a whole edge of the finest leaf
    around it, so giving every leaf edge gives every edge. ``None`` means every leaf for faces and
    the full sweep for edges. Neighbours and rings are located against all leaves. Work is
    proportional to the rows given, plus the location tables. The leaves are trusted: only what
    the built rows touch is validated. If lattice codes could overflow, this is the full complex.
    """
    xp = namespace(origins, spans)
    domain = tuple(int(d) for d in host(domain_shape))
    if _codes_overflow(xp, origins, spans, domain):
        return adaptive_complex_arrays(origins, spans, domain_shape)
    o, w = xp.astype(origins, np.int64), xp.astype(spans, np.int64)
    every = xp.arange(len(o))
    deferred = []
    tables = _level_tables(xp, o, w, domain)
    locate = _locator(xp, tables, xp.asarray(domain, dtype=np.int64), deferred)
    key, cells = _face_records(xp, o, w, every if face_leaves is None else face_leaves, domain, tables, locate, deferred)
    edges = (_edge_records(xp, o, w, domain, tables, locate, deferred) if leaf_edges is None else
             _leaf_edge_records(xp, o, w, *leaf_edges, domain, tables, locate, deferred))
    _raise_first(xp, deferred)
    return AdaptiveComplex(xp.copy(key[:, 0]), xp.copy(key[:, 1:4]), xp.copy(key[:, 4]), cells, *edges)


class _HostFallback(Exception):
    """Inputs the device path does not order exactly (invalid or overflowing codes)."""


def _codes_overflow(xp, origins, spans, domain):
    """Exactness guard: every quantity of the build must stay far inside int64."""
    largest_value = max(abs(int(origins.min())), abs(int(origins.max())), abs(int(spans.min())),
                        abs(int(spans.max())), *domain)
    positive = sorted({int(w) for w in xp.unique(spans).tolist() if w > 0})
    level_sizes = [(-(-domain[0] // w)) * (-(-domain[1] // w)) * (-(-domain[2] // w)) for w in positive]
    return 8 * largest_value >= _INT64_SAFE or any(size >= _INT64_SAFE for size in level_sizes)


def _is_integer(a):
    if is_tensor(a):
        return not (a.dtype.is_floating_point or a.dtype.is_complex) and str(a.dtype) != 'torch.bool'
    return a.dtype.kind in 'iu'


def _adaptive_complex_arrays(xp, origins, spans, domain_shape) -> AdaptiveComplex:
    domain_shape = np.asarray(host(domain_shape))
    if (origins.ndim != 2 or tuple(origins.shape[1:]) != (3,) or
            tuple(spans.shape) != (len(origins),) or domain_shape.shape != (3,)):
        raise ValueError('invalid_shape: expected origins (N,3), spans (N,), domain_shape (3,)')
    if not all(_is_integer(array) for array in (origins, spans, domain_shape)):
        raise ValueError('invalid_integer_input: lattice coordinates and spans must be integers')
    domain = tuple(map(int, domain_shape))
    if not len(origins) or any(size <= 0 for size in domain):
        raise ValueError('invalid_domain: require nonempty leaves and positive domain_shape')
    if _codes_overflow(xp, origins, spans, domain):
        if xp.torch:
            raise _HostFallback
        return complex_arrays(_build_adaptive_complex_reference(origins, spans, domain_shape))

    o = xp.astype(origins, np.int64)
    w = xp.astype(spans, np.int64)
    extent = xp.asarray(domain, dtype=np.int64)
    n = len(o)

    # region Per-leaf validation, original precedence (first failing leaf, first check)
    bad_span = (w <= 0) | ((w & (w - 1)) != 0)
    safe = xp.where(bad_span, 1, w)
    unaligned = (o % safe[:, None] != 0).any(axis=1)
    outside = ((o < 0) | (o + safe[:, None] > extent)).any(axis=1)
    # Codes of out-of-range leaves are never trusted: an invalid span or origin fails first.
    span_radix = int(w.max())+1 if int(w.min()) >= 0 else None
    keys = (_ordering_code(xp, (w, *o.T), (span_radix, *domain)) if span_radix is not None
            and int(o.min()) >= 0 and bool((xp.max(o, axis=0) < extent).all()) else None)
    if keys is None:
        if xp.torch:
            raise _HostFallback
        keys = np.column_stack((w, o))
        _, first_rows, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
    else:
        _, first_rows, inverse = xp.unique(keys, return_index=True, return_inverse=True)
    duplicate = first_rows[inverse] != xp.arange(n)
    checks = (
        (bad_span, 'invalid_span: spans must be positive powers of two'),
        (unaligned, 'unaligned_origin: origins must be multiples of their spans'),
        (outside, 'outside_domain: leaf exceeds domain_shape'),
        (duplicate, 'overlapping_leaves: duplicate leaf'),
    )
    failing = xp.flatnonzero(bad_span | unaligned | outside | duplicate)
    if len(failing):
        leaf = int(failing[0])
        raise ValueError(next(message for mask, message in checks if bool(mask[leaf])))
    # endregion

    tables = _level_tables(xp, o, w, domain)
    for i, (width, _, _, _) in enumerate(tables):
        for ancestor, codes, _, dims in tables[i + 1:]:
            rows = xp.flatnonzero(w == width)
            if len(rows) and bool(_member(xp, _codes(o[rows] // ancestor, dims), codes).any()):
                raise ValueError('overlapping_leaves: leaf and ancestor both present')
    widths_found, counts = xp.unique(w, return_counts=True)
    if sum(int(c) * int(v) ** 3 for v, c in zip(widths_found.tolist(), counts.tolist())) != \
            domain[0] * domain[1] * domain[2]:
        raise ValueError('incomplete_coverage: leaves must cover the full rectangular domain')
    # Balance, coverage and incidence checks are read back together at the end (one host read).
    deferred = []
    locate = _locator(xp, tables, extent, deferred)
    key, cells = _face_records(xp, o, w, xp.arange(n), domain, tables, locate, deferred)
    edges = _edge_records(xp, o, w, domain, tables, locate, deferred)
    _raise_first(xp, deferred)
    return AdaptiveComplex(xp.copy(key[:, 0]), xp.copy(key[:, 1:4]), xp.copy(key[:, 4]), cells, *edges)


def _level_tables(xp, o, w, domain):
    """Per leaf width, ascending: ``(width, sorted codes, their leaf rows, lattice dims)``."""
    tables = []
    for width in xp.unique(w).tolist():
        dims = tuple(-(-d // width) for d in domain)
        rows = xp.flatnonzero(w == width)
        codes = _codes(o[rows] // width, dims)
        order = xp.argsort(codes)
        tables.append((width, codes[order], rows[order], dims))
    return tables


def _locator(xp, tables, extent, deferred):
    def locate(points4):
        """Leaf row containing each quarter-lattice probe; -1 outside the domain.

        Every level is probed for every point (leaves do not overlap, so at most one hits): no
        per-level compaction; the coverage check is deferred.
        """
        result = xp.full(len(points4), -1, dtype=np.int64)
        inside = ((points4 >= 0) & (points4 < 4 * extent)).all(axis=1)
        for width, codes, cells, dims in tables:
            probe = _codes(points4 // (4 * width), dims)
            found = xp.minimum(xp.searchsorted(codes, probe), len(codes) - 1)
            hit = inside & (result < 0) & (codes[found] == probe)
            result = xp.where(hit, cells[found], result)
        deferred.append(((inside & (result < 0)).any(), 'incomplete_coverage: no leaf contains an interior probe'))
        return result
    return locate


def _face_records(xp, o, w, rows, domain, tables, locate, deferred):
    """Sorted tiles of the leaves ``rows``: keys (F, 5) ``(axis, origin, span)`` and cells (F, 2).

    Every face of those leaves (6 per leaf, side-major as (axis, side)), located in one probe; then every
    quarter of the faces whose neighbour is finer, in a second. The faces' order does not matter:
    they are sorted by key and deduplicated.
    """
    if not len(rows):
        return xp.zeros((0, 5), dtype=np.int64), xp.zeros((0, 2), dtype=np.int64)
    a, s, n = o[rows], w[rows], len(rows)
    quarter, largest = tables[0][0], tables[-1][0]
    sides = np.array([(axis, side) for axis in range(3) for side in (0, 1)])
    unit = np.eye(3, dtype=np.int64)
    normal = unit[sides[:, 0]]  # (6, 3)
    plane = 1 - normal  # the two in-plane axes
    probe_offset = np.where(sides[:, 1:], quarter, -quarter) * normal  # into the neighbour
    lower = a[None] + xp.asarray(sides[:, 1:] * normal)[:, None, :] * s[None, :, None]  # (6, N, 3)
    neighbor = locate((4 * lower + xp.asarray(probe_offset)[:, None, :]
                       + xp.asarray(plane)[:, None, :] * (2 * s)[None, :, None]).reshape(-1, 3)).reshape(6, n)
    deferred.append(_balance_violation(xp, s, neighbor, w, 'face'))
    split = (neighbor >= 0) & (w[xp.maximum(neighbor, 0)] < s)
    whole_rows, split_rows = xp.flatnonzero(~split), xp.flatnonzero(split)  # rows of the (6 N) faces
    whole_kind, whole_leaf, split_kind, split_leaf = whole_rows // n, whole_rows % n, split_rows // n, split_rows % n
    half = s[split_leaf] // 2
    # Quarter (du, dv) of each split face: offsets along its two in-plane axes, as product((0, 1), repeat=2).
    quarters = np.array([[du * unit[a] + dv * unit[b] for du, dv in product((0, 1), repeat=2)]
                         for a, b in (np.flatnonzero(row) for row in plane)])  # (6, 4, 3)
    child = lower[split_kind, split_leaf][:, None, :] + half[:, None, None] * xp.asarray(quarters)[split_kind]
    child_neighbor = locate((4 * child + xp.asarray(probe_offset)[split_kind][:, None, :]
                             + xp.asarray(plane)[split_kind][:, None, :] * (2 * half)[:, None, None]).reshape(-1, 3))
    child_neighbor = child_neighbor.reshape(-1, 4)
    deferred.append(_balance_violation(xp, s[split_leaf][:, None], child_neighbor, w, 'face'))
    deferred.append((((child_neighbor >= 0) & (w[xp.maximum(child_neighbor, 0)] < half[:, None])).any(),
                     'unsupported_unbalanced_octree: face depth difference exceeds one'))
    kind = xp.concatenate((whole_kind, xp.repeat(split_kind, 4)))
    face_origin = xp.concatenate((lower.reshape(-1, 3)[whole_rows], child.reshape(-1, 3)))
    face_size = xp.concatenate((s[whole_leaf], xp.repeat(half, 4)))
    owners = rows[xp.concatenate((whole_leaf, xp.repeat(split_leaf, 4)))]
    neighbors = xp.concatenate((neighbor.reshape(-1)[whole_rows], child_neighbor.reshape(-1)))
    face_axis, face_side = xp.asarray(sides[:, 0])[kind], xp.asarray(sides[:, 1])[kind]
    # (negative-side, positive-side); boundary tiles are (owner, None).
    face_negative = xp.where((neighbors < 0) | (face_side == 1), owners, neighbors)
    face_positive = xp.where(neighbors < 0, -1, xp.where(face_side == 0, owners, neighbors))
    order = _sort_order(xp, (face_axis, *face_origin.T, face_size), (3, *(d+1 for d in domain), largest+1))
    key = xp.concatenate((face_axis[:, None], face_origin, face_size[:, None]), axis=1)[order]
    cells = xp.stack((face_negative, face_positive), axis=1)[order]
    starts = xp.ones(len(key), dtype=bool)
    starts[1:] = (key[1:] != key[:-1]).any(axis=1)
    group = xp.cumsum(xp.astype(starts, np.int64)) - 1
    deferred.append(((cells != cells[xp.flatnonzero(starts)][group]).any(),
                     'inconsistent_face_incidence: adjacent references disagree'))
    return key[starts], cells[starts]


def _edge_records(xp, o, w, domain, tables, locate, deferred):
    """Sorted minimal edges (axis, origin, span, ring): sweep the union of all leaf edges on every grid line."""
    n = len(o)
    quarter, largest = tables[0][0], tables[-1][0]
    line_axis, line_t, line_coord, line_delta = [], [], [], []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for du, dv in product((0, 1), repeat=2):
            transverse = xp.stack((o[:, other[0]] + du * w, o[:, other[1]] + dv * w), axis=1)
            for coord, delta in ((o[:, axis], 1), (o[:, axis] + w, -1)):
                line_axis.append(xp.full(n, axis, dtype=np.int64))
                line_t.append(transverse)
                line_coord.append(coord)
                line_delta.append(xp.full(n, delta, dtype=np.int64))
    line_axis, line_t = xp.concatenate(line_axis), xp.concatenate(line_t)
    line_coord, line_delta = xp.concatenate(line_coord), xp.concatenate(line_delta)
    bound = max(domain)+1
    order = _sort_order(xp, (line_axis, line_t[:, 0], line_t[:, 1], line_coord), (3, bound, bound, bound))
    events = xp.concatenate((line_axis[:, None], line_t, line_coord[:, None]), axis=1)[order]
    line_delta = line_delta[order]
    new_event = xp.ones(len(events), dtype=bool)
    new_event[1:] = (events[1:] != events[:-1]).any(axis=1)
    event_rows = xp.flatnonzero(new_event)
    events = events[event_rows]
    # Per-event sums of the sorted deltas (np.add.reduceat).
    running = xp.cumsum(line_delta)
    event_ends = xp.concatenate((event_rows[1:], xp.full(1, len(line_delta), dtype=np.int64))) - 1
    deltas = running[event_ends] - xp.concatenate((xp.zeros(1, dtype=np.int64), running[event_ends[:-1]]))
    new_line = xp.ones(len(events), dtype=bool)
    new_line[1:] = (events[1:, :3] != events[:-1, :3]).any(axis=1)
    total = xp.cumsum(deltas)
    line_start = xp.flatnonzero(new_line)
    before_line = xp.concatenate((xp.zeros(1, dtype=np.int64), total[line_start[1:] - 1]))
    line_end = xp.concatenate((line_start[1:], xp.full(1, len(events), dtype=np.int64)))
    active = total - xp.repeat(before_line, line_end - line_start)
    segment = xp.flatnonzero((active[:-1] != 0) & ~new_line[1:])
    seg_axis = events[segment, 0]
    seg_start, seg_end = events[segment, 3], events[segment + 1, 3]
    # Per axis: the edge's own axis, then its two transverse axes (the event columns 1 and 2).
    unit = np.eye(3, dtype=np.int64)
    axis_unit = xp.asarray(unit)[seg_axis]
    transverse = xp.asarray(np.stack([unit[[a for a in range(3) if a != axis]] for axis in range(3)]))[seg_axis]
    seg_lower = (seg_start[:, None] * axis_unit + events[segment, 1][:, None] * transverse[:, 0]
                 + events[segment, 2][:, None] * transverse[:, 1])
    seg_span = seg_end - seg_start
    rings = _edge_rings(xp, seg_axis, seg_lower, seg_span, quarter, locate)
    deferred.append(_ring_violation(xp, rings, w))
    order = _sort_order(xp, (seg_axis, *seg_lower.T, seg_span), (3, *(d+1 for d in domain), largest+1))
    return seg_axis[order], seg_lower[order], seg_span[order], rings[order]


def _leaf_edge_records(xp, o, w, leaves, edges, domain, tables, locate, deferred):
    """Sorted minimal edges among leaf edges ``(leaves, edges)``: those no ring leaf is narrower than."""
    if not len(leaves):
        empty = xp.zeros(0, dtype=np.int64)
        return empty, xp.zeros((0, 3), dtype=np.int64), empty, xp.zeros((0, 4), dtype=np.int64)
    quarter, largest = tables[0][0], tables[-1][0]
    axis = xp.asarray(np.argmax(CORNERS[EDGE_START] != CORNERS[EDGE_END], axis=1), dtype=np.int64)[edges]
    span = w[leaves]
    lower = o[leaves] + span[:, None] * xp.asarray(np.minimum(CORNERS[EDGE_START], CORNERS[EDGE_END]))[edges]
    order = _sort_order(xp, (axis, *lower.T, span), (3, *(d+1 for d in domain), largest+1))
    axis, lower, span = axis[order], lower[order], span[order]
    new = xp.ones(len(axis), dtype=bool)
    new[1:] = (axis[1:] != axis[:-1]) | (lower[1:] != lower[:-1]).any(axis=1) | (span[1:] != span[:-1])
    axis, lower, span = axis[new], lower[new], span[new]
    rings = _edge_rings(xp, axis, lower, span, quarter, locate)
    deferred.append(_ring_violation(xp, rings, w))
    minimal = ((rings < 0) | (w[xp.maximum(rings, 0)] >= span[:, None])).all(axis=1)
    return axis[minimal], lower[minimal], span[minimal], rings[minimal]


def _edge_rings(xp, axis, lower, span, quarter, locate):
    """(E, 4) cyclic leaf rings around edges' midpoints, in _INCIDENT_OFFSETS order; -1 outside."""
    midpoint = 4 * lower + (2 * span)[:, None] * xp.asarray(np.eye(3, dtype=np.int64))[axis]
    # The four quadrant probes around each edge (the axis' own offset is 0).
    ring_offsets = np.array([[[0 if a == edge_axis else quarter if offsets[a] == 0 else -quarter for a in range(3)]
                              for offsets in _INCIDENT_OFFSETS[edge_axis]] for edge_axis in range(3)])  # (3, 4, 3)
    return locate((midpoint[:, None, :] + xp.asarray(ring_offsets)[axis]).reshape(-1, 3)).reshape(-1, 4)


def _ring_violation(xp, rings, widths):
    """(flag, message): whether a ring holds leaves more than one level apart; not read back."""
    present = rings >= 0
    ring_widths = xp.where(present, widths[xp.maximum(rings, 0)], 0)
    widest = xp.max(ring_widths, axis=1)
    narrowest = xp.min(xp.where(present, ring_widths, INT64_MAX), axis=1)
    return (widest > 2 * narrowest).any(), 'unsupported_unbalanced_octree: edge depth difference exceeds one'


@contextlib.contextmanager
def _paused_gc():
    """Pause automatic cyclic GC while allocating many acyclic records.

    Each automatic collection scans the whole heap (hundreds of thousands of
    import-time objects), which dominates building tens of thousands of small
    dicts. The records hold no cycles, so reference counting frees them.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def _ordering_code(xp, columns, radices):
    """One int64 code ordering rows like ``np.lexsort`` of ``columns`` (most significant first).

    Every column must lie in ``[0, radix)``; ``None`` if the code could overflow int64.
    """
    total = 1
    for radix in radices:
        total *= int(radix)
    if total >= _INT64_SAFE:
        return None
    code = xp.zeros(len(columns[0]), dtype=np.int64)
    for column, radix in zip(columns, radices):
        code = code*int(radix) + column
    return code


def _sort_order(xp, columns, radices):
    """Stable order of rows by ``columns`` (most significant first), as ``np.lexsort``."""
    code = _ordering_code(xp, columns, radices)
    if code is None:
        return xp.asarray(np.lexsort([host(c) for c in columns[::-1]]))
    return xp.argsort(code)


def _codes(coordinates, dims):
    """Mixed-radix code of integer lattice coordinates (exact; guarded against overflow)."""
    return (coordinates[:, 0] * int(dims[1]) + coordinates[:, 1]) * int(dims[2]) + coordinates[:, 2]


def _member(xp, values, sorted_codes):
    found = xp.minimum(xp.searchsorted(sorted_codes, values), len(sorted_codes) - 1)
    return sorted_codes[found] == values


def _balance_violation(xp, width, neighbor, widths, kind):
    """(flag, message): whether a neighbor is more than one level coarser or finer; not read back."""
    present = neighbor >= 0
    neighbor_width = widths[xp.maximum(neighbor, 0)]
    return ((present & (xp.maximum(width, neighbor_width) > 2 * xp.minimum(width, neighbor_width))).any(),
            f'unsupported_unbalanced_octree: {kind} depth difference exceeds one')


def _raise_first(xp, deferred):
    """Raise the first failed (flag, message) check, reading all flags back at once."""
    if not deferred:
        return
    failed = host(xp.stack([xp.asarray(flag) for flag, _ in deferred]))
    for (_, message), bad in zip(deferred, failed):
        if bad:
            raise ValueError(message)
