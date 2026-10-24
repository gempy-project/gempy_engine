"""Sparse, array-only incidence for a fully covered, face/edge-balanced octree."""

import contextlib
import gc
from itertools import product
from typing import NamedTuple

import numpy as np

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


class _HostFallback(Exception):
    """Inputs the device path does not order exactly (invalid or overflowing codes)."""


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
    # Exactness guard: every quantity below must stay far inside int64.
    largest_value = max(abs(int(origins.min())), abs(int(origins.max())), abs(int(spans.min())),
                        abs(int(spans.max())), *domain)
    positive = sorted({int(w) for w in xp.unique(spans).tolist() if w > 0})
    level_sizes = [(-(-domain[0] // w)) * (-(-domain[1] // w)) * (-(-domain[2] // w)) for w in positive]
    if 8 * largest_value >= _INT64_SAFE or any(size >= _INT64_SAFE for size in level_sizes):
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

    levels = xp.unique(w).tolist()
    level_codes, level_cells, level_dims = [], [], []
    for width in levels:
        dims = tuple(-(-d // width) for d in domain)
        rows = xp.flatnonzero(w == width)
        codes = _codes(o[rows] // width, dims)
        order = xp.argsort(codes)
        level_codes.append(codes[order])
        level_cells.append(rows[order])
        level_dims.append(dims)
    largest = levels[-1]
    for i, width in enumerate(levels):
        for j in range(i + 1, len(levels)):
            ancestor = levels[j]
            rows = xp.flatnonzero(w == width)
            if len(rows) and bool(_member(xp, _codes(o[rows] // ancestor, level_dims[j]), level_codes[j]).any()):
                raise ValueError('overlapping_leaves: leaf and ancestor both present')
    widths_found, counts = xp.unique(w, return_counts=True)
    if sum(int(c) * int(v) ** 3 for v, c in zip(widths_found.tolist(), counts.tolist())) != \
            domain[0] * domain[1] * domain[2]:
        raise ValueError('incomplete_coverage: leaves must cover the full rectangular domain')
    quarter = int(levels[0])

    def locate(points4):
        """Leaf row containing each quarter-lattice probe; -1 outside the domain."""
        result = xp.full(len(points4), -1, dtype=np.int64)
        inside = ((points4 >= 0) & (points4 < 4 * extent)).all(axis=1)
        pending = xp.flatnonzero(inside)
        for width, codes, cells, dims in zip(levels, level_codes, level_cells, level_dims):
            if not len(pending):
                break
            probe = _codes(points4[pending] // (4 * width), dims)
            found = xp.minimum(xp.searchsorted(codes, probe), len(codes) - 1)
            hit = codes[found] == probe
            result[pending[hit]] = cells[found[hit]]
            pending = pending[~hit]
        if len(pending):
            raise ValueError('incomplete_coverage: no leaf contains an interior probe')
        return result

    # region Faces
    face_axis, face_origin, face_size, face_negative, face_positive = [], [], [], [], []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for side in (0, 1):
            lower = xp.copy(o)
            lower[:, axis] += side * w
            neighbor = _face_neighbors(locate, lower, w, axis, other, side, quarter)
            _check_balance(xp, w, neighbor, w, 'face')
            split = (neighbor >= 0) & (xp.where(neighbor >= 0, w[xp.maximum(neighbor, 0)], 0) < w)
            whole = ~split
            tiles = [lower[whole]]
            sizes = [w[whole]]
            owners = [xp.flatnonzero(whole)]
            neighbors = [neighbor[whole]]
            rows = xp.flatnonzero(split)
            if len(rows):
                half = w[rows] // 2
                for du, dv in product((0, 1), repeat=2):
                    child = xp.copy(lower[rows])
                    child[:, other[0]] += du * half
                    child[:, other[1]] += dv * half
                    child_neighbor = _face_neighbors(locate, child, half, axis, other, side, quarter)
                    _check_balance(xp, w[rows], child_neighbor, w, 'face')
                    if bool(((child_neighbor >= 0) & (w[xp.maximum(child_neighbor, 0)] < half)).any()):
                        raise ValueError('unsupported_unbalanced_octree: face depth difference exceeds one')
                    tiles.append(child)
                    sizes.append(half)
                    owners.append(rows)
                    neighbors.append(child_neighbor)
            tiles, sizes = xp.concatenate(tiles), xp.concatenate(sizes)
            owners, neighbors = xp.concatenate(owners), xp.concatenate(neighbors)
            # (negative-side, positive-side); boundary tiles are (owner, None).
            negative = xp.where((neighbors < 0) | (side == 1), owners, neighbors)
            positive = xp.where(neighbors < 0, -1, owners if side == 0 else neighbors)
            face_axis.append(xp.full(len(tiles), axis, dtype=np.int64))
            face_origin.append(tiles)
            face_size.append(sizes)
            face_negative.append(negative)
            face_positive.append(positive)
    face_axis, face_origin, face_size = map(xp.concatenate, (face_axis, face_origin, face_size))
    face_negative, face_positive = xp.concatenate(face_negative), xp.concatenate(face_positive)
    order = _sort_order(xp, (face_axis, *face_origin.T, face_size), (3, *(d+1 for d in domain), largest+1))
    key = xp.concatenate((face_axis[:, None], face_origin, face_size[:, None]), axis=1)[order]
    cells = xp.stack((face_negative, face_positive), axis=1)[order]
    starts = xp.ones(len(key), dtype=bool)
    starts[1:] = (key[1:] != key[:-1]).any(axis=1)
    group = xp.cumsum(xp.astype(starts, np.int64)) - 1
    if bool((cells != cells[xp.flatnonzero(starts)][group]).any()):
        raise ValueError('inconsistent_face_incidence: adjacent references disagree')
    key, cells = key[starts], cells[starts]
    # endregion

    # region Edges: sweep the union of original leaf edges on every grid line
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
    seg_lower = xp.zeros((len(segment), 3), dtype=np.int64)
    rings = xp.zeros((len(segment), 4), dtype=np.int64)
    for axis in range(3):
        rows = xp.flatnonzero(seg_axis == axis)
        if not len(rows):
            continue
        other = [a for a in range(3) if a != axis]
        seg_lower[rows, axis] = seg_start[rows]
        for column, a in enumerate(other):
            seg_lower[rows, a] = events[segment[rows], 1 + column]
        midpoint = 4 * seg_lower[rows]
        midpoint[:, axis] += 2 * (seg_end[rows] - seg_start[rows])
        for k, offsets in enumerate(_INCIDENT_OFFSETS[axis]):
            probe = xp.copy(midpoint)
            for a in other:
                probe[:, a] += quarter if offsets[a] == 0 else -quarter
            rings[rows, k] = locate(probe)
    present = rings >= 0
    ring_widths = xp.where(present, w[xp.maximum(rings, 0)], 0)
    widest = xp.max(ring_widths, axis=1)
    narrowest = xp.min(xp.where(present, ring_widths, INT64_MAX), axis=1)
    if bool((widest > 2 * narrowest).any()):
        raise ValueError('unsupported_unbalanced_octree: edge depth difference exceeds one')
    seg_span = seg_end - seg_start
    order = _sort_order(xp, (seg_axis, *seg_lower.T, seg_span), (3, *(d+1 for d in domain), largest+1))
    # endregion

    return AdaptiveComplex(xp.copy(key[:, 0]), xp.copy(key[:, 1:4]), xp.copy(key[:, 4]), cells,
                           seg_axis[order], seg_lower[order], seg_span[order], rings[order])


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


def _face_neighbors(locate, tiles, sizes, axis, other, side, quarter):
    probe = 4 * tiles
    probe[:, axis] += quarter if side else -quarter
    for a in other:
        probe[:, a] += 2 * sizes
    return locate(probe)


def _check_balance(xp, width, neighbor, widths, kind):
    present = neighbor >= 0
    neighbor_width = widths[xp.maximum(neighbor, 0)]
    if bool((present & (xp.maximum(width, neighbor_width) > 2 * xp.minimum(width, neighbor_width))).any()):
        raise ValueError(f'unsupported_unbalanced_octree: {kind} depth difference exceeds one')
