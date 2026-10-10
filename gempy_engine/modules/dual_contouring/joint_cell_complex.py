"""Sparse, array-only incidence for a fully covered, face/edge-balanced octree."""

import contextlib
import gc
from itertools import product

import numpy as np


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


def build_adaptive_complex(origins, spans, domain_shape) -> dict:
    """Vectorised ``_build_adaptive_complex_reference``: same records, order and errors.

    Point location uses per-level mixed-radix lattice codes and ``searchsorted``
    instead of per-probe dictionary lookups. If any code or quarter-lattice probe
    could exceed int64, the exact pure-Python reference is used instead.
    """
    arrays = [np.asarray(value) for value in (origins, spans, domain_shape)]
    origins, spans, domain_shape = arrays
    if (origins.ndim != 2 or origins.shape[1:] != (3,) or
            spans.shape != (len(origins),) or domain_shape.shape != (3,)):
        raise ValueError('invalid_shape: expected origins (N,3), spans (N,), domain_shape (3,)')
    if any(array.dtype.kind not in 'iu' for array in arrays):
        raise ValueError('invalid_integer_input: lattice coordinates and spans must be integers')
    domain = tuple(map(int, domain_shape))
    if not len(origins) or any(size <= 0 for size in domain):
        raise ValueError('invalid_domain: require nonempty leaves and positive domain_shape')
    # Exactness guard: every quantity below must stay far inside int64.
    largest_value = max(abs(int(origins.min())), abs(int(origins.max())), abs(int(spans.min())),
                        abs(int(spans.max())), *domain)
    positive = sorted({int(w) for w in spans.tolist() if w > 0})
    level_sizes = [(-(-domain[0] // w)) * (-(-domain[1] // w)) * (-(-domain[2] // w)) for w in positive]
    if 8 * largest_value >= _INT64_SAFE or any(size >= _INT64_SAFE for size in level_sizes):
        return _build_adaptive_complex_reference(origins, spans, domain_shape)

    o = origins.astype(np.int64)
    w = spans.astype(np.int64)
    extent = np.array(domain, dtype=np.int64)
    n = len(o)

    # region Per-leaf validation, original precedence (first failing leaf, first check)
    bad_span = (w <= 0) | ((w & (w - 1)) != 0)
    safe = np.where(bad_span, 1, w)
    unaligned = np.any(o % safe[:, None] != 0, axis=1)
    outside = np.any((o < 0) | (o + safe[:, None] > extent), axis=1)
    keys = np.column_stack((w, o))
    _, first_rows, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    duplicate = first_rows[inverse.reshape(-1)] != np.arange(n)
    checks = (
        (bad_span, 'invalid_span: spans must be positive powers of two'),
        (unaligned, 'unaligned_origin: origins must be multiples of their spans'),
        (outside, 'outside_domain: leaf exceeds domain_shape'),
        (duplicate, 'overlapping_leaves: duplicate leaf'),
    )
    failing = np.flatnonzero(np.any([mask for mask, _ in checks], axis=0))
    if len(failing):
        leaf = failing[0]
        raise ValueError(next(message for mask, message in checks if mask[leaf]))
    # endregion

    levels = sorted(set(w.tolist()))
    level_codes, level_cells, level_dims = [], [], []
    for width in levels:
        dims = -(-extent // width)
        rows = np.flatnonzero(w == width)
        codes = _codes(o[rows] // width, dims)
        order = np.argsort(codes, kind='stable')
        level_codes.append(codes[order])
        level_cells.append(rows[order])
        level_dims.append(dims)
    largest = levels[-1]
    for i, width in enumerate(levels):
        for j in range(i + 1, len(levels)):
            ancestor = levels[j]
            rows = np.flatnonzero(w == width)
            if len(rows) and np.any(_member(_codes(o[rows] // ancestor, level_dims[j]), level_codes[j])):
                raise ValueError('overlapping_leaves: leaf and ancestor both present')
    widths_found, counts = np.unique(w, return_counts=True)
    if sum(int(c) * int(v) ** 3 for v, c in zip(widths_found, counts)) != domain[0] * domain[1] * domain[2]:
        raise ValueError('incomplete_coverage: leaves must cover the full rectangular domain')
    quarter = int(levels[0])
    del largest

    def locate(points4):
        """Leaf row containing each quarter-lattice probe; -1 outside the domain."""
        result = np.full(len(points4), -1, dtype=np.int64)
        inside = np.all((points4 >= 0) & (points4 < 4 * extent), axis=1)
        pending = np.flatnonzero(inside)
        for width, codes, cells, dims in zip(levels, level_codes, level_cells, level_dims):
            if not len(pending):
                break
            probe = _codes(points4[pending] // (4 * width), dims)
            found = np.minimum(np.searchsorted(codes, probe), len(codes) - 1)
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
            lower = o.copy()
            lower[:, axis] += side * w
            neighbor = _face_neighbors(locate, lower, w, axis, other, side, quarter)
            _check_balance(w, neighbor, w, 'face')
            split = (neighbor >= 0) & (np.where(neighbor >= 0, w[np.maximum(neighbor, 0)], 0) < w)
            whole = ~split
            tiles = [lower[whole]]
            sizes = [w[whole]]
            owners = [np.flatnonzero(whole)]
            neighbors = [neighbor[whole]]
            rows = np.flatnonzero(split)
            if len(rows):
                half = w[rows] // 2
                for du, dv in product((0, 1), repeat=2):
                    child = lower[rows].copy()
                    child[:, other[0]] += du * half
                    child[:, other[1]] += dv * half
                    child_neighbor = _face_neighbors(locate, child, half, axis, other, side, quarter)
                    _check_balance(w[rows], child_neighbor, w, 'face')
                    if np.any((child_neighbor >= 0) & (w[np.maximum(child_neighbor, 0)] < half)):
                        raise ValueError('unsupported_unbalanced_octree: face depth difference exceeds one')
                    tiles.append(child)
                    sizes.append(half)
                    owners.append(rows)
                    neighbors.append(child_neighbor)
            tiles, sizes = np.concatenate(tiles), np.concatenate(sizes)
            owners, neighbors = np.concatenate(owners), np.concatenate(neighbors)
            # (negative-side, positive-side); boundary tiles are (owner, None).
            negative = np.where((neighbors < 0) | (side == 1), owners, neighbors)
            positive = np.where(neighbors < 0, -1, np.where(side == 1, neighbors, owners))
            face_axis.append(np.full(len(tiles), axis))
            face_origin.append(tiles)
            face_size.append(sizes)
            face_negative.append(negative)
            face_positive.append(positive)
    face_axis, face_origin, face_size = map(np.concatenate, (face_axis, face_origin, face_size))
    face_negative, face_positive = np.concatenate(face_negative), np.concatenate(face_positive)
    order = np.lexsort((face_size, face_origin[:, 2], face_origin[:, 1], face_origin[:, 0], face_axis))
    key = np.column_stack((face_axis, face_origin, face_size))[order]
    cells = np.column_stack((face_negative, face_positive))[order]
    starts = np.ones(len(key), dtype=bool)
    starts[1:] = np.any(key[1:] != key[:-1], axis=1)
    group = np.cumsum(starts) - 1
    if np.any(np.any(cells != cells[np.flatnonzero(starts)][group], axis=1)):
        raise ValueError('inconsistent_face_incidence: adjacent references disagree')
    key, cells = key[starts], cells[starts]
    # endregion

    # region Edges: sweep the union of original leaf edges on every grid line
    line_axis, line_t, line_coord, line_delta = [], [], [], []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for du, dv in product((0, 1), repeat=2):
            transverse = np.column_stack((o[:, other[0]] + du * w, o[:, other[1]] + dv * w))
            for coord, delta in ((o[:, axis], 1), (o[:, axis] + w, -1)):
                line_axis.append(np.full(n, axis))
                line_t.append(transverse)
                line_coord.append(coord)
                line_delta.append(np.full(n, delta))
    line_axis, line_t = np.concatenate(line_axis), np.concatenate(line_t)
    line_coord, line_delta = np.concatenate(line_coord), np.concatenate(line_delta)
    order = np.lexsort((line_coord, line_t[:, 1], line_t[:, 0], line_axis))
    events = np.column_stack((line_axis, line_t, line_coord))[order]
    line_delta = line_delta[order]
    new_event = np.ones(len(events), dtype=bool)
    new_event[1:] = np.any(events[1:] != events[:-1], axis=1)
    event_rows = np.flatnonzero(new_event)
    events = events[event_rows]
    deltas = np.add.reduceat(line_delta, event_rows)
    new_line = np.ones(len(events), dtype=bool)
    new_line[1:] = np.any(events[1:, :3] != events[:-1, :3], axis=1)
    total = np.cumsum(deltas)
    line_start = np.flatnonzero(new_line)
    before_line = np.concatenate(([0], total[line_start[1:] - 1]))
    active = total - np.repeat(before_line, np.diff(np.append(line_start, len(events))))
    segment = np.flatnonzero((active[:-1] != 0) & ~new_line[1:])
    seg_axis = events[segment, 0]
    seg_start, seg_end = events[segment, 3], events[segment + 1, 3]
    seg_lower = np.zeros((len(segment), 3), dtype=np.int64)
    rings = np.zeros((len(segment), 4), dtype=np.int64)
    for axis in range(3):
        rows = np.flatnonzero(seg_axis == axis)
        if not len(rows):
            continue
        other = [a for a in range(3) if a != axis]
        seg_lower[rows, axis] = seg_start[rows]
        seg_lower[rows[:, None], other] = events[segment[rows]][:, 1:3]
        midpoint = 4 * seg_lower[rows]
        midpoint[:, axis] += 2 * (seg_end[rows] - seg_start[rows])
        for k, offsets in enumerate(_INCIDENT_OFFSETS[axis]):
            probe = midpoint.copy()
            for a in other:
                probe[:, a] += quarter if offsets[a] == 0 else -quarter
            rings[rows, k] = locate(probe)
    present = rings >= 0
    ring_widths = np.where(present, w[np.maximum(rings, 0)], 0)
    widest = ring_widths.max(axis=1)
    narrowest = np.where(present, ring_widths, np.iinfo(np.int64).max).min(axis=1)
    if np.any(widest > 2 * narrowest):
        raise ValueError('unsupported_unbalanced_octree: edge depth difference exceeds one')
    seg_span = seg_end - seg_start
    order = np.lexsort((seg_span, seg_lower[:, 2], seg_lower[:, 1], seg_lower[:, 0], seg_axis))
    # endregion

    # Records are the public contract: Python ints, None for exterior cells.
    face_cells = cells.astype(object)
    face_cells[cells < 0] = None
    ring_cells = rings[order].astype(object)
    ring_cells[rings[order] < 0] = None
    with _paused_gc():
        return {
            'faces': [{'axis': a, 'origin': (x, y, z), 'span': size, 'cells': pair}
                      for a, x, y, z, size, pair in zip(*key.T.tolist(), map(tuple, face_cells.tolist()))],
            'edges': [{'axis': a, 'origin': (x, y, z), 'span': size, 'cells': ring}
                      for a, x, y, z, size, ring in zip(seg_axis[order].tolist(), *seg_lower[order].T.tolist(),
                                                        seg_span[order].tolist(), map(tuple, ring_cells.tolist()))],
        }


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


def _codes(coordinates, dims):
    """Mixed-radix code of integer lattice coordinates (exact; guarded against overflow)."""
    return (coordinates[:, 0] * dims[1] + coordinates[:, 1]) * dims[2] + coordinates[:, 2]


def _member(values, sorted_codes):
    found = np.minimum(np.searchsorted(sorted_codes, values), len(sorted_codes) - 1)
    return sorted_codes[found] == values


def _face_neighbors(locate, tiles, sizes, axis, other, side, quarter):
    probe = 4 * tiles
    probe[:, axis] += quarter if side else -quarter
    for a in other:
        probe[:, a] += 2 * sizes
    return locate(probe)


def _check_balance(width, neighbor, widths, kind):
    present = neighbor >= 0
    neighbor_width = widths[np.maximum(neighbor, 0)]
    if np.any(present & (np.maximum(width, neighbor_width) > 2 * np.minimum(width, neighbor_width))):
        raise ValueError(f'unsupported_unbalanced_octree: {kind} depth difference exceeds one')
