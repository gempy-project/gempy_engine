"""Sparse, array-only incidence for a fully covered, face/edge-balanced octree."""

from itertools import product

import numpy as np


# Same cyclic quadrant order as topology_extraction.INCIDENT_OFFSETS.
_INCIDENT_OFFSETS = (
    ((0, -1, -1), (0, -1, 0), (0, 0, 0), (0, 0, -1)),
    ((-1, 0, -1), (-1, 0, 0), (0, 0, 0), (0, 0, -1)),
    ((-1, -1, 0), (-1, 0, 0), (0, 0, 0), (0, -1, 0)),
)


def build_adaptive_complex(origins, spans, domain_shape) -> dict:
    """Build canonical face tiles and minimal original-edge segments.

    Inputs are integer arrays of shapes (N, 3), (N,), and (3,). Leaves
    are aligned dyadic cubes in [0, domain_shape), in finest-lattice units.
    Cell IDs are input row indices; inputs are not modified. The domain may
    contain multiple roots and need not itself be a cube or power of two.

    Returns exactly ``{'faces': [...], 'edges': [...]}``. Each record has
    ``axis`` (0/1/2), ``origin`` (integer tuple of length 3), ``span`` (int),
    and ``cells``. Face cells are (negative-side, positive-side) for interior
    tiles and (owner, None) for boundaries. Edge cells are a four-entry cyclic
    quadrant ring matching topology_extraction.INCIDENT_OFFSETS; exterior
    quadrants are None and coarse cells may occur twice. Both lists are sorted
    by (axis, origin, span), independent of input order except for cell IDs.

    Faces split only at actual shared leaf faces. Edges partition the union
    of ORIGINAL leaf edges at all collinear endpoints, including fine edges
    hanging inside coarse faces. No virtual leaves, resampling, or triangles
    are introduced. Face/edge depth differences above one raise ValueError
    with ``unsupported_unbalanced_octree``; corner-only differences are allowed.
    Storage is proportional to leaves and incidence, never domain volume.
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
