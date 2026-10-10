"""Leaf corner numbering and sampled branch classification for joint extraction."""

from itertools import product

import numpy as np


# Match production Hermite numbering and complete-quad cyclic cell order.
EDGE_START = np.array([4, 5, 6, 7, 2, 3, 6, 7, 1, 3, 5, 7])
EDGE_END = np.array([0, 1, 2, 3, 0, 1, 4, 5, 0, 2, 4, 6])
EDGE_OFFSETS = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                         [0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1],
                         [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]])
INCIDENT_OFFSETS = np.array([
    [[0, -1, -1], [0, -1, 0], [0, 0, 0], [0, 0, -1]],
    [[-1, 0, -1], [-1, 0, 0], [0, 0, 0], [0, 0, -1]],
    [[-1, -1, 0], [-1, 0, 0], [0, 0, 0], [0, -1, 0]],
])
CORNERS = np.array(list(product((0, 1), repeat=3)))


def classify_cell_branches(values, valid_edges, unsafe_warnings=None, skip=None):
    """Classify sampled boundary contours without prescribing Hermite normals.

    Affine approximants, bilinear extrusions and single-branch strictly monotone
    sampled cells have supported connectivity. Other interiors need more data.
    Face saddle ties are rejected deterministically, never arbitrarily joined.
    ``skip`` marks cells whose geometry the caller discards; they are neither
    checked nor labelled (count zero).
    """
    tolerance = 1e-10
    design = np.column_stack((CORNERS, np.ones(8)))
    coefficients = np.linalg.lstsq(design, values.T, rcond=None)[0].T
    affine = np.max(np.abs(coefficients @ design.T - values), axis=1) < tolerance
    labels = np.full(valid_edges.shape, -1, dtype=int)
    counts = np.zeros(len(values), dtype=int)
    edge_lookup = {frozenset((a, b)): i for i, (a, b) in enumerate(zip(EDGE_START, EDGE_END))}
    face_nodes = []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for side in (0, 1):
            face = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                bit = np.zeros(3, dtype=int)
                bit[axis], bit[other[0]], bit[other[1]] = side, u, v
                face.append(int(np.flatnonzero(np.all(CORNERS == bit, axis=1))[0]))
            face_nodes.append(face)
    ambiguous_count = 0
    # Per-cell predicates, vectorized; the loop visits only cells that can raise
    # or carry crossings, in the original order.
    near_zero = np.abs(values) < tolerance
    degenerate = near_zero.all(axis=1)
    cubes = values.reshape(-1, 2, 2, 2)
    extrusions = np.zeros(len(values), dtype=bool)
    monotones = np.zeros(len(values), dtype=bool)
    for a in range(3):
        extrusions |= np.isclose(np.take(cubes, 0, axis=a + 1), np.take(cubes, 1, axis=a + 1),
                                 atol=tolerance, rtol=0).all(axis=(1, 2))
        steps = np.diff(cubes, axis=a + 1).reshape(len(values), -1)
        monotones |= (steps > tolerance).all(axis=1) | (steps < -tolerance).all(axis=1)
    crossed_cells = valid_edges.any(axis=1)
    visit = degenerate | ~(affine | extrusions | monotones) | crossed_cells
    if skip is not None:
        visit &= ~np.asarray(skip, dtype=bool)
    for cell in np.flatnonzero(visit):
        if degenerate[cell]:
            raise ValueError('degenerate_field: entire cell lies on the interface')
        extrusion = extrusions[cell]
        if not affine[cell] and not extrusion and not monotones[cell]:
            raise ValueError('insufficient_interior_topology: unresolved sampled cell; supply refinement or interior evidence')
        if not crossed_cells[cell]:
            continue
        if near_zero[cell].any():
            raise ValueError('sample_aligned_interface: zero node needs a sided extraction rule')
        graph = {int(e): set() for e in np.flatnonzero(valid_edges[cell])}
        for nodes in face_nodes:
            face_edges = [edge_lookup[frozenset((a, b))] for a, b in zip(nodes, nodes[1:] + nodes[:1])]
            crossed = [i for i, e in enumerate(face_edges) if valid_edges[cell, e]]
            if len(crossed) == 2:
                connections = [crossed]
            elif len(crossed) == 4:
                ambiguous_count += 1
                f = values[cell, nodes]
                determinant = f[0] * f[2] - f[1] * f[3]
                if abs(determinant) <= tolerance:
                    if unsafe_warnings is None:
                        raise ValueError('ambiguous_face_tie: bilinear saddle is on the isosurface')
                    unsafe_warnings.append('ambiguous_face_tie: using raw determinant sign')
                connections = [(0, 1), (2, 3)] if determinant > 0 else [(0, 3), (1, 2)]
            elif not crossed:
                continue
            else:
                raise ValueError('unsupported_face_crossings: expected zero, two or four')
            for a, b in connections:
                ea, eb = face_edges[a], face_edges[b]
                graph[ea].add(eb)
                graph[eb].add(ea)
        remaining = set(graph)
        while remaining:
            pending = [min(remaining)]
            component = set()
            while pending:
                edge = pending.pop()
                if edge in component:
                    continue
                component.add(edge)
                pending.extend(graph[edge] - component)
            remaining -= component
            labels[cell, list(component)] = counts[cell]
            counts[cell] += 1
        if counts[cell] > 1 and not affine[cell] and not extrusion:
            raise ValueError('insufficient_interior_topology: multiple boundary contours need an interior connectivity certificate')
    return labels, counts, coefficients, affine, ambiguous_count
