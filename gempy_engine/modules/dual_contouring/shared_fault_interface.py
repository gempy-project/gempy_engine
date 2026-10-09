"""One conforming fault interface shared by the two banks of a closed-volume complex.

Banked joint extraction gives each fault bank its own fault skin. Both skins
cover the same fault surface, but each bank triangulates it with its own
junctions (horizon-fault seams, rim contour points), so vertices of one bank
hang on edges of the other. This module replaces the two skins with one
triangulation that contains every vertex and constraint edge of both:

1. Bank 0's skin is the starting triangulation, in 2D coordinates of the
   fault plane (any fault that is a graph over that plane works the same way).
2. Bank 1's skin vertices that something else attaches to are inserted
   (coincident ones are welded).
3. Bank 1's constraint edges (seams, rim contour, unit changes) are recovered
   by edge flips, after inserting their crossings with bank 0's constraints.
4. Unconstrained edges are flipped to constrained Delaunay.
5. Every triangle is labelled with the unit of each bank it lies in, so it
   separates bank 0's unit from bank 1's unit directly.

Horizon and cap triangles whose edges gained vertices are split along them.
Nothing moves: inserted points are original vertices or exact constraint
crossings, so unit volumes are unchanged.
"""

from collections import defaultdict

import numpy as np

from .adaptive_closed_volumes import OUTSIDE, _triangulate_convex
from .planar_triangulation import PlaneTriangulation, _cross, _edge, _vertices_on_segment


def merge_fault_skins(parts, normal):
    """Bank-namespaced patches and positions with one shared fault interface.

    ``parts`` maps bank (0, 1) to ``build_closed_volumes`` outputs; ``normal``
    is the fault plane normal. Keys are ``(bank, key)``; crossings of the two
    banks' constraints are ``('fault_crossing', i)``.
    """
    if sorted(parts) != [0, 1]:
        raise ValueError('unsupported_shared_fault: expected exactly banks 0 and 1')
    normal = np.asarray(normal, dtype=float)[:3]
    normal = normal/np.linalg.norm(normal)
    positions = {(bank, key): np.asarray(xyz, dtype=float)
                 for bank, part in parts.items() for key, xyz in part['positions'].items()}
    skins, others = {}, []
    for bank, part in parts.items():
        skins[bank] = []
        for keys, kind, negative, positive in part['patches']:
            keys = tuple((bank, k) for k in keys)
            if kind == 'fault_skin':
                if (negative == OUTSIDE) == (positive == OUTSIDE):
                    raise ValueError('unsupported_shared_fault: fault skin must border the bank exterior')
                skins[bank].append((keys, positive if negative == OUTSIDE else negative,
                                    negative == OUTSIDE))
            else:
                others.append((keys, kind, negative, positive))
    plus = {bank: _unit_side(skins[bank], positions, normal) for bank in (0, 1)}
    if plus[0] == plus[1]:
        raise ValueError('unsupported_shared_fault: both banks lie on the same side of the fault')

    scale = max(np.ptp(np.array([positions[k] for keys, _, _ in skins[0] for k in keys]), axis=0).max(), 1e-300)
    plane = PlaneTriangulation(normal, 1e-9*scale)
    # Bank 0 skin is the starting triangulation.
    ids = {}
    for keys, _, _ in skins[0]:
        for key in keys:
            if key not in ids:
                ids[key] = plane.add_vertex(positions[key], key)
    for keys, _, _ in skins[0]:
        plane.add_triangle(*(ids[k] for k in keys), orient=True)
    used = {k for keys, _, _, _ in others for k in keys}
    attached = {_key_edge(keys[i], keys[(i+1) % 3]) for keys, _, _, _ in others for i in range(3)}
    for key in sorted(_constraints(skins[0], attached), key=repr):
        plane.constrained.add(_edge(ids[key[0]], ids[key[1]]))
    # Bank 1 vertices something attaches to, then its constraints.
    constraints1 = _constraints(skins[1], attached)
    needed = sorted({k for edge in constraints1 for k in edge} |
                    {k for keys, _, _ in skins[1] for k in keys if k in used}, key=repr)
    aliases = {}
    for key in needed:
        vertex = plane.insert(positions[key], key)
        if plane.keys[vertex] != key:
            aliases[key] = plane.keys[vertex]
        ids[key] = vertex
    for a, b in sorted(constraints1, key=repr):
        plane.recover(ids[a], ids[b])
    plane.make_delaunay()

    # Units of both banks for every shared triangle (wound along the normal).
    triangles = plane.triangles()
    labels = {bank: _locate_units(plane, triangles, skins[bank], positions) for bank in (0, 1)}
    below, above = (1, 0) if plus[0] else (0, 1)
    merged_positions = {plane.keys[v]: plane.xyz[v] for v in range(len(plane.keys))}
    positions.update(merged_positions)
    patches = []
    for t, tri in enumerate(triangles):
        keys = tuple(plane.keys[v] for v in tri)
        a, b, c = (positions[k] for k in keys)
        if np.cross(b-a, c-a) @ normal < 0:
            keys = (keys[0], keys[2], keys[1])
        patches.append((keys, 'fault_skin', labels[below][t], labels[above][t]))

    # Horizon and cap triangles take the shared vertices on their skin edges.
    vertex_xyz = np.array([plane.xyz[v] for v in range(len(plane.keys))])
    on_skin = set(merged_positions)
    split = 0
    for keys, kind, negative, positive in others:
        keys = tuple(aliases.get(k, k) for k in keys)
        ring = []
        for i, key in enumerate(keys):
            ring.append(key)
            following = keys[(i+1) % 3]
            if key in on_skin and following in on_skin:
                inner = _vertices_on_segment(vertex_xyz, positions[key], positions[following], plane.tolerance)
                ring.extend(plane.keys[v] for v in inner if plane.keys[v] not in (key, following))
        if len(ring) == 3:
            patches.append((keys, kind, negative, positive))
            continue
        split += 1
        patches.extend((tri, kind, negative, positive) for tri in _triangulate_convex(ring, positions))
    report = dict(shared_fault_triangles=len(triangles), bank_skin_triangles={b: len(skins[b]) for b in (0, 1)},
                  welded_vertices=len(aliases), inserted_vertices=len(needed)-len(aliases),
                  constraint_crossings=plane.crossings, flips=plane.flips, split_triangles=split)
    return patches, positions, report


def _unit_side(skin, positions, normal):
    """Whether this bank's units lie on the +normal side of its skin."""
    sides = set()
    for keys, _, unit_is_positive in skin:
        a, b, c = (positions[k] for k in keys)
        sides.add(bool((np.cross(b-a, c-a) @ normal > 0) == unit_is_positive))
    if len(sides) != 1:
        raise ValueError('unsupported_shared_fault: fault skin is not a graph over the fault plane')
    return sides.pop()


def _key_edge(a, b):
    return (a, b) if repr(a) < repr(b) else (b, a)


def _constraints(skin, attached):
    """Skin edges other patches attach to, on the skin boundary, or between different units."""
    owners = defaultdict(list)
    for keys, unit, _ in skin:
        for i in range(3):
            owners[_key_edge(keys[i], keys[(i+1) % 3])].append(unit)
    return {edge for edge, units in owners.items()
            if len(units) == 1 or len(set(units)) > 1 or edge in attached}


def _locate_units(plane, triangles, skin, positions):
    """Unit of ``skin`` (one bank's original skin) containing each shared triangle."""
    from scipy.spatial import cKDTree

    uv = plane.project(np.array([[positions[k] for k in keys] for keys, _, _ in skin]).reshape(-1, 3))
    uv = uv.reshape(-1, 3, 2)
    units = [unit for _, unit, _ in skin]
    centroids = np.array([np.mean([plane.uv[v] for v in tri], axis=0) for tri in triangles])
    candidates = cKDTree(uv.mean(axis=1)).query(centroids, k=min(16, len(uv)))[1].reshape(len(centroids), -1)
    weights = _barycentric(uv[candidates], centroids)
    best = np.argmax(weights.min(axis=-1), axis=1)
    rows = np.arange(len(centroids))
    chosen = candidates[rows, best]
    missed = np.flatnonzero(weights[rows, best].min(axis=-1) < -1e-6)
    if len(missed):  # a long, thin skin triangle: search all of them
        weights = _barycentric(np.broadcast_to(uv, (len(missed),) + uv.shape), centroids[missed])
        best = np.argmax(weights.min(axis=-1), axis=1)
        if np.any(weights[np.arange(len(missed)), best].min(axis=-1) < -1e-6):
            raise ValueError('unsupported_shared_fault: shared triangle outside a bank fault skin')
        chosen[missed] = best
    return [units[i] for i in chosen.tolist()]


def _barycentric(corners, points):
    """Barycentric weights (M, K, 3) of ``points`` (M, 2) in triangles ``corners`` (M, K, 3, 2)."""
    a, b, c = corners[..., 0, :], corners[..., 1, :], corners[..., 2, :]
    p = points[:, None, :]
    area = _cross(b-a, c-a)
    return np.stack((_cross(b-p, c-p), _cross(c-p, a-p), _cross(a-p, b-p)), axis=-1)/area[..., None]


def _cross(u, v):
    return u[..., 0]*v[..., 1]-u[..., 1]*v[..., 0]
