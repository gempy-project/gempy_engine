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

DELAUNAY_MARGIN = 1e-6  # radians


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
    plane = _PlaneTriangulation(normal, 1e-9*scale)
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


def _edge(a, b):
    return (a, b) if a < b else (b, a)


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


def _vertices_on_segment(xyz, p, q, tolerance):
    """Vertex ids strictly inside segment pq (within tolerance), in order from p."""
    d = q-p
    length2 = d @ d
    t = (xyz-p) @ d/length2
    distance = np.linalg.norm(np.cross(xyz-p, d), axis=1)/np.sqrt(length2)
    inside = np.flatnonzero((distance < tolerance) & (t*np.sqrt(length2) > tolerance) &
                            ((1-t)*np.sqrt(length2) > tolerance))
    return inside[np.argsort(t[inside])].tolist()


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


class _PlaneTriangulation:
    """A small constrained triangulation in the coordinates of the fault plane."""

    def __init__(self, normal, tolerance):
        helper = np.eye(3)[np.argmin(np.abs(normal))]
        first = np.cross(normal, helper)
        first /= np.linalg.norm(first)
        self.basis = np.stack((first, np.cross(normal, first)))
        self.tolerance = tolerance
        self.uv, self.xyz, self.keys = [], [], []
        self.tris = {}
        self.edge_tri = {}
        self.constrained = set()
        self.next_id = 0
        self.crossings = 0
        self.flips = 0

    def project(self, xyz):
        return np.asarray(xyz, dtype=float) @ self.basis.T

    # region Storage
    def add_vertex(self, xyz, key):
        self.uv.append(self.project(xyz))
        self.xyz.append(np.asarray(xyz, dtype=float))
        self.keys.append(key)
        return len(self.keys)-1

    def orient(self, a, b, c):
        return _cross(self.uv[b]-self.uv[a], self.uv[c]-self.uv[a])

    def add_triangle(self, a, b, c, orient=False):
        """Store triangle abc (counter-clockwise; ``orient`` reorders a skin triangle first)."""
        if orient and self.orient(a, b, c) < 0:
            b, c = c, b
        t = self.next_id
        self.next_id += 1
        self.tris[t] = (a, b, c)
        for i in range(3):
            edge = ((a, b, c)[i], (a, b, c)[(i+1) % 3])
            if edge in self.edge_tri:
                raise ValueError('unsupported_shared_fault: fault skin is not a manifold triangulation')
            self.edge_tri[edge] = t
        return t

    def remove_triangle(self, t):
        tri = self.tris.pop(t)
        for i in range(3):
            del self.edge_tri[(tri[i], tri[(i+1) % 3])]
        return tri

    def triangles(self):
        return [self.tris[t] for t in sorted(self.tris)]

    def opposite(self, a, b):
        """Third vertex of the triangle holding directed edge a->b."""
        t = self.edge_tri.get((a, b))
        if t is None:
            return None
        tri = self.tris[t]
        return tri[(tri.index(a)+2) % 3]
    # endregion

    # region Insertion
    def insert(self, xyz, key):
        """Vertex id for ``xyz``: an existing vertex, or a new one splitting a triangle or edge."""
        p = self.project(xyz)
        ids = sorted(self.tris)
        tri = np.array([self.tris[t] for t in ids])
        uv = np.array(self.uv)
        corners = uv[tri]
        distances = np.stack([_cross(corners[:, (i+1) % 3]-corners[:, i], p-corners[:, i]) /
                              np.linalg.norm(corners[:, (i+1) % 3]-corners[:, i], axis=1) for i in range(3)], axis=1)
        best = int(np.argmax(distances.min(axis=1)))
        if distances[best].min() < -self.tolerance:
            raise ValueError('unsupported_shared_fault: a bank vertex lies outside the other bank fault skin')
        near = np.linalg.norm(corners[best]-p, axis=1)
        if near.min() <= self.tolerance:
            return int(tri[best][np.argmin(near)])
        vertex = self.add_vertex(xyz, key)
        a, b, c = tri[best].tolist()
        on_edge = np.flatnonzero(distances[best] <= self.tolerance)
        if len(on_edge):
            i = int(on_edge[0])
            self.split_edge((a, b, c)[i], (a, b, c)[(i+1) % 3], vertex)
        else:
            self.remove_triangle(ids[best])
            for u, v in ((a, b), (b, c), (c, a)):
                self.add_triangle(u, v, vertex)
        return vertex

    def split_edge(self, a, b, vertex):
        for u, v in ((a, b), (b, a)):
            w = self.opposite(u, v)
            if w is None:
                continue
            self.remove_triangle(self.edge_tri[(u, v)])
            self.add_triangle(u, vertex, w)
            self.add_triangle(vertex, v, w)
        if _edge(a, b) in self.constrained:
            self.constrained.discard(_edge(a, b))
            self.constrained.update((_edge(a, vertex), _edge(vertex, b)))
    # endregion

    # region Constraints
    def recover(self, p, q):
        """Make segment pq a chain of constrained edges."""
        if p == q:
            return
        if (p, q) in self.edge_tri or (q, p) in self.edge_tri:
            self.constrained.add(_edge(p, q))
            return
        # Split by constrained edges it crosses, then by vertices on it.
        constrained = np.array(sorted(self.constrained), dtype=np.int64).reshape(-1, 2)
        for (a, b), s in zip(*self._crossing(constrained, p, q)):
            self.crossings += 1
            self.insert(self.xyz[a]+s*(self.xyz[b]-self.xyz[a]), ('fault_crossing', self.crossings))
        uv = np.array(self.uv)
        chain = [p] + [v for v in _vertices_on_segment(np.c_[uv, np.zeros(len(uv))],
                                                         np.r_[self.uv[p], 0.], np.r_[self.uv[q], 0.],
                                                         self.tolerance) if v not in (p, q)] + [q]
        for u, v in zip(chain, chain[1:]):
            self._recover_edge(u, v)

    def _crossing(self, edges, p, q):
        """Edges (E, 2) that strictly cross segment pq, and the crossing parameter along each."""
        uv = np.array(self.uv)
        a, b = uv[edges[:, 0]], uv[edges[:, 1]]
        P, Q = uv[p], uv[q]
        d1, d2 = _cross(Q-P, a-P), _cross(Q-P, b-P)
        d3, d4 = _cross(b-a, P-a), _cross(b-a, Q-a)
        tol_pq = self.tolerance*np.linalg.norm(Q-P)
        tol_ab = self.tolerance*np.linalg.norm(b-a, axis=1)
        shared = (edges == p).any(axis=1) | (edges == q).any(axis=1)
        strict = (((d1 > tol_pq) & (d2 < -tol_pq)) | ((d1 < -tol_pq) & (d2 > tol_pq))) & \
                 (((d3 > tol_ab) & (d4 < -tol_ab)) | ((d3 < -tol_ab) & (d4 > tol_ab))) & ~shared
        rows = np.flatnonzero(strict)
        return [tuple(e) for e in edges[rows].tolist()], (d1[rows]/(d1[rows]-d2[rows])).tolist()

    def _recover_edge(self, u, v):
        if (u, v) in self.edge_tri or (v, u) in self.edge_tri:
            self.constrained.add(_edge(u, v))
            return
        edges = np.array(sorted({_edge(*e) for e in self.edge_tri}), dtype=np.int64)
        crossing = self._crossing(edges, u, v)[0]
        if any(edge in self.constrained for edge in crossing):
            raise ValueError('unsupported_shared_fault: bank constraints cross without a shared vertex')
        budget = 50*len(crossing)+100
        while crossing:
            budget -= 1
            if budget < 0:
                raise ValueError('unsupported_shared_fault: constraint recovery did not converge')
            a, b = crossing.pop(0)
            c, d = self.opposite(a, b), self.opposite(b, a)
            if c is None or d is None or not self._convex(a, b, c, d):
                crossing.append((a, b))
                continue
            self._flip(a, b)
            if self._crossing(np.array([_edge(c, d)]), u, v)[0]:
                crossing.append(_edge(c, d))
        if (u, v) not in self.edge_tri and (v, u) not in self.edge_tri:
            raise ValueError('unsupported_shared_fault: constraint edge was not recovered')
        self.constrained.add(_edge(u, v))

    def _convex(self, a, b, c, d):
        """Quad a, d, b, c (diagonal ab, apexes c and d) strictly convex."""
        return self.orient(c, a, d) > 0 and self.orient(d, b, c) > 0 and \
            min(self.orient(c, a, d), self.orient(d, b, c)) > 1e-12*np.linalg.norm(self.uv[a]-self.uv[b])**2

    def _flip(self, a, b):
        c, d = self.opposite(a, b), self.opposite(b, a)
        self.remove_triangle(self.edge_tri[(a, b)])
        self.remove_triangle(self.edge_tri[(b, a)])
        self.add_triangle(c, a, d)
        self.add_triangle(c, d, b)
        self.flips += 1
    # endregion

    def make_delaunay(self):
        """Lawson flips of unconstrained edges to the constrained Delaunay triangulation."""
        queue = sorted({_edge(*e) for e in self.edge_tri})
        queued = set(queue)
        budget = 100*len(queue)+1000
        while queue:
            budget -= 1
            if budget < 0:
                raise ValueError('unsupported_shared_fault: Delaunay flips did not converge')
            a, b = queue.pop()
            queued.discard((a, b))
            if (a, b) in self.constrained:
                continue
            c, d = self.opposite(a, b), self.opposite(b, a)
            if c is None or d is None or not self._in_circle(a, b, c, d) or not self._convex(a, b, c, d):
                continue
            self._flip(a, b)
            for edge in (_edge(a, c), _edge(c, b), _edge(b, d), _edge(d, a)):
                if edge not in queued:
                    queue.append(edge)
                    queued.add(edge)

    def _in_circle(self, a, b, c, d):
        """Edge ab is not locally Delaunay: the opposite angles sum past pi (+ round-off margin)."""
        def angle(apex, p, q):
            u, v = self.uv[p]-self.uv[apex], self.uv[q]-self.uv[apex]
            return np.arctan2(abs(_cross(u, v)), u @ v)
        return angle(c, a, b)+angle(d, b, a) > np.pi+DELAUNAY_MARGIN
