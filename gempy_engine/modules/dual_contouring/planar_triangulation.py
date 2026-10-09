"""Small constrained planar triangulation: insertion, constraint recovery, Delaunay flips.

Used for the shared fault interface and the extent caps of closed volumes.
Vertices keep their own 3D positions; 2D coordinates in the plane only drive
the combinatorics.
"""

import numpy as np

DELAUNAY_MARGIN = 1e-6  # radians


def _edge(a, b):
    return (a, b) if a < b else (b, a)


def _vertices_on_segment(xyz, p, q, tolerance):
    """Vertex ids strictly inside segment pq (within tolerance), in order from p."""
    d = q-p
    length2 = d @ d
    t = (xyz-p) @ d/length2
    distance = np.linalg.norm(np.cross(xyz-p, d), axis=1)/np.sqrt(length2)
    inside = np.flatnonzero((distance < tolerance) & (t*np.sqrt(length2) > tolerance) &
                            ((1-t)*np.sqrt(length2) > tolerance))
    return inside[np.argsort(t[inside])].tolist()


def _cross(u, v):
    return u[..., 0]*v[..., 1]-u[..., 1]*v[..., 0]


class PlaneTriangulation:
    """A small constrained triangulation in 2D coordinates of a plane.

    Triangles may carry a label (a unit); splits keep it and flips only join
    triangles with equal labels.
    """

    def __init__(self, normal, tolerance):
        helper = np.eye(3)[np.argmin(np.abs(normal))]
        first = np.cross(normal, helper)
        first /= np.linalg.norm(first)
        self.basis = np.stack((first, np.cross(normal, first)))
        self.tolerance = tolerance
        self.uv, self.xyz, self.keys = [], [], []
        self.tris = {}
        self.labels = {}
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

    def add_triangle(self, a, b, c, orient=False, label=None):
        """Store triangle abc (counter-clockwise; ``orient`` reorders an input triangle first)."""
        if orient and self.orient(a, b, c) < 0:
            b, c = c, b
        t = self.next_id
        self.next_id += 1
        self.tris[t] = (a, b, c)
        self.labels[t] = label
        for i in range(3):
            edge = ((a, b, c)[i], (a, b, c)[(i+1) % 3])
            if edge in self.edge_tri:
                raise ValueError('non_manifold_triangulation: an edge is used twice in one direction')
            self.edge_tri[edge] = t
        return t

    def remove_triangle(self, t):
        tri = self.tris.pop(t)
        for i in range(3):
            del self.edge_tri[(tri[i], tri[(i+1) % 3])]
        return tri, self.labels.pop(t)

    def labelled_triangles(self):
        return [(self.tris[t], self.labels[t]) for t in sorted(self.tris)]

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
            _, label = self.remove_triangle(ids[best])
            for u, v in ((a, b), (b, c), (c, a)):
                self.add_triangle(u, v, vertex, label=label)
        return vertex

    def split_edge(self, a, b, vertex):
        for u, v in ((a, b), (b, a)):
            w = self.opposite(u, v)
            if w is None:
                continue
            _, label = self.remove_triangle(self.edge_tri[(u, v)])
            self.add_triangle(u, vertex, w, label=label)
            self.add_triangle(vertex, v, w, label=label)
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
        _, label = self.remove_triangle(self.edge_tri[(a, b)])
        _, other = self.remove_triangle(self.edge_tri[(b, a)])
        if label != other:
            raise ValueError('constrained_flip: flipping an edge between differently labelled triangles')
        self.add_triangle(c, a, d, label=label)
        self.add_triangle(c, d, b, label=label)
        self.flips += 1
    # endregion

    def make_delaunay(self):
        """Lawson flips of unconstrained edges to the constrained Delaunay triangulation.

        Edges between differently labelled triangles are kept as constraints.
        """
        queue = sorted({_edge(*e) for e in self.edge_tri})
        queued = set(queue)
        budget = 100*len(queue)+1000
        while queue:
            budget -= 1
            if budget < 0:
                raise ValueError('delaunay_flips_not_converged')
            a, b = queue.pop()
            queued.discard((a, b))
            if (a, b) in self.constrained:
                continue
            c, d = self.opposite(a, b), self.opposite(b, a)
            if c is not None and d is not None and \
                    self.labels[self.edge_tri[(a, b)]] != self.labels[self.edge_tri[(b, a)]]:
                continue
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
