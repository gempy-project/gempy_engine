"""Feature-preserving short-edge collapse on a closed-volume complex.

Dual contouring and rim clipping leave needle triangles: a vertex very close
to another (a contour crossing next to a rim node, two QEF vertices of
neighbouring leaves). Needles make poor tetrahedra. Collapsing the short edge
merges the two vertices in the whole shared complex at once, so every unit
shell stays closed and conforming.

Each triangle has a patch label: kind and side units, plus the box face for
extent triangles. A feature edge borders anything other than two triangles
with one label (seams, contours, fault traces, box edges). Vertices with no
feature edge may merge into a neighbour of their patch. Vertices with exactly
two feature edges may slide along that line onto a neighbour on it. All other
vertices (junctions, corners) never move. A collapse must keep the link
condition (no new duplicate edges) and must not fold or degenerate any
triangle. Geometry moves by at most the collapsed edge length, so unit volumes
change slightly; the report records it.
"""

from collections import defaultdict

import numpy as np

from .adaptive_closed_volumes import OUTSIDE, audit_shells, count_hanging_nodes


def collapse_short_edges(volumes, ratio=.2, max_passes=10):
    """A new ``ClosedUnitVolumes`` with needle edges collapsed.

    An edge is collapsed when it is shorter than ``ratio`` times the longest
    edge of a triangle it bounds.
    """
    from ...core.data.closed_unit_volumes import ClosedUnitVolumes

    xyz = np.asarray(volumes.vertices, dtype=float)
    triangles = {t: tuple(tri) for t, tri in enumerate(np.asarray(volumes.triangles, dtype=np.int64).tolist())}
    labels = {t: _label(volumes.patch_kinds[t], volumes.unit_sides[t], xyz[list(tri)])
              for t, tri in triangles.items()}
    sides = dict(enumerate(volumes.unit_sides))
    kinds = dict(enumerate(volumes.patch_kinds))
    vertex_tris = defaultdict(set)
    for t, tri in triangles.items():
        for v in tri:
            vertex_tris[v].add(t)
    collapsed = 0
    for _ in range(max_passes):
        done = 0
        for u, v in _candidates(triangles, xyz, ratio):
            if u not in vertex_tris or v not in vertex_tris:
                continue
            for moving, target in ((u, v), (v, u)):
                if _try_collapse(moving, target, triangles, labels, vertex_tris, xyz):
                    done += 1
                    break
        collapsed += done
        if not done:
            break

    order = sorted(triangles)
    used = sorted({v for t in order for v in triangles[t]})
    remap = {v: i for i, v in enumerate(used)}
    new_triangles = np.array([[remap[v] for v in triangles[t]] for t in order], dtype=np.int64).reshape(-1, 3)
    new_vertices = xyz[used]
    keys = [volumes.vertex_keys[v] for v in used]
    unit_sides = [sides[t] for t in order]
    shells = defaultdict(list)
    for tri, (negative, positive) in zip(new_triangles.tolist(), unit_sides):
        if negative != OUTSIDE:
            shells[negative].append(tuple(tri))
        if positive != OUTSIDE:
            shells[positive].append((tri[0], tri[2], tri[1]))
    audit = audit_shells(shells, dict(enumerate(new_vertices)))
    before = volumes.volumes
    report = dict(volumes.report, volumes=audit['volumes'], all_closed=audit['all_closed'],
                  all_positive=audit['all_positive'], open_edges=audit['open_edges'],
                  hanging_nodes=count_hanging_nodes(new_vertices, new_triangles),
                  collapse=dict(ratio=ratio, collapsed_edges=collapsed,
                                volume_change={u: audit['volumes'].get(u, 0.)-v for u, v in before.items()}))
    if not (report['all_closed'] and report['all_positive'] and not report['hanging_nodes']):
        raise ValueError('closed_volume_collapse_failed: collapsed complex is not closed and conforming')
    return ClosedUnitVolumes(vertices=new_vertices, triangles=new_triangles, unit_sides=unit_sides,
                             patch_kinds=[kinds[t] for t in order], vertex_keys=keys, report=report)


def _label(kind, sides, corners):
    if kind != 'extent':
        return kind, sides
    normal = np.cross(corners[1]-corners[0], corners[2]-corners[0])
    axis = int(np.argmax(np.abs(normal)))
    return kind, sides, axis, bool(normal[axis] > 0)


def _candidates(triangles, xyz, ratio):
    """Short edges of needle triangles, shortest first."""
    edges = {}
    for tri in triangles.values():
        p = xyz[list(tri)]
        lengths = [np.linalg.norm(p[(i+1) % 3]-p[i]) for i in range(3)]
        shortest = int(np.argmin(lengths))
        if lengths[shortest] < ratio*max(lengths):
            a, b = tri[shortest], tri[(shortest+1) % 3]
            edges[(min(a, b), max(a, b))] = lengths[shortest]
    return sorted(edges, key=lambda e: (edges[e], e))


def _feature_edges(vertex, triangles, labels, vertex_tris):
    """Edges at ``vertex`` that are not interior to one patch."""
    around = defaultdict(list)
    for t in vertex_tris[vertex]:
        for w in triangles[t]:
            if w != vertex:
                around[w].append(labels[t])
    return {w for w, owners in around.items() if len(owners) != 2 or owners[0] != owners[1]}


def _try_collapse(moving, target, triangles, labels, vertex_tris, xyz):
    features = _feature_edges(moving, triangles, labels, vertex_tris)
    if len(features) == 1 or len(features) > 2:
        return False  # junction or corner: fixed
    if len(features) == 2 and target not in features:
        return False  # a line vertex only slides along its line
    shared = vertex_tris[moving] & vertex_tris[target]
    if not shared:
        return False
    # Link condition: common neighbours are exactly the apexes of the collapsed triangles.
    neighbours = lambda x: {w for t in vertex_tris[x] for w in triangles[t]} - {x}
    apexes = {w for t in shared for w in triangles[t]} - {moving, target}
    if neighbours(moving) & neighbours(target) != apexes:
        return False
    # No folded or degenerate triangle; no duplicate triangle.
    existing = {frozenset(triangles[t]) for t in vertex_tris[target]}
    for t in vertex_tris[moving] - shared:
        tri = triangles[t]
        new = tuple(target if w == moving else w for w in tri)
        if frozenset(new) in existing:
            return False
        p_old, p_new = xyz[list(tri)], xyz[list(new)]
        n_old = np.cross(p_old[1]-p_old[0], p_old[2]-p_old[0])
        n_new = np.cross(p_new[1]-p_new[0], p_new[2]-p_new[0])
        longest = max(np.linalg.norm(p_new[(i+1) % 3]-p_new[i]) for i in range(3))
        if n_new @ n_old <= .5*np.linalg.norm(n_old)*np.linalg.norm(n_new) or \
                np.linalg.norm(n_new) <= 1e-3*longest**2:
            return False
    for t in shared:
        for w in triangles.pop(t):
            vertex_tris[w].discard(t)
        labels.pop(t)
    for t in list(vertex_tris[moving]):
        triangles[t] = tuple(target if w == moving else w for w in triangles[t])
        vertex_tris[target].add(t)
    del vertex_tris[moving]
    return True
