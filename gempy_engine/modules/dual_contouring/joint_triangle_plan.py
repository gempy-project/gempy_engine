"""Pure canonical-tile evidence and symbolic adaptive dual incidence.

Vertices are symbolic keys ``(kind, identities, leaf origin, leaf span, 0)``:
``regular`` keys are one surface's own QEF vertex, ``joint`` keys a shared
contact vertex. Triangles are planned and validated on keys before any face
is allocated.
"""

from collections import Counter

import numpy as np


def face_pair_root(values):
    """Isolated bilinear pair root on one face, ordered 00, 10, 01, 11."""
    face = np.asarray(values, dtype=float)
    if np.any((face.min(axis=1) > 0) | (face.max(axis=1) < 0)):
        return None
    scales = np.max(np.abs(face), axis=1)
    if np.any(scales < 1e-14):
        raise ValueError('degenerate_sampled_face: interface occupies entire tile')
    face = face / scales[:, None]
    swapped = False
    for attempt in range(2):
        a = face[:, 0]
        b, c = face[:, 1] - a, face[:, 2] - a
        d = face[:, 3] - a - b - c
        polynomial = np.array([b[0]*d[1]-b[1]*d[0],
                               a[0]*d[1]+b[0]*c[1]-a[1]*d[0]-b[1]*c[0],
                               a[0]*c[1]-a[1]*c[0]])
        nonzero = np.flatnonzero(np.abs(polynomial) > 1e-12)
        if len(nonzero):
            break
        if attempt:
            raise ValueError('degenerate_sampled_face: nonisolated bilinear intersection')
        # A pair depending only on u has a zero v-resultant even when disjoint.
        face = face[:, [0, 2, 1, 3]]
        swapped = True
    roots = []
    for root in np.roots(polynomial[nonzero[0]:]):
        if abs(root.imag) > 1e-10 or not -1e-10 <= root.real <= 1+1e-10:
            continue
        u = float(root.real)
        denominator = c + d*u
        row = np.argmax(np.abs(denominator))
        if abs(denominator[row]) < 1e-12:
            if np.max(np.abs(a+b*u)) < 1e-10:
                raise ValueError('degenerate_sampled_face: nonisolated bilinear intersection')
            continue
        v = -(a[row]+b[row]*u)/denominator[row]
        if not -1e-10 <= v <= 1+1e-10:
            continue
        if np.max(np.abs(a+b*u+c*v+d*u*v)) > 1e-9:
            raise ValueError('unresolved_sampled_face: resultant residual')
        if abs(np.linalg.det(np.column_stack((b+d*v, c+d*u)))) < 1e-10:
            raise ValueError('degenerate_sampled_face: tangential intersection')
        if min(u, v, 1-u, 1-v) < 1e-10:
            raise ValueError('grid_edge_junction: tile-edge root needs sided support')
        roots.append(np.array([v, u]) if swapped else np.array([u, v]))
    if len(roots) > 1:
        raise ValueError('unsupported_sampled_junction: multiple tile roots')
    return roots[0] if roots else None


def plan_adaptive_junctions(faces, tile_xyz, tile_fields, pairs, identities,
                            origins, spans, counts, edge_xyz, edge_normals, flags,
                            *, controllers=None, leaf_fields=None, overridden=None):
    """One pair per leaf, with exactly two canonical shared boundary roots.

    ``overridden`` maps surfaces to leaves whose vertex is replaced by a fault
    vertex. A junction needing such a leaf, or one that cannot be planned
    unambiguously, is given up: participants keep unshared vertices there.
    Returns ``(junctions, evidence, fallback)``; evidence of fallback leaves is dropped.
    """
    evidence, leaf_roots, fallback = [], {}, set()
    for pair in sorted(pairs):
        controller, target = pair
        additional = {(c, sign) for participant in pair
                      for c, sign in (controllers or {}).get(participant, []) if c not in pair}
        decisions = {}
        if additional:
            signed = np.array([sign*leaf_fields[c] for c, sign in additional])
            excluded = np.any(np.all(signed < 0, axis=2), axis=0)
            retained = np.all(signed > 0, axis=(0, 2))
            decisions = dict(enumerate(np.where(excluded, 'excluded', np.where(
                retained, 'retained', 'competing')).tolist()))
        # face_pair_root returns None unless both participants change sign on
        # the tile; nothing earlier in the loop can raise, so skip those tiles.
        pair_fields = np.asarray(tile_fields[[controller, target]], dtype=float)
        straddles = ~np.any((pair_fields.min(axis=2) > 0) | (pair_fields.max(axis=2) < 0), axis=0)
        skipped = set().union(*(overridden.get(p, ()) for p in pair)) if overridden else set()
        for tile in np.flatnonzero(straddles).tolist():
            face = faces[tile]
            cells = tuple(c for c in face['cells'] if c is not None)
            if skipped and all(c in skipped for c in cells):
                continue
            if additional:
                signed = np.array([sign*tile_fields[c, tile] for c, sign in additional])
                tile_decision = ('excluded' if np.any(np.all(signed < 0, axis=1)) else
                                 'retained' if np.all(signed > 0) else 'competing')
                if tile_decision == 'excluded' and all(decisions[c] == 'excluded' for c in cells):
                    continue
            root = face_pair_root(tile_fields[[controller, target], tile])
            if root is None:
                continue
            if additional:
                if any(decisions[c] != tile_decision for c in cells) or tile_decision == 'competing':
                    fallback.update(cells)
                    continue
                if tile_decision == 'excluded':
                    continue
            xyz = tile_xyz[tile]
            point = xyz[0] + root[0]*(xyz[1]-xyz[0]) + root[1]*(xyz[2]-xyz[0])
            evidence.append(dict(tile=tile, pair=pair, cells=cells, point=point))
            for cell in cells:
                leaf_roots.setdefault(cell, {}).setdefault(pair, []).append(point)
    junctions = {}
    for cell, candidates in sorted(leaf_roots.items()):
        if cell in fallback or (overridden and any(cell in overridden.get(s, ()) for pair in candidates for s in pair)):
            fallback.add(cell)
            continue
        pair, roots = next(iter(candidates.items()))
        # Competing pairs, missing roots, extra branches or one stack: no shared junction.
        if (len(candidates) != 1 or len(roots) != 2 or any(counts[s, cell] != 1 for s in pair)
                or identities[pair[0]][0] == identities[pair[1]][0]):
            fallback.add(cell)
            continue
        points = np.concatenate([edge_xyz[s, cell, flags[s, cell]] for s in pair])
        normals = np.concatenate([edge_normals[s, cell, flags[s, cell]] for s in pair])
        point, end = roots
        direction = end-point
        projected = normals @ direction
        rhs = np.sum(normals*(points-point), axis=1)
        denominator = projected @ projected + direction @ direction
        if not np.isfinite(denominator) or denominator <= 1e-24:
            fallback.add(cell)
            continue
        parameter = np.clip((projected @ rhs + (points.mean(axis=0)-point) @ direction)/denominator, 0., 1.)
        key = ('joint', tuple(sorted(identities[s] for s in pair)),
               tuple(map(int, origins[cell])), int(spans[cell]), 0)
        junctions[cell] = dict(key=key, pair=pair, position=point+parameter*direction)
    evidence = [item for item in evidence if not any(c in fallback for c in item['cells'])]
    return junctions, evidence, fallback


def collapse_ring(keys):
    """Collapse only cyclic consecutive repetitions caused by coarse leaves."""
    ring = []
    for key in keys:
        if not ring or key != ring[-1]:
            ring.append(key)
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring.pop()
    if len(ring) not in (3, 4) or len(set(ring)) != len(ring):
        raise ValueError('unsupported_adaptive_ring: need three or four distinct cyclic leaves')
    return ring


def ring_triangles(keys):
    if len(keys) == 3:
        return [tuple(keys)]
    return [tuple(keys[i] for i in offsets) for offsets in ((0, 1, 3), (2, 3, 1))]


def plan_edge_rows(selected, edges, crossing_normals, regular_key, vertex_key, borrowed, fault_identities):
    """One dual-face row per selected (surface, minimal edge), on the edge's leaf ring.

    ``regular_key(s, leaf)`` / ``vertex_key(s, leaf)`` give a surface's own and
    its used vertex key. Where a surface borrows fault vertices
    (``borrowed`` {(surface, leaf): fault}), rows are split into production
    diagonals, dropping triangles lying wholly on vertices of
    ``fault_identities``. Returns per-surface rows and the dropped count.
    """
    plans = [[] for _ in range(len(selected))]
    filtered = 0
    for s in range(len(selected)):
        for e in np.flatnonzero(selected[s]).tolist():
            cells = edges[e]['cells']
            regular_keys = collapse_ring([regular_key(s, c) for c in cells])
            keys = collapse_ring([vertex_key(s, c) for c in cells])
            row = dict(keys=keys, regular_keys=regular_keys, normal=crossing_normals[s, e], edge=e)
            if any((s, c) in borrowed for c in cells):
                for triangle in ring_triangles(keys):
                    if all(k[1][0] in fault_identities for k in triangle):
                        filtered += 1
                    else:
                        plans[s].append(dict(row, keys=list(triangle)))
                continue
            plans[s].append(row)
    return plans, filtered


def validate_adaptive_incidence(plans, junctions, evidence, spans):
    """Seams between junction vertices supported by two controller sides and one target side.

    Symbolic triangle sides are counted including production triangulation
    diagonals; junction faces without that support are not reported as seams.
    """
    counts = [Counter() for _ in plans]
    for surface, rows in enumerate(plans):
        for row in rows:
            for triangle in ring_triangles(row['keys']):
                for a, b in zip(triangle, triangle[1:]+triangle[:1]):
                    if a[0] == b[0] == 'joint' and a[1] == b[1]:
                        counts[surface][tuple(sorted((a, b)))] += 1
    expected, coarsefine = set(), set()
    for item in evidence:
        # Only interior tiles between two planned junctions form seams.
        if len(item['cells']) != 2 or any(c not in junctions for c in item['cells']):
            continue
        a, b = item['cells']
        seam = tuple(sorted((junctions[a]['key'], junctions[b]['key'])))
        controller, target = item['pair']
        if counts[controller][seam] != 2 or counts[target][seam] != 1:
            continue
        expected.add(seam)
        if spans[a] != spans[b]:
            coarsefine.add(seam)
    return expected, coarsefine


def plan_adaptive_triangles(plans, positions):
    """Production winding on symbolic triangles; zero-area triangles are dropped."""
    flat = [(surface, keys, row['normal']) for surface, rows in enumerate(plans)
            for row in rows for keys in ring_triangles(row['keys'])]
    triangles = [[] for _ in plans]
    if not flat:
        return triangles
    corners = np.array([[positions[k] for k in keys] for _, keys, _ in flat], dtype=float)
    hermite = np.array([normal for _, _, normal in flat], dtype=float)
    normals = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    degenerate = np.linalg.norm(normals, axis=1) < 1e-12
    dots = np.einsum('ij,ij->i', normals, hermite)
    for (surface, keys, normal), flip, skip in zip(flat, (dots < 0).tolist(), degenerate.tolist()):
        if skip:
            continue
        triangles[surface].append(dict(keys=(keys[0], keys[2], keys[1]) if flip else keys, normal=normal))
    return triangles


def emit_adaptive_triangles(triangles, key_to_id):
    """Allocate faces only from a validated, already oriented symbolic plan."""
    faces, affected = [], []
    for rows in triangles:
        faces.append(np.asarray([[key_to_id[k] for k in row['keys']] for row in rows], dtype=np.int64).reshape(-1, 3))
        affected.append(np.asarray([any(k[0] == 'joint' for k in row['keys']) for row in rows], dtype=bool))
    return faces, affected
