"""Pure canonical-tile evidence and symbolic adaptive dual incidence.

Vertices are integer ids: ``s*N + leaf`` is surface ``s``'s own (``regular``)
QEF vertex in a leaf, ``S*N + j`` the ``j``-th shared contact (``joint``)
vertex. Triangles are planned and validated on ids before any face is
allocated; the caller maps ids to the public symbolic keys.
"""

import numpy as np

from .joint_xp import host, namespace


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


def plan_adaptive_junctions(face_cells, tile_xyz, tile_fields, pairs, identities,
                            origins, spans, counts, edge_xyz, edge_normals, flags,
                            *, controllers=None, leaf_fields=None, overridden=None):
    """One pair per leaf, with exactly two canonical shared boundary roots.

    ``face_cells`` (F, 2) are each tile's leaves (-1 outside). ``overridden``
    maps surfaces to (N,) masks of leaves whose vertex is replaced by a fault
    vertex. A junction needing such a leaf, or one that cannot be planned
    unambiguously, is given up: participants keep unshared vertices there.
    Returns ``(junctions, evidence, fallback)``; evidence of fallback leaves is dropped.
    """
    evidence, leaf_roots, fallback = [], {}, set()
    overridden = overridden or {}
    for pair in sorted(pairs):
        controller, target = pair
        additional = {(c, sign) for participant in pair
                      for c, sign in (controllers or {}).get(participant, []) if c not in pair}
        decisions = None
        if additional:
            signed = np.array([sign*leaf_fields[c] for c, sign in additional])
            excluded = np.any(np.all(signed < 0, axis=2), axis=0)
            retained = np.all(signed > 0, axis=(0, 2))
            decisions = np.where(excluded, 'excluded', np.where(retained, 'retained', 'competing'))
        # face_pair_root returns None unless both participants change sign on
        # the tile; nothing earlier in the loop can raise, so skip those tiles.
        pair_fields = np.asarray(tile_fields[[controller, target]], dtype=float)
        straddles = ~np.any((pair_fields.min(axis=2) > 0) | (pair_fields.max(axis=2) < 0), axis=0)
        masks = [overridden[p] for p in pair if p in overridden]
        skipped = np.logical_or.reduce(masks) if masks else None
        for tile in np.flatnonzero(straddles).tolist():
            cells = tuple(c for c in face_cells[tile].tolist() if c >= 0)
            if skipped is not None and all(skipped[c] for c in cells):
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
        if cell in fallback or any(overridden[s][cell] for pair in candidates for s in pair if s in overridden):
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


# Production quad split of a collapsed ring: (0, 1, 3), (2, 3, 1); a triangle ring is kept.
_QUAD = np.array([[0, 1, 3], [2, 3, 1]])
_TRIANGLE = np.array([0, 1, 2])


def collapse_rings(rings):
    """Collapse only cyclic consecutive repetitions (coarse leaves) of (R, 4) vertex rings.

    Returns the rings compacted to the front and padded with -1, and their
    lengths. A ring that is not three or four distinct cyclic vertices raises.
    """
    xp = namespace(rings)
    rings = xp.asarray(rings, dtype=np.int64).reshape(-1, 4)
    keep = xp.ones(tuple(rings.shape), dtype=bool)
    keep[:, 1:] = rings[:, 1:] != rings[:, :-1]
    # The last run wraps onto the first one.
    last = xp.max(xp.where(keep, xp.arange(4)[None, :], -1), axis=1)
    rows = xp.arange(len(rings))
    keep[rows, last] &= ~((rings[:, 3] == rings[:, 0]) & (keep.sum(axis=1) > 1))
    lengths = keep.sum(axis=1)
    order = _front(xp, keep)
    collapsed = xp.where(xp.arange(4)[None, :] < lengths[:, None], xp.take_along_axis(rings, order, axis=1), -1)
    distinct = xp.ones(len(rings), dtype=bool)
    for a in range(4):
        for b in range(a+1, 4):
            distinct &= (collapsed[:, a] != collapsed[:, b]) | (collapsed[:, b] < 0)
    if bool((((lengths != 3) & (lengths != 4)) | ~distinct).any()):
        raise ValueError('unsupported_adaptive_ring: need three or four distinct cyclic leaves')
    return collapsed, lengths


def _front(xp, keep):
    """Column order per row moving kept columns to the front, keeping their order."""
    position = xp.cumsum(xp.astype(keep, np.int64), axis=1) - 1
    dropped = xp.cumsum(xp.astype(~keep, np.int64), axis=1) - 1 + keep.sum(axis=1)[:, None]
    target = xp.where(keep, position, dropped)
    order = xp.zeros(tuple(keep.shape), dtype=np.int64)
    rows = xp.repeat(xp.arange(len(keep))[:, None], 4, axis=1)
    order[rows, target] = xp.repeat(xp.arange(4)[None, :], len(keep), axis=0)
    return order


def ring_triangles(collapsed, lengths):
    """Production triangles of collapsed rings: ``(row, (T, 3) vertices, first)``, in ring order."""
    xp = namespace(collapsed, lengths)
    count = xp.where(lengths == 4, 2, 1)
    row = xp.repeat(xp.arange(len(lengths)), count)
    slot = xp.arange(len(row))-xp.repeat(xp.cumsum(count)-count, count)
    quad, triangle = xp.asarray(_QUAD, dtype=np.int64), xp.asarray(_TRIANGLE, dtype=np.int64)
    corners = xp.where((lengths[row] == 4)[:, None], quad[xp.minimum(slot, 1)], triangle[None, :])
    return row, xp.take_along_axis(collapsed[row], corners, axis=1), slot == 0


def plan_edge_rows(selected, edge_cells, vertex_ids, borrowed, fault_vertex):
    """Symbolic triangles of every selected (surface, minimal edge) dual face.

    ``vertex_ids`` (S, N) are each surface's used vertex per leaf; its own are
    ``s*N + leaf``. Where a surface borrows fault vertices (``borrowed`` (S, N)
    mask), rings are split into production diagonals, dropping triangles lying
    wholly on vertices of ``fault_vertex`` (per vertex id). Returns the plan
    ``dict(surface, edge, keys (T, 3), regular (T, 4), entry)`` in surface, edge
    and ring order, with ``regular`` the collapsed own-vertex ring and ``entry``
    marking each emitted face's first triangle, and the dropped triangle count.
    """
    xp = namespace(selected, edge_cells, vertex_ids)
    n_leaves = vertex_ids.shape[1]
    surfaces, edges = xp.nonzero(selected)
    cells = edge_cells[edges]
    regular, _ = collapse_rings(surfaces[:, None]*n_leaves + cells)
    keys, lengths = collapse_rings(vertex_ids[surfaces[:, None], cells])
    row, triangles, first = ring_triangles(keys, lengths)
    split = borrowed[surfaces[:, None], cells].any(axis=1)[row]
    dropped = split & fault_vertex[triangles].all(axis=1)
    kept = ~dropped
    plan = dict(surface=surfaces[row][kept], edge=edges[row][kept], keys=triangles[kept],
                regular=regular[row][kept], entry=(first | split)[kept])
    return plan, int(dropped.sum())


def validate_adaptive_incidence(plan, junction_ids, junction_pairs, joint_start, evidence, spans):
    """Seams between junction vertices supported by two controller sides and one target side.

    ``junction_ids`` (N,) give each junction leaf's vertex id (-1 elsewhere);
    junction vertex ``joint_start + j`` joins the sorted surface pair
    ``junction_pairs[j]``. Symbolic triangle sides are counted including
    production triangulation diagonals. Returns ``(seams, coarsefine)``: sets of
    ``(min id, max id)``; junction faces without that support are not seams.
    """
    counts = {}
    if len(junction_pairs) and len(evidence):
        xp = namespace(plan['keys'])
        keys = plan['keys']
        sides = xp.stack((keys, xp.roll(keys, -1, axis=1)), axis=-1).reshape(-1, 2)
        joint = (sides >= joint_start).all(axis=1)
        # Junction sides are few: count them on the host.
        sides, surface = host(sides[joint]), host(xp.repeat(plan['surface'], 3)[joint])
        pairs = np.asarray(host(junction_pairs))[sides-joint_start]
        same = np.all(pairs[:, 0] == pairs[:, 1], axis=1)
        rows, found = np.unique(np.column_stack((surface[same], np.sort(sides[same], axis=1))), axis=0,
                                return_counts=True)
        counts = dict(zip(map(tuple, rows.tolist()), found.tolist()))
    junction_ids, spans = host(junction_ids), host(spans)
    expected, coarsefine = set(), set()
    for item in evidence:
        # Only interior tiles between two planned junctions form seams.
        if len(item['cells']) != 2 or any(junction_ids[c] < 0 for c in item['cells']):
            continue
        a, b = item['cells']
        seam = tuple(sorted((int(junction_ids[a]), int(junction_ids[b]))))
        controller, target = item['pair']
        if counts.get((controller,)+seam, 0) != 2 or counts.get((target,)+seam, 0) != 1:
            continue
        expected.add(seam)
        if spans[a] != spans[b]:
            coarsefine.add(seam)
    return expected, coarsefine


def plan_adaptive_triangles(plan, positions, normals, joint_start, n_surfaces):
    """Production winding on symbolic triangles; zero-area triangles are dropped.

    ``positions`` (V, 3) per vertex id (NaN where none exists), ``normals``
    (T, 3) the Hermite normal of each triangle's dual face. Returns
    ``dict(surface, keys, affected, n_surfaces)`` for ``emit_adaptive_triangles``.
    """
    xp = namespace(plan['keys'], positions)
    keys = plan['keys']
    corners = positions[keys]
    if bool(xp.isnan(corners).any()):
        raise ValueError('missing_adaptive_vertex: triangle references a vertex without a position')
    cross = xp.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    kept = xp.norm(cross, axis=1) >= 1e-12
    flip = xp.dot_rows(cross, normals) < 0
    oriented = xp.where(flip[:, None], keys[:, [0, 2, 1]], keys)[kept]
    return dict(surface=plan['surface'][kept], keys=oriented,
                affected=(oriented >= joint_start).any(axis=1), n_surfaces=n_surfaces)


def emit_adaptive_triangles(triangles, key_to_id):
    """Allocate faces only from a validated, already oriented symbolic plan.

    ``key_to_id`` maps vertex ids to output vertex rows. Returns per-surface
    host faces (T, 3) and their ``affected`` (joint-vertex) masks.
    """
    surface, keys, affected = (host(triangles[k]) for k in ('surface', 'keys', 'affected'))
    faces = np.asarray(host(key_to_id), dtype=np.int64)[keys].reshape(-1, 3)
    order = np.argsort(surface, kind='stable')
    bounds = np.searchsorted(surface[order], np.arange(triangles['n_surfaces']+1))
    return ([faces[order[a:b]] for a, b in zip(bounds[:-1], bounds[1:])],
            [np.asarray(affected[order[a:b]], dtype=bool) for a, b in zip(bounds[:-1], bounds[1:])])
