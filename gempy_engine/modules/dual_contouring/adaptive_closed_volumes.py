"""Closed unit volumes for adaptive joint extraction.

The extent rim is the finest lattice of the six box faces, each finest square
split into two triangles and clipped exactly by every surface's piecewise-linear
interpolant. Rim vertices are lattice nodes, edge crossings and contour
intersections, all keyed canonically, so faces, triangles and partitions share
them without welding. Surfaces close against the rim through strips (the dual
faces of boundary minimal edges) and fans (a boundary leaf's vertex to its rim
contour segments). Every triangle separates two units; per-unit shells are
audited for closure (each edge used exactly twice, in opposite directions) and
for volumes summing to the partition region. Nothing is repaired or welded.
"""

from collections import defaultdict

import numpy as np

from ...core.data.stack_relation_type import StackRelationType as R

OUTSIDE = ('outside',)


# region Units

def classify_units(stack_values, relations, stack_isovalues):
    """Active stack and interval per point, mirroring production lithology masking.

    ``stack_values`` are raw stack scalars (S, M). Mirrors ``_lithology_mask``
    (erosion/onlap components, onlap chains, preceding-mask exclusion). The
    interval is the number of the active stack's surfaces above the value.
    """
    values = np.asarray(stack_values, dtype=float)
    n, m = values.shape

    def component(i, kind):
        isovalues = np.asarray(stack_isovalues[i], dtype=float)
        if kind is R.ERODE:
            return values[i] > isovalues.min()
        if kind is R.ONLAP:
            return values[i] > isovalues.max()
        if kind is R.FAULT:
            return np.zeros(m, dtype=bool)
        if kind is False or kind is R.BASEMENT:
            return np.ones(m, dtype=bool)
        raise ValueError(f'unsupported_closed_volume_relation: {kind}')

    mask = np.zeros((n, m), dtype=bool)
    chain = 0
    for i in range(n):
        continues = relations[i - 1] in (R.ONLAP, R.FAULT)
        began = relations[i - 1 - chain] is R.ONLAP
        chain = (chain + 1) * continues * began
        if chain:
            mask[i - 1] = component(i, R.ONLAP)
            block = mask[i - chain:i]
            mask[i - chain:i] = np.flip(np.cumprod(np.flip(block, axis=0), axis=0), axis=0).astype(bool)
        relation = relations[i]
        if relation is R.ONLAP:
            continue
        if relation is R.NULL_SPACE:
            raise ValueError('unsupported_closed_volume_relation: null space')
        mask[i] = component(i, R.BASEMENT if relation is False else relation)
    final = np.zeros_like(mask)
    final[0] = mask[-1]
    final[1:] = np.cumprod(~mask[:-1], axis=0).astype(bool)
    final &= mask
    if not final.any(axis=0).all():
        raise ValueError('unclassified_closed_volume_point: no active stack')
    active = np.argmax(final, axis=0)
    interval = np.array([int(np.sum(np.asarray(stack_isovalues[g]) > values[g, j]))
                         for j, g in enumerate(active.tolist())], dtype=np.int64)
    return active, interval

# endregion


# region Rim lattice

def _rim_lattice(domain):
    """Finest nodes of the box surface and two triangles per finest face square."""
    domain = [int(v) for v in domain]
    index, coords = {}, []

    def node(point):
        point = tuple(point)
        if point not in index:
            index[point] = len(coords)
            coords.append(point)
        return index[point]

    triangles, squares, normals = [], [], []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for side in (0, 1):
            outward = np.zeros(3)
            outward[axis] = 1. if side else -1.
            for i in range(domain[u]):
                for j in range(domain[v]):
                    corner = {}
                    for du in (0, 1):
                        for dv in (0, 1):
                            point = [0, 0, 0]
                            point[axis] = side*domain[axis]
                            point[u], point[v] = i+du, j+dv
                            corner[du, dv] = node(point)
                    for tri in ((corner[0, 0], corner[1, 0], corner[1, 1]),
                                (corner[0, 0], corner[1, 1], corner[0, 1])):
                        triangles.append(tri)
                        squares.append((axis, side, i, j))
                        normals.append(outward)
    return index, np.asarray(coords, dtype=np.int64), np.asarray(triangles, dtype=np.int64), squares, \
        np.asarray(normals)


def _boundary_leaf_map(origins, spans, domain):
    """Finest face square -> the boundary leaf whose face tile covers it."""
    leaf_of = {}
    for cell, (origin, span) in enumerate(zip(origins.tolist(), spans.tolist())):
        for axis in range(3):
            u, v = [a for a in range(3) if a != axis]
            for side, touches in ((0, origin[axis] == 0), (1, origin[axis]+span == domain[axis])):
                if not touches:
                    continue
                for i in range(origin[u], origin[u]+span):
                    for j in range(origin[v], origin[v]+span):
                        leaf_of[(axis, side, i, j)] = cell
    return leaf_of

# endregion


def build_closed_volumes(closure, *, sample_stacks, relations, stack_isovalues, separator=None,
                         partition=0, region_volume=None):
    """Closed unit shells for one partition (whole box, or one fault bank).

    ``closure`` is ``extract_adaptive_topology(..., closure=True)['closure']``.
    ``sample_stacks(points) -> (values (S, M), gradients (S, M, 3))`` returns
    raw stack fields for this partition (bank-conditioned for a fault bank).
    ``separator`` = (surface index, sign) of the bank-normalised fault separator
    (positive side outside the partition), or None for the whole box.
    Returns keyed rim vertices, extra surface triangles, unit shells and a report.
    """
    identities = closure['identities']
    levels = np.asarray(closure['levels'], dtype=float)
    surface_stacks = np.array([g for g, _ in identities], dtype=np.int64)
    n = len(identities)
    domain = np.asarray(closure['domain'], dtype=np.int64)
    bounds, spacing = np.asarray(closure['bounds'], dtype=float), np.asarray(closure['spacing'], dtype=float)
    signs = np.ones(n)
    raw_levels = levels.copy()
    if separator is not None:
        sep, sign = separator
        signs[sep] = sign
        raw_levels[sep] = float(stack_isovalues[surface_stacks[sep]][identities[sep][1]])

    def surface_fields(stack_values):
        return signs[:, None]*(np.asarray(stack_values)[surface_stacks]-raw_levels[:, None])

    # region Rim samples and clipping
    index, coords, triangles, squares, outward = _rim_lattice(domain)
    node_xyz = bounds[::2]+coords*spacing
    node_stacks, _ = sample_stacks(node_xyz)
    node_stacks = np.asarray(node_stacks, dtype=float)
    node_fields = surface_fields(node_stacks)
    if np.any(np.abs(node_fields) < 1e-10):
        raise ValueError('sample_aligned_rim: a finest rim node lies on a surface')
    node_signs = node_fields > 0

    positions = {}

    def node_key(a):
        key = ('rn', int(a))
        positions.setdefault(key, node_xyz[a])
        return key

    def crossing_key(s, a, b):
        a, b = (a, b) if a < b else (b, a)
        key = ('rc', int(s), int(a), int(b))
        if key not in positions:
            fa, fb = node_fields[s, a], node_fields[s, b]
            t = fa/(fa-fb)
            positions[key] = node_xyz[a]+t*(node_xyz[b]-node_xyz[a])
        return key, a, b

    regions = []  # (triangle, polygon [(key, bary, carrier)])
    for t_index, tri in enumerate(triangles):
        poly = [(node_key(tri[k]), np.eye(3)[k], ('E', int(min(tri[k], tri[(k+1) % 3])),
                                                  int(max(tri[k], tri[(k+1) % 3]))))
                for k in range(3)]
        pieces = [poly]
        tri_fields = node_fields[:, tri]
        for s in np.flatnonzero(np.any(node_signs[:, tri], axis=1) & ~np.all(node_signs[:, tri], axis=1)):
            split_pieces = []
            for piece in pieces:
                split_pieces.extend(_split_polygon(piece, s, tri, tri_fields, crossing_key, positions,
                                                   node_xyz, t_index))
            pieces = split_pieces
        regions.extend((t_index, piece) for piece in pieces)
    # endregion

    # region Region units (linear stack values at region centroids)
    centroids = np.array([np.mean([bary for _, bary, _ in piece], axis=0) for _, piece in regions])
    region_stacks = np.einsum('rk,srk->sr', centroids, node_stacks[:, triangles[[t for t, _ in regions]]])
    region_fields = surface_fields(region_stacks)
    outside = region_fields[separator[0]] > 0 if separator is not None else np.zeros(len(regions), dtype=bool)
    active, interval = classify_units(region_stacks, relations, stack_isovalues)
    region_units = [OUTSIDE if out else (partition, int(g), int(k))
                    for out, g, k in zip(outside.tolist(), active.tolist(), interval.tolist())]
    # endregion

    # region Caps and interface contour edges
    caps, cap_units = [], []
    contour_sides = defaultdict(list)
    for r_index, ((t_index, piece), unit) in enumerate(zip(regions, region_units)):
        keys = [key for key, _, _ in piece]
        for k, (key, _, carrier) in enumerate(piece):
            if carrier[0] == 'C':
                contour_sides[(carrier[1], frozenset((key, keys[(k+1) % len(keys)])))].append(
                    (t_index, unit, key, keys[(k+1) % len(keys)], r_index))
        if unit is OUTSIDE or len(keys) < 3:
            continue
        for tri_keys in _triangulate_convex(keys, positions):
            a, b, c = (positions[x] for x in tri_keys)
            if np.cross(b-a, c-a) @ outward[t_index] < 0:
                tri_keys = (tri_keys[0], tri_keys[2], tri_keys[1])
            caps.append(tri_keys)
            cap_units.append(unit)
    # The two sides of a rim contour edge are labelled exactly by its regions;
    # fans and strips touching the rim take their units from here, so they
    # always agree with the caps they close against.
    interfaces = []
    crossing_sides = defaultdict(set)
    for (s, _), sides in contour_sides.items():
        if len(sides) != 2:
            raise ValueError('inconsistent_rim_contour: contour edge must border two rim regions')
        (t_index, unit_a, p, q, r_a), (_, unit_b, _, _, r_b) = sides
        if unit_a == unit_b:
            continue
        a_positive = region_fields[s, r_a] > 0
        if a_positive == (region_fields[s, r_b] > 0):
            raise ValueError('inconsistent_rim_contour: contour edge does not separate its surface sides')
        side_units = (unit_b, unit_a) if a_positive else (unit_a, unit_b)
        interfaces.append((int(s), t_index, p, q, side_units))
        for key in (p, q):
            if key[0] == 'rc':
                crossing_sides[key].add(side_units)
    # endregion

    # region Strips and fans to boundary leaves
    leaf_of = _boundary_leaf_map(closure['origins'], closure['spans'], domain)
    boundary_keys = closure['boundary_keys']
    surface_triangles = [[] for _ in range(n)]
    components = defaultdict(list)
    rim_sides = [[] for _ in range(n)]
    for s, t_index, p, q, side_units in interfaces:
        leaf = leaf_of[squares[t_index]]
        vertex = boundary_keys.get((s, leaf))
        if vertex is None:
            raise ValueError('unsupported_rim_branches: rim contour crosses a boundary leaf without a vertex')
        surface_triangles[s].append((vertex, p, q))
        rim_sides[s].append(side_units)
        components[(s, leaf)].append((p, q))
    for (s, leaf), segments in components.items():
        if _component_count(segments) != 1:
            raise ValueError('unsupported_rim_branches: several rim contour branches in one boundary leaf')
    edges = closure['edges']
    for row in closure['rows']:
        s, edge = row['surface'], edges[row['edge']]
        start = np.asarray(edge['origin'], dtype=np.int64)
        step = np.zeros(3, dtype=np.int64)
        step[edge['axis']] = 1
        nodes = [index[tuple((start+k*step).tolist())] for k in range(edge['span']+1)]
        crossed = [(a, b) for a, b in zip(nodes, nodes[1:]) if node_signs[s, a] != node_signs[s, b]]
        if len(crossed) != 1:
            raise ValueError('unsupported_rim_branches: boundary edge crossing differs on the finest rim')
        key, _, _ = crossing_key(s, *crossed[0])
        if len(crossing_sides.get(key, ())) != 1:
            raise ValueError('unsupported_rim_branches: boundary crossing is not one rim interface')
        surface_triangles[s].append((*row['keys'], key))
        rim_sides[s].append(next(iter(crossing_sides[key])))
    # endregion

    # region Orientation and unit sides
    all_positions = dict(closure['positions'])
    all_positions.update(positions)
    core = [[tuple(row['keys']) for row in rows] for rows in closure['triangles']]
    oriented_new = _orient_by_gradient(surface_triangles, all_positions, sample_stacks, surface_stacks, signs)
    labelled = []  # (keys, kind, surface)
    for s in range(n):
        kind = 'fault_skin' if separator is not None and s == separator[0] else 'interface'
        labelled.extend((keys, kind, s) for keys in core[s])
    sides = _surface_sides([(keys, s) for keys, _, s in labelled], all_positions, sample_stacks, relations,
                           stack_isovalues, identities, surface_stacks, signs, raw_levels, separator, partition)
    for s in range(n):
        kind = 'fault_skin' if separator is not None and s == separator[0] else 'interface'
        labelled.extend((keys, kind, s) for keys in oriented_new[s])
        sides.extend(rim_sides[s])
    # endregion

    # region Shells and audit
    shells = defaultdict(list)
    non_interface = 0
    patches = []
    for (keys, kind, s), (negative, positive) in zip(labelled, sides):
        if negative == positive:
            non_interface += 1
            continue
        patches.append((keys, kind, negative, positive))
    patches.extend((keys, 'extent', unit, OUTSIDE) for keys, unit in zip(caps, cap_units))
    for keys, kind, negative, positive in patches:
        if negative is not OUTSIDE:
            shells[negative].append(keys)
        if positive is not OUTSIDE:
            shells[positive].append((keys[0], keys[2], keys[1]))
    report = audit_shells(shells, all_positions)
    report.update(partition=partition, region_volume=region_volume, non_interface_triangles=non_interface,
                  rim_nodes=len(coords), rim_regions=len(regions), rim_interface_edges=len(interfaces),
                  strip_triangles=len(closure['rows']), fan_triangles=sum(len(c) for c in components.values()))
    if region_volume is not None:
        total = sum(report['volumes'].values())
        report['volume_error'] = abs(total-region_volume)
        report['volumes_sum_to_region'] = bool(abs(total-region_volume) <= 1e-6*max(1., abs(region_volume)))
    # endregion
    return dict(positions=all_positions, rim_positions=positions, surface_triangles=oriented_new,
                patches=patches, shells=dict(shells), report=report)


def _split_polygon(piece, s, tri, tri_fields, crossing_key, positions, node_xyz, t_index):
    """Split a convex rim polygon by surface ``s`` (linear on the triangle)."""
    values = [bary @ tri_fields[s] for _, bary, _ in piece]
    negative = [v < 0 for v in values]
    if all(negative) or not any(negative):
        return [piece]
    parts = {True: [], False: []}
    count = len(piece)
    for k in range(count):
        key, bary, carrier = piece[k]
        side = negative[k]
        parts[side].append((key, bary, carrier))
        if negative[(k+1) % count] != side:
            x_key, x_bary = _crossing(s, carrier, tri, tri_fields, crossing_key, positions, node_xyz, t_index)
            parts[side].append((x_key, x_bary, ('C', int(s))))
            parts[not side].append((x_key, x_bary, carrier))
    return [part for part in parts.values() if len(part) >= 3]


def _crossing(s, carrier, tri, tri_fields, crossing_key, positions, node_xyz, t_index):
    if carrier[0] == 'E':
        key, a, b = crossing_key(s, carrier[1], carrier[2])
        local = {int(v): k for k, v in enumerate(tri)}
        fa, fb = tri_fields[s, local[a]], tri_fields[s, local[b]]
        t = fa/(fa-fb)
        bary = np.zeros(3)
        bary[local[a]], bary[local[b]] = 1-t, t
        return key, bary
    other = carrier[1]
    first, second = sorted((int(s), int(other)))
    key = ('rj', first, second, int(t_index))
    system = np.vstack((tri_fields[first], tri_fields[second], np.ones(3)))
    bary = np.linalg.solve(system, np.array([0., 0., 1.]))
    if np.any(bary < -1e-12):
        raise ValueError('degenerate_rim_junction: contour intersection outside its rim triangle')
    positions.setdefault(key, bary @ node_xyz[tri])
    return key, bary


def _component_count(segments):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p, q in segments:
        parent[find(p)] = find(q)
    return len({find(x) for x in parent})


def _orient_by_gradient(surface_triangles, positions, sample_stacks, surface_stacks, signs):
    """Wind new surface triangles along their surface's actual gradient (production rule)."""
    flat = [(s, keys) for s, rows in enumerate(surface_triangles) for keys in rows]
    oriented = [[] for _ in surface_triangles]
    if not flat:
        return oriented
    corners = np.array([[positions[k] for k in keys] for _, keys in flat])
    _, gradients = sample_stacks(corners.mean(axis=1))
    gradients = np.asarray(gradients)
    normals = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
    for i, (s, keys) in enumerate(flat):
        gradient = signs[s]*gradients[surface_stacks[s], i]
        dot = normals[i] @ gradient
        if abs(dot) <= 1e-9*np.linalg.norm(normals[i])*np.linalg.norm(gradient):
            raise ValueError('unresolved_rim_winding: surface gradient parallel to a rim triangle')
        oriented[s].append(keys if dot > 0 else (keys[0], keys[2], keys[1]))
    return oriented


def _surface_sides(items, positions, sample_stacks, relations, stack_isovalues, identities, surface_stacks,
                   signs, raw_levels, separator, partition):
    """(unit on the negative side, unit on the positive side) of each surface triangle.

    The surface's own field is forced just below / above its level at the
    triangle centroid; every other field keeps its actual value there.
    """
    if not items:
        return []
    centroids = np.array([np.mean([positions[k] for k in keys], axis=0) for keys, _ in items])
    values, _ = sample_stacks(centroids)
    values = np.asarray(values, dtype=float)
    sides = []
    for side in (-1., 1.):
        forced = values.copy()
        for i, (_, s) in enumerate(items):
            g = surface_stacks[s]
            levels_g = np.asarray(stack_isovalues[g], dtype=float)
            gaps = np.diff(np.sort(levels_g))
            delta = 1e-6*(gaps.min() if len(gaps) else 1.)
            forced[g, i] = raw_levels[s]+side*delta/signs[s]
        units = list(zip(*classify_units(forced, relations, stack_isovalues)))
        out = np.zeros(len(items), dtype=bool)
        if separator is not None:
            sep, sign = separator
            out = sign*(forced[surface_stacks[sep]]-raw_levels[sep]) > 0
        sides.append([OUTSIDE if o else (partition, int(g), int(k)) for o, (g, k) in zip(out.tolist(), units)])
    return list(zip(*sides))


def _triangulate_convex(keys, positions):
    """Triangulate a convex rim region without zero-area triangles.

    Region sides carry collinear points shared with neighbours (crossings), so a
    plain fan would create zero-area triangles whose long edge passes through
    them (geometric hanging nodes). Ears are clipped only at strictly convex
    corners, the first in ring order, so the result depends on topology rather
    than on round-off; collinear points always end up as vertices.
    """
    ring = list(keys)
    xyz = {k: np.asarray(positions[k], dtype=float) for k in ring}
    normal = np.zeros(3)
    for i in range(len(ring)):  # Newell normal of the polygon
        p, q = xyz[ring[i]], xyz[ring[(i+1) % len(ring)]]
        normal += np.cross(p, q)
    scale = max(np.linalg.norm(xyz[k]-xyz[ring[0]]) for k in ring) or 1.
    triangles = []
    while len(ring) > 3:
        best = None
        for i in range(len(ring)):
            u, v, w = xyz[ring[i-1]], xyz[ring[i]], xyz[ring[(i+1) % len(ring)]]
            if np.cross(v-u, w-v) @ normal > 1e-9*scale**2*np.linalg.norm(normal):
                best = i  # strictly convex corner; collinear ones are skipped
                break
        if best is None:
            raise ValueError('degenerate_rim_region: no strictly convex corner left')
        triangles.append((ring[best-1], ring[best], ring[(best+1) % len(ring)]))
        ring.pop(best)
    u, v, w = (xyz[k] for k in ring)
    if np.linalg.norm(np.cross(v-u, w-u)) > 1e-12*scale**2:
        triangles.append(tuple(ring))
    return triangles


def audit_shells(shells, positions):
    """Closure (each edge exactly twice, opposite directions) and signed volume per unit."""
    report = dict(closed={}, volumes={}, degenerate={}, open_edges={})
    for unit, triangles in shells.items():
        keys = sorted({k for tri in triangles for k in tri})
        ids = {k: i for i, k in enumerate(keys)}
        faces = np.array([[ids[k] for k in tri] for tri in triangles], dtype=np.int64).reshape(-1, 3)
        xyz = np.array([positions[k] for k in keys]).reshape(-1, 3)
        directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        codes = directed[:, 0]*len(keys)+directed[:, 1]
        reverse = directed[:, 1]*len(keys)+directed[:, 0]
        unique, counts = np.unique(codes, return_counts=True)
        closed = bool(np.all(counts == 1) and np.isin(reverse, unique).all())
        a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
        area = np.linalg.norm(np.cross(b-a, c-a), axis=1)
        report['closed'][unit] = closed
        report['open_edges'][unit] = int(np.sum(~np.isin(reverse, unique)) + np.sum(counts > 1))
        report['degenerate'][unit] = int(np.sum(area < 1e-14))
        report['volumes'][unit] = float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum()/6)
    report['all_closed'] = all(report['closed'].values()) and not any(report['degenerate'].values())
    report['all_positive'] = all(v > 0 for v in report['volumes'].values())
    return report
