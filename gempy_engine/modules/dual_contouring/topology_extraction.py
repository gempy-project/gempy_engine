"""CPU topology evidence and branch-aware dual incidence; no module orchestration."""

from collections import Counter
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


def classify_cell_branches(values, valid_edges, unsafe_warnings=None):
    """Classify sampled boundary contours without prescribing Hermite normals.

    Affine approximants, bilinear extrusions and single-branch strictly monotone
    sampled cells have supported connectivity. Other interiors need more data.
    Face saddle ties are rejected deterministically, never arbitrarily joined.
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
    for cell in range(len(values)):
        if np.all(np.abs(values[cell]) < tolerance):
            raise ValueError('degenerate_field: entire cell lies on the interface')
        cube = values[cell].reshape(2, 2, 2)
        extrusion = any(np.allclose(np.take(cube, 0, axis=a), np.take(cube, 1, axis=a),
                                   atol=tolerance, rtol=0) for a in range(3))
        monotone = any(np.all(np.diff(cube, axis=a) > tolerance) or
                       np.all(np.diff(cube, axis=a) < -tolerance) for a in range(3))
        if not affine[cell] and not extrusion and not monotone:
            raise ValueError('insufficient_interior_topology: unresolved sampled cell; supply refinement or interior evidence')
        if not valid_edges[cell].any():
            continue
        if np.any(np.abs(values[cell]) < tolerance):
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


def plan_joint_representatives(coordinates, corners_xyz, fields, ownership, branches,
                               coefficients, affine, pairs, identities, edge_xyz, edge_normals, valid_edges,
                               unsafe_warnings=None):
    """Establish joint cell constraints and shared primal-face evidence, not meshes."""
    tolerance = 1e-10
    junctions = {}
    face_evidence = {}
    for controller, target in sorted(pairs, key=lambda p: (identities[p[0]], identities[p[1]])):
        own = ownership[target]
        if np.all(own > 0) or np.all(own < 0):
            continue
        if not (np.allclose(own, fields[controller], atol=tolerance, rtol=0)
                or np.allclose(own, -fields[controller], atol=tolerance, rtol=0)):
            raise ValueError('unsupported_ownership: need signed eligible controller field')
        for cell in range(len(coordinates)):
            if not valid_edges[controller, cell].any() or not valid_edges[target, cell].any():
                continue
            if branches[controller][cell].max() > 0 or branches[target][cell].max() > 0:
                raise ValueError('unsupported_junction_branches: contact requires one sampled branch per participant')
            if not affine[controller][cell] or not affine[target][cell]:
                raise ValueError('unsupported_junction_field: need cell-affine corner approximants for consistent junction-face evidence')
            normals = np.stack((coefficients[controller][cell, :3], coefficients[target][cell, :3]))
            rhs = -np.array([coefficients[controller][cell, 3], coefficients[target][cell, 3]])
            row_norms = np.linalg.norm(normals, axis=1)
            if np.any(row_norms < 1e-12):
                raise ValueError('degenerate_junction_constraint: nonzero affine gradients required')
            normals = normals / row_norms[:, None]
            rhs = rhs / row_norms
            direction = np.cross(*normals)
            length = np.linalg.norm(direction)
            if length < tolerance:
                if np.linalg.matrix_rank(np.column_stack((normals, rhs)), tol=tolerance) == 1:
                    raise ValueError('coincident_interfaces: no unique junction line')
                continue
            direction /= length
            point = np.linalg.lstsq(normals, rhs, rcond=None)[0]
            lo, hi = -np.inf, np.inf
            for a in range(3):
                if abs(direction[a]) < tolerance:
                    if point[a] < -tolerance or point[a] > 1 + tolerance:
                        lo, hi = 1., 0.
                        break
                else:
                    bounds = sorted((-point[a] / direction[a], (1 - point[a]) / direction[a]))
                    lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
            if hi - lo <= tolerance:
                continue  # No nondegenerate joint segment, even if cells coincide.
            endpoints = np.stack((point + lo * direction, point + hi * direction))
            if np.any(endpoints < -tolerance) or np.any(endpoints > 1 + tolerance):
                continue
            if cell in junctions:
                if unsafe_warnings is None:
                    raise ValueError('unsupported_multiway_junction: competing participants in one cell')
                unsafe_warnings.append('unsupported_multiway_junction: overwriting earlier contact pair')
            if identities[controller][0] == identities[target][0]:
                raise ValueError('same_group_junction: participants must be group-distinct')
            origin = corners_xyz[cell, 0]
            spacing = corners_xyz[cell, 7] - origin
            # The seam comes from shared corner approximants, NOT independent
            # tangent planes. Minimize the actual Hermite QEF plus unit mass bias
            # along that bounded segment; curved tangent residuals need not vanish.
            participants = sorted((controller, target), key=lambda s: identities[s])
            hermite_points = np.concatenate([edge_xyz[s, cell, valid_edges[s, cell]] for s in participants])
            hermite_normals = np.concatenate([edge_normals[s, cell, valid_edges[s, cell]] for s in participants])
            mass = hermite_points.mean(axis=0)
            physical_direction = direction * spacing
            physical_point = origin + point * spacing
            projected_normals = hermite_normals @ physical_direction
            residual_rhs = np.sum(hermite_normals * (hermite_points - physical_point), axis=1)
            numerator = np.dot(projected_normals, residual_rhs) + np.dot(mass - physical_point, physical_direction)
            denominator = np.dot(projected_normals, projected_normals) + np.dot(physical_direction, physical_direction)
            if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
                raise ValueError('degenerate_joint_qef: bounded Hermite objective is not finite and positive definite')
            parameter = numerator / denominator
            parameter = np.clip(parameter, lo, hi)
            position = physical_point + parameter * physical_direction
            members = tuple(sorted((identities[controller], identities[target])))
            key = ('joint', members, tuple(map(int, coordinates[cell])), 0)
            junctions[cell] = dict(controller=controller, target=target, key=key, position=position)
            for endpoint in endpoints:
                boundary = [(a, int(round(endpoint[a]))) for a in range(3)
                            if abs(endpoint[a]) < tolerance or abs(endpoint[a] - 1) < tolerance]
                if len(boundary) != 1:
                    if unsafe_warnings is None:
                        raise ValueError('grid_edge_junction: need extra face/support samples or refinement')
                    unsafe_warnings.append('grid_edge_junction: using first endpoint face')
                axis, side = boundary[0]
                face_origin = coordinates[cell].copy()
                face_origin[axis] += side
                face_key = (members, axis, tuple(map(int, face_origin)))
                physical_endpoint = origin + endpoint * spacing
                entries = face_evidence.setdefault(face_key, [])
                if entries and not np.allclose(entries[0][1], physical_endpoint, atol=tolerance, rtol=0):
                    raise ValueError('inconsistent_joint_face: neighboring constraints disagree')
                entries.append((cell, physical_endpoint))
    return junctions, face_evidence


def plan_dual_quads(coordinates, domain_shape, valid_edges, branches, ownership_at_edges,
                    identities, junctions):
    """One canonical primal-edge incidence plan, shared by regular and joint output."""
    coordinate_index = {tuple(c): i for i, c in enumerate(coordinates)}
    plans = [[] for _ in identities]
    for surface in range(len(identities)):
        canonical = {}
        for cell, edge in zip(*np.nonzero(valid_edges[surface])):
            origin = coordinates[cell] + EDGE_OFFSETS[edge]
            key = (*map(int, origin), int(edge // 4))
            retained = ownership_at_edges[surface, cell, edge] > 0
            entries = canonical.setdefault(key, [])
            if entries and entries[0][2] != retained:
                raise ValueError('inconsistent_ownership_edge: shared sample decisions disagree')
            entries.append((cell, edge, retained))
        for edge_key, entries in sorted(canonical.items()):
            if not entries[0][2]:
                continue
            origin, axis = np.array(edge_key[:3]), edge_key[3]
            incident = origin + INCIDENT_OFFSETS[axis]
            if np.any(incident < 0) or np.any(incident >= np.array(domain_shape)):
                continue  # Open crop, exactly as complete-quad production DC.
            cells = [coordinate_index[tuple(c)] for c in incident]
            local_edges = []
            for cell in cells:
                matches = [e for c, e, _ in entries if c == cell]
                if len(matches) != 1:
                    raise ValueError('inconsistent_primal_edge: four Hermite incidences required')
                local_edges.append(matches[0])
            regular_keys = [('regular', (identities[surface],), tuple(map(int, coordinates[c])),
                             int(branches[surface][c, e])) for c, e in zip(cells, local_edges)]
            joint_keys = [junctions[c]['key'] if c in junctions and surface in
                          (junctions[c]['controller'], junctions[c]['target']) else key
                          for c, key in zip(cells, regular_keys)]
            plans[surface].append(dict(edge=edge_key, cells=cells, local_edges=local_edges,
                                       regular_keys=regular_keys, keys=joint_keys))
    return plans


def validate_joint_incidence(plans, junctions, face_evidence, coordinates, domain_shape):
    """Validate seam support on QUADS before triangle allocation or emission."""
    per_surface = []
    for surface_plans in plans:
        counts = Counter()
        for plan in surface_plans:
            keys = plan['keys']
            for a, b in zip(keys, keys[1:] + keys[:1]):
                if a[0] == b[0] == 'joint':
                    counts[tuple(sorted((a, b)))] += 1
            if any(all(keys[i][0] == 'joint' for i in triangle)
                   for triangle in ((0, 1, 3), (2, 3, 1))):
                raise ValueError('unsupported_shared_patch: quad split would emit a fully shared triangle')
        per_surface.append(counts)
    expected = set()
    for (_, axis, origin), entries in face_evidence.items():
        if len(entries) == 1:
            if origin[axis] not in (0, domain_shape[axis]):
                raise ValueError('unresolved_junction_face: interior seam has missing neighboring evidence')
            continue
        if len(entries) != 2:
            raise ValueError('inconsistent_joint_face: expected two incident cells')
        a, b = (junctions[c] for c, _ in entries)
        seam = tuple(sorted((a['key'], b['key'])))
        expected.add(seam)
        if per_surface[a['controller']][seam] != 2 or per_surface[a['target']][seam] != 1:
            raise ValueError('unsupported_dual_junction_incidence: refine; controller needs two sides and target one')
    observed = set().union(*(set(c) for c in per_surface))
    if observed != expected:
        raise ValueError('unsupported_dual_junction_incidence: incidence lacks matching primal-face evidence')
    if junctions and not expected:
        raise ValueError('insufficient_dual_seam_support: need at least two neighboring junction cells')
    return expected


def emit_dual_triangles(plans, vertices, key_to_id, edge_normals, unsafe_warnings=None):
    """Production complete-quad 1--3 split and winding, with branch-aware IDs."""
    faces, origins, affected = [], [], []
    for surface, surface_plans in enumerate(plans):
        triangles, primal_edges, neighborhood = [], [], []
        for plan in surface_plans:
            quad = [key_to_id[k] for k in plan['keys']]
            reference = sum(edge_normals[surface, c, e] for c, e in zip(plan['cells'], plan['local_edges']))
            for offsets in ((0, 1, 3), (2, 3, 1)):
                triangle = [quad[i] for i in offsets]
                a, b, c = vertices[triangle]
                normal = np.cross(b - a, c - a)
                if np.linalg.norm(normal) < 1e-12:
                    if unsafe_warnings is None:
                        raise ValueError('degenerate_dual_face: refine or change sampled support')
                    unsafe_warnings.append('degenerate_dual_face: emitting triangle below area threshold')
                dot = np.dot(normal, reference)
                if abs(dot) < 1e-12:
                    if unsafe_warnings is None:
                        raise ValueError('unresolved_dual_winding: Hermite orientation is degenerate')
                    unsafe_warnings.append('unresolved_dual_winding: using raw orientation sign')
                if dot < 0:
                    triangle[1], triangle[2] = triangle[2], triangle[1]
                triangles.append(triangle)
                primal_edges.append(plan['edge'])
                neighborhood.append(any(k[0] == 'joint' for k in plan['keys']))
        faces.append(np.array(triangles, dtype=np.int64).reshape(-1, 3))
        origins.append(np.array(primal_edges, dtype=np.int64).reshape(-1, 4))
        affected.append(np.array(neighborhood, dtype=bool))
    return faces, origins, affected


def triangle_quality(vertices, faces):
    """Angles in degrees; aspect = longest edge / smallest altitude (L^2 / 2A)."""
    if not len(faces):
        return dict(count=0, minimum_angle_degrees=None, angle_percentiles=None, aspect_percentiles=None)
    points = vertices[faces]
    lengths = np.stack([np.linalg.norm(points[:, (i + 1) % 3] - points[:, i], axis=1) for i in range(3)], axis=1)
    area_twice = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
    angles = np.stack([np.degrees(np.arccos(np.clip((lengths[:, i] ** 2 + lengths[:, (i + 1) % 3] ** 2 -
                                                   lengths[:, (i + 2) % 3] ** 2) /
                                                  (2 * lengths[:, i] * lengths[:, (i + 1) % 3]), -1, 1)))
                       for i in range(3)], axis=1)
    minimum = angles.min(axis=1)
    aspect = lengths.max(axis=1) ** 2 / area_twice
    return dict(count=len(faces), minimum_angle_degrees=float(minimum.min()),
                percentile_levels=[0, 5, 50, 95, 100],
                angle_percentiles=np.percentile(minimum, [0, 5, 50, 95, 100]).tolist(),
                aspect_percentiles=np.percentile(aspect, [0, 5, 50, 95, 100]).tolist())
