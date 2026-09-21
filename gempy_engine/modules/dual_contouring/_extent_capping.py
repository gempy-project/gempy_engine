"""Best-effort NumPy caps for a uniform dual-contouring lattice.

This is not a mesh repair operation: only the six supplied extent faces are
sampled. Internal mask/fault holes remain untouched. Ambiguous/tied contours,
missing cells and unconstrained QEFs can leave holes; inspect ``capping_report``.
"""

from collections import defaultdict
from itertools import product
import warnings

import numpy as np

from ._scalar_crossing import scalar_crossing_parameters


def boundary_lattice(shape, extent):
    """Return integer corners, physical positions, and face metadata.

    ``shape`` is the three *cell* counts. ``extent`` is
    ``(xmin, xmax, ymin, ymax, zmin, zmax)``. Corners are unique globally,
    including box edges/corners. Metadata contains ``triangles`` (indices into
    these arrays) and ``cells`` (one owning in-domain cell per triangle).
    Faces are triangulated with a consistent low-to-high diagonal and outward
    winding. Positions use the exact supplied bounds, not QEF-derived bounds.

    Storage scales with the full boundary area, not the active contour. The
    Python lists/dictionaries used during construction have substantial peak
    memory overhead on large grids. Reuse this tuple via ``boundary_geometry``
    rather than constructing it twice; this is not a streaming implementation.
    """
    shape = np.asarray(shape)
    bounds = np.asarray(extent, dtype=float).reshape(3, 2)
    if shape.shape != (3,) or np.any(shape < 1) or np.any(shape != shape.astype(int)):
        raise ValueError("shape must contain three positive integer cell counts")
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("extent must contain finite increasing bounds")
    shape = shape.astype(int)
    coordinates, triangles, cells, lookup = [], [], [], {}
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for side in (0, 1):
            for i, j in product(range(shape[u]), range(shape[v])):
                corners = []
                for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    c = [0, 0, 0]
                    c[axis], c[u], c[v] = side * shape[axis], i + du, j + dv
                    key = tuple(c)
                    if key not in lookup:
                        lookup[key] = len(coordinates)
                        coordinates.append(c)
                    corners.append(lookup[key])
                cell = [0, 0, 0]
                cell[axis], cell[u], cell[v] = side * (shape[axis] - 1), i, j
                for a, b, c in ((0, 1, 2), (0, 2, 3)):
                    triangles.append([corners[a], corners[b if side else c], corners[c if side else b]])
                    cells.append(cell.copy())
    coordinates = np.asarray(coordinates, dtype=np.int64)
    positions = bounds[:, 0] + coordinates / shape * (bounds[:, 1] - bounds[:, 0])
    for axis in range(3):
        positions[coordinates[:, axis] == shape[axis], axis] = bounds[axis, 1]
    return coordinates, positions, {
        "triangles": np.asarray(triangles, dtype=np.int64),
        "cells": np.asarray(cells, dtype=np.int64),
    }


def cap_mesh(mesh, cell_coordinates, shape, extent, boundary_coordinates,
             boundary_scalar, isovalue, *, boundary_geometry=None, skip_reason=None):
    """Mutate NumPy ``mesh.vertices``/``mesh.edges`` and return ``mesh``.

    ``edges`` means triangle indices (M, 3). ``cell_coordinates`` is (N, 3)
    in the exact row order of the original QEF vertices, not sorted order.
    Boundary samples may be permuted, but must cover ``boundary_lattice``.
    The retained solid is ``scalar <= isovalue``. Original vertices/triangles
    keep their indices and winding. ``mesh.capping_report`` records counts and
    whole-mesh topology audits; a RuntimeWarning summarizes limitations.
    Invalid API array shapes raise ValueError; unsupported geometry does not.
    ``boundary_geometry`` may be the unmodified tuple from ``boundary_lattice``
    for this same shape/extent, avoiding a second lattice construction.

    Report ``open_edges_before/after`` counts incidence-one edges.
    ``non_extent_open_edges_before/after`` excludes edges whose endpoints share
    an extent face: original QEF membership comes from owning cells, whereas
    new cap points use physical positions. This is a cell-level classification,
    not proof that an opening in a boundary cell was caused by the extent.
    ``watertight`` audits topology only; ``closure_success`` additionally
    requires valid finite geometry and no missing/escaped/unsupported cases.
    ``added_cap_triangles`` and ``added_transition_triangles`` count retained
    triangles after cleanup. ``cap_max_plane_error`` is the maximum, over cap
    triangles, of the distance to their nearest common extent plane (physical
    units, zero if no cap). Invalid original geometry is preserved and reported
    unsuccessful without attempting additions.
    """
    report = dict(added_vertices=0, added_triangles=0, contour_edges=0,
                  missing_qef=0, escaped_qef=0, unsupported_topology=0,
                  discarded_components=0, removed_triangles=0, warnings=[],
                  added_cap_triangles=0, added_transition_triangles=0,
                  cap_max_plane_error=0.0, closure_success=False)
    mesh.capping_report = report

    def audit():
        result = {}
        v = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.edges)
        valid = np.all(np.isfinite(triangles) & (triangles >= 0) & (triangles < len(v))
                       & (triangles == np.floor(triangles)), axis=1)
        result["invalid_triangles"] = int(np.count_nonzero(~valid))
        result["nonfinite_vertices"] = int(np.count_nonzero(~np.isfinite(v).all(axis=1)))
        triangles = triangles[valid].astype(np.int64)
        p = v[triangles]
        with np.errstate(over="ignore", invalid="ignore"):
            cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        result["nonfinite_triangles"] = int(np.count_nonzero(~np.isfinite(cross).all(axis=1)))
        result["degenerate_triangles"] = int(np.count_nonzero(np.all(cross == 0, axis=1)))
        membership = np.zeros((len(v), 3, 2), dtype=bool)
        membership[:len(cells), :, 0] = cells == 0
        membership[:len(cells), :, 1] = cells == shape - 1
        tolerance = 64 * np.finfo(float).eps * np.maximum(1, np.abs(bounds))
        membership[len(cells):] = np.abs(v[len(cells):, :, None] - bounds) <= tolerance
        incidence = defaultdict(list)
        links = defaultdict(list)
        for a, b, c in triangles:
            for x, y in ((a, b), (b, c), (c, a)):
                incidence[tuple(sorted((int(x), int(y))))].append((int(x), int(y)))
            for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
                links[int(x)].append((int(y), int(z)))
        openings = [edge for edge, entries in incidence.items() if len(entries) == 1]
        result["open_edges"] = len(openings)
        result["non_extent_open_edges"] = sum(not np.any(membership[a] & membership[b]) for a, b in openings)
        result["nonmanifold_edges"] = sum(len(e) > 2 for e in incidence.values())
        result["orientation_conflicts"] = sum(len(e) == 2 and e[0] == e[1] for e in incidence.values())
        bad_links = 0
        for edges in links.values():
            graph = defaultdict(list)
            for a, b in edges:
                graph[a].append(b)
                graph[b].append(a)
            seen, todo = set(), [next(iter(graph))]
            while todo:
                node = todo.pop()
                if node not in seen:
                    seen.add(node)
                    todo.extend(graph[node])
            degrees = [len(neighbors) for neighbors in graph.values()]
            if len(seen) != len(graph) or max(degrees) > 2 or degrees.count(1) not in (0, 2):
                bad_links += 1
        result["nonmanifold_vertices"] = bad_links
        result["watertight"] = bool(incidence) and not any(result[k] for k in (
            "invalid_triangles", "open_edges", "nonmanifold_edges", "orientation_conflicts", "nonmanifold_vertices"))
        return result

    def finish():
        after = audit()
        report.update(after)
        report["open_edges_after"] = after["open_edges"]
        report["non_extent_open_edges_after"] = after["non_extent_open_edges"]
        report["boundary_edges"] = after["open_edges"]
        report["closure_success"] = report["watertight"] and not any(report[k] for k in (
            "missing_qef", "escaped_qef", "unsupported_topology", "invalid_triangles",
            "nonfinite_vertices", "nonfinite_triangles", "degenerate_triangles"))
        plane_tolerance = 64 * np.finfo(float).eps * max(1, np.max(np.abs(bounds)))
        if report["cap_max_plane_error"] > plane_tolerance:
            report["closure_success"] = False
            report["warnings"].append("Cap triangles are off the supplied extent planes")
        for key in ("missing_qef", "escaped_qef", "unsupported_topology", "boundary_edges",
                    "nonmanifold_edges", "orientation_conflicts", "nonmanifold_vertices",
                    "invalid_triangles", "nonfinite_vertices", "nonfinite_triangles", "degenerate_triangles"):
            if report[key]:
                report["warnings"].append(f"{key}: {report[key]}")
        if report["warnings"]:
            warnings.warn("Extent capping best effort: " + "; ".join(report["warnings"]),
                          RuntimeWarning, stacklevel=2)
        return mesh

    if not isinstance(mesh.vertices, np.ndarray) or not isinstance(mesh.edges, np.ndarray):
        report["unsupported_topology"] += 1
        report["warnings"].append("Only NumPy meshes are supported")
        warnings.warn(report["warnings"][0], RuntimeWarning, stacklevel=2)
        return mesh
    vertices = np.asarray(mesh.vertices)
    cells = np.asarray(cell_coordinates)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or cells.shape != vertices.shape:
        raise ValueError("cell_coordinates must have one (x, y, z) row per QEF vertex")
    if mesh.edges.ndim != 2 or mesh.edges.shape[1] != 3:
        raise ValueError("mesh.edges must be triangle indices (M, 3)")
    bounds = np.asarray(extent, dtype=float).reshape(3, 2)
    shape = np.asarray(shape)
    if shape.shape != (3,) or np.any(shape < 1) or not np.isfinite(shape).all() or np.any(shape != np.floor(shape)):
        raise ValueError("shape must contain three positive integer cell counts")
    if not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("extent must contain finite increasing bounds")
    shape = shape.astype(int)
    before = audit()
    report.update({key + "_before": value for key, value in before.items()})
    if skip_reason is not None:
        report["unsupported_topology"] += 1
        report["warnings"].append(skip_reason)
        report["skipped_reason"] = skip_reason
        return finish()
    if any(before[k] for k in ("invalid_triangles", "nonfinite_vertices", "nonfinite_triangles", "degenerate_triangles")):
        return finish()
    coords, positions, faces = (boundary_lattice(shape, extent) if boundary_geometry is None
                                else boundary_geometry)
    supplied = np.asarray(boundary_coordinates)
    scalar = np.asarray(boundary_scalar, dtype=float).reshape(-1)
    if supplied.shape != (len(scalar), 3):
        raise ValueError("boundary_coordinates and boundary_scalar must have matching rows")
    samples = {tuple(c): s for c, s in zip(supplied, scalar)}
    if len(samples) != len(supplied) or len(samples) != len(coords) or any(tuple(c) not in samples for c in coords):
        report["unsupported_topology"] += 1
        report["warnings"].append("Incomplete or duplicate boundary lattice")
        return finish()
    values = np.asarray([samples[tuple(c)] for c in coords])
    # Canonical endpoint order gives every shared edge the identical parameter.
    source = faces["triangles"]
    ends = np.roll(source, -1, axis=1)
    starts, ends = np.minimum(source, ends), np.maximum(source, ends)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            crossing, parameters = scalar_crossing_parameters(values[starts], values[ends], isovalue)
    except ValueError as error:
        report["unsupported_topology"] += 1
        report["warnings"].append(f"Nonfinite boundary crossing: {error}")
        return finish()
    step = (bounds[:, 1] - bounds[:, 0]) / shape
    qef = {}
    seen_cells = set()
    for i, cell in enumerate(cells):
        key = tuple(cell)
        if np.any(cell != np.floor(cell)) or np.any(cell < 0) or np.any(cell >= shape) or key in seen_cells:
            report["unsupported_topology"] += 1
            continue
        seen_cells.add(key)
        low = bounds[:, 0] + cell * step
        tolerance = 64 * np.finfo(float).eps * np.maximum(1, np.maximum(abs(low), abs(low + step)))
        if not np.isfinite(vertices[i]).all() or np.any(vertices[i] < low - tolerance) or np.any(vertices[i] > low + step + tolerance):
            report["escaped_qef"] += 1
            continue
        qef[key] = i

    points = list(vertices.astype(float))
    point_ids = {}

    def point(a, b=None, t=None):
        # Endpoint ties must share the lattice-point key, not an edge key.
        if b is None or values[a] == isovalue:
            key, p = ("p", a), positions[a]
        elif values[b] == isovalue:
            key, p = ("p", b), positions[b]
        else:
            a, b = sorted((a, b))
            if t == 0 or t == 1:
                endpoint = a if t == 0 else b
                key, p = ("p", endpoint), positions[endpoint]
            else:
                key = ("e", a, b)  # Shared by primal edges and face diagonals.
                p = positions[a] + t * (positions[b] - positions[a])
        if key not in point_ids:
            point_ids[key] = len(points)
            points.append(p)
        return point_ids[key]

    cap, owners = [], []
    for i, (triangle, cell) in enumerate(zip(faces["triangles"], faces["cells"])):
        polygon = []
        for j, (a, b) in enumerate(zip(triangle, np.roll(triangle, -1))):
            if values[a] <= isovalue:
                polygon.append(point(a))
            if crossing[i, j]:
                polygon.append(point(a, b, parameters[i, j]))
        polygon = list(dict.fromkeys(polygon))
        for j in range(1, len(polygon) - 1):
            cap.append((polygon[0], polygon[j], polygon[j + 1]))
            owners.append(tuple(cell))

    # Connected cap patches are seeded only by contour edges with a usable QEF.
    incidence = defaultdict(list)
    for i, triangle in enumerate(cap):
        for a, b in zip(triangle, np.roll(triangle, -1)):
            incidence[tuple(sorted((a, b)))].append((i, a, b))
    adjacency = defaultdict(set)
    contours, seeds = [], set()
    for entries in incidence.values():
        if len(entries) == 2:
            i, j = entries[0][0], entries[1][0]
            adjacency[i].add(j)
            adjacency[j].add(i)
        elif len(entries) == 1:
            i, a, b = entries[0]
            report["contour_edges"] += 1
            q = qef.get(owners[i])
            if q is None:
                report["missing_qef"] += 1
            else:
                seeds.add(i)
                contours.append((i, a, b, q))
        else:
            report["unsupported_topology"] += 1
    retained, visited = set(), set()
    for start in range(len(cap)):
        if start in visited:
            continue
        component, todo = set(), [start]
        while todo:
            i = todo.pop()
            if i not in component:
                component.add(i)
                todo.extend(adjacency[i] - component)
        visited.update(component)
        if component & seeds:
            retained.update(component)
        else:
            report["discarded_components"] += 1
    added = [cap[i] for i in sorted(retained)]
    cap_count = len(added)
    fans = [(b, a, q) for i, a, b, q in contours if i in retained]
    added.extend(fans)

    # The two in-domain cells around a boundary primal edge need one more
    # triangle. Derive its winding from the existing directed face-fan edges.
    spokes = defaultdict(list)
    for b, a, q in fans:
        spokes[(a, q)].append((a, q))
        spokes[(b, q)].append((q, b))
    residual = defaultdict(list)
    for (p, q), edges in spokes.items():
        forward = sum(a == p for a, b in edges)
        reverse = len(edges) - forward
        if forward != reverse:
            residual[p].extend([(p, q) if forward > reverse else (q, p)] * abs(forward - reverse))
    for p, edges in residual.items():
        if len(edges) == 2:
            incoming = [a for a, b in edges if b == p]
            outgoing = [b for a, b in edges if a == p]
            if (len(incoming) == len(outgoing) == 1
                    and np.sum(np.abs(cells[incoming[0]] - cells[outgoing[0]])) == 1):
                added.append((p, incoming[0], outgoing[0]))
                continue
        report["unsupported_topology"] += 1

    # Never remove/reindex original geometry, even if it already has defects.
    seen = {tuple(sorted(t)) for t in mesh.edges}
    clean = []
    for index, triangle in enumerate(added):
        key = tuple(sorted(triangle))
        a, b, c = (points[i] for i in triangle)
        cross = np.cross(b - a, c - a)
        if len(set(triangle)) < 3 or key in seen or not np.any(cross) or not np.isfinite(cross).all():
            report["removed_triangles"] += 1
            continue
        seen.add(key)
        clean.append(triangle)
        if index < cap_count:
            report["added_cap_triangles"] += 1
            # A cap triangle must have all three vertices on the same plane.
            distance = np.abs(np.asarray([a, b, c])[:, :, None] - bounds)
            report["cap_max_plane_error"] = max(report["cap_max_plane_error"], float(distance.max(axis=0).min()))
        else:
            report["added_transition_triangles"] += 1
    used = sorted({i for t in clean for i in t if i >= len(vertices)})
    remap = {old: len(vertices) + i for i, old in enumerate(used)}
    if clean:
        mesh.vertices = np.concatenate((vertices, np.asarray([points[i] for i in used]).reshape(-1, 3)))
        new_edges = np.asarray([[remap.get(i, i) for i in t] for t in clean], dtype=np.int64)
        mesh.edges = np.concatenate((mesh.edges, new_edges))
    report["added_vertices"], report["added_triangles"] = len(used), len(clean)
    return finish()
