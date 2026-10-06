"""Array-only reconciliation of finite, planar triangle patches.

No mesh state is mutated here. Callers own pass ordering and plane inference.
"""

import numpy as np


def _validate_patch(vertices, faces, plane, tolerance):
    vertices = np.array(vertices, dtype=float, copy=True)
    faces_array = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Vertices must have shape (n, 3).")
    if faces_array.ndim != 2 or faces_array.shape[1] != 3:
        raise ValueError("Faces must have shape (m, 3).")
    if not np.issubdtype(faces_array.dtype, np.integer):
        raise ValueError("Face indices must be integers.")
    faces = np.array(faces_array, dtype=np.int64, copy=True)
    normal = np.asarray(plane[0], dtype=float)
    offset = float(plane[1])
    if normal.shape != (3,) or not np.all(np.isfinite(normal)) or not np.isfinite(offset):
        raise ValueError("A plane must contain a finite unit normal and offset.")
    normal_length = np.linalg.norm(normal)
    if not np.isclose(normal_length, 1.0, rtol=0, atol=1e-8):
        raise ValueError("Plane normals must be normalized.")
    normal, offset = normal / normal_length, offset / normal_length
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Vertices must be finite.")
    coordinate_scale = max(float(np.max(np.abs(vertices), initial=0)), abs(offset))
    if np.spacing(coordinate_scale) > tolerance:
        raise ValueError("Tolerance is below representable coordinate precision (float64 spacing).")
    if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("Face indices are out of bounds.")
    if len(vertices) and np.any(np.abs(vertices @ normal - offset) > tolerance):
        raise ValueError("Nonplanar patch: vertices do not lie on the supplied plane.")
    if len(faces):
        triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                        triangles[:, 2] - triangles[:, 0]), axis=1)
        if np.any(areas == 0) or not np.all(np.isfinite(areas)):
            raise ValueError("Input triangles must be nondegenerate.")
    return vertices, faces, normal, offset


def _clip_polygon(polygon, normal, offset, sign, tolerance):
    residuals = sign * (polygon @ normal - offset)
    residuals[np.abs(residuals) <= tolerance] = 0.0
    result = []
    for i, point in enumerate(polygon):
        following = polygon[(i + 1) % len(polygon)]
        a, b = residuals[i], residuals[(i + 1) % len(polygon)]
        if a >= 0:
            result.append(point)
        if (a < 0 < b) or (b < 0 < a):
            result.append(point + (following - point) * (a / (a - b)))
    return np.asarray(result, dtype=float).reshape(-1, 3)


def _line_intervals(vertices, faces, normal, offset, origin, direction,
                    tolerance):
    intervals = []
    for face in faces:
        triangle = vertices[face]
        residuals = triangle @ normal - offset
        residuals[np.abs(residuals) <= tolerance] = 0.0
        if np.all(residuals == 0):
            raise ValueError("Unsupported support: a triangle is indistinguishable from the seam at this tolerance.")
        points = list(triangle[residuals == 0])
        for i in range(3):
            j = (i + 1) % 3
            if residuals[i] * residuals[j] < 0:
                points.append(triangle[i] + (triangle[j] - triangle[i]) *
                              (residuals[i] / (residuals[i] - residuals[j])))
        if len(points) < 2:
            intervals.append(None)
            continue
        parameters = (np.asarray(points) - origin) @ direction
        low, high = float(parameters.min()), float(parameters.max())
        intervals.append((low, high) if high - low > tolerance else None)
    return intervals


def _merge_intervals(intervals, tolerance):
    merged = []
    for low, high in sorted(interval for interval in intervals if interval is not None):
        if merged and low <= merged[-1][1] + tolerance:
            merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
        else:
            merged.append((low, high))
    return merged


def _build_patch(vertices, faces, polygons, origin, direction, knots, tolerance):
    """Insert line knots into polygon boundaries before triangulation.

    Center fans keep every collinear boundary knot as a real edge endpoint.
    They also propagate new edge points into unsplit incident triangles.
    """
    result_vertices = list(vertices.copy())
    result_faces = []
    result_normals = []
    knot_points = origin + knots[:, None] * direction
    knot_indices = {}
    edge_indices = {}

    def vertex_index(point):
        if len(knots):
            distances = np.linalg.norm(knot_points - point, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= tolerance:
                if nearest not in knot_indices:
                    matches = np.flatnonzero(np.linalg.norm(vertices - knot_points[nearest], axis=1)
                                             <= tolerance)
                    index = int(matches[0]) if len(matches) else len(result_vertices)
                    if index == len(result_vertices):
                        result_vertices.append(knot_points[nearest].copy())
                    else:
                        result_vertices[index] = knot_points[nearest].copy()
                    knot_indices[nearest] = index
                return knot_indices[nearest]
        # Non-seam vertices are original polygon corners, not global weld candidates.
        matches = np.flatnonzero(np.all(vertices == point, axis=1))
        if len(matches):
            return int(matches[0])
        key = tuple(point)
        if key not in edge_indices:
            edge_indices[key] = len(result_vertices)
            result_vertices.append(point.copy())
        return edge_indices[key]

    for face, pieces in zip(faces, polygons):
        for polygon in pieces:
            if len(polygon) < 3:
                continue
            boundary = []
            for i, a in enumerate(polygon):
                b = polygon[(i + 1) % len(polygon)]
                edge = b - a
                length = np.linalg.norm(edge)
                if length == 0:
                    continue
                boundary.append(vertex_index(a))
                if len(knots):
                    along = (knot_points - a) @ (edge / length)
                    distance = np.linalg.norm(knot_points - a - along[:, None] * (edge / length), axis=1)
                    interior = np.flatnonzero((distance <= tolerance) &
                                              (along > tolerance) & (along < length - tolerance))
                    for k in interior[np.argsort(along[interior])]:
                        boundary.append(vertex_index(knot_points[k]))
            boundary = [index for i, index in enumerate(boundary)
                        if index != boundary[i - 1]]
            if len(set(boundary)) < 3:
                continue
            first_face = len(result_faces)
            if len(boundary) == 3:
                # Preserve the original indices/order when the triangle is untouched.
                if np.array_equal(polygon, vertices[face]) and not any(
                        index in knot_indices.values() for index in boundary):
                    result_faces.append(tuple(face))
                else:
                    result_faces.append(tuple(boundary))
            else:
                center = np.mean(np.asarray(result_vertices)[boundary], axis=0)
                center_index = len(result_vertices)
                result_vertices.append(center)
                result_faces.extend((boundary[i], boundary[(i + 1) % len(boundary)], center_index)
                                    for i in range(len(boundary)))
            triangle = vertices[face]
            source_normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            result_normals.extend([source_normal] * (len(result_faces) - first_face))
    result_vertices = np.asarray(result_vertices, dtype=float).reshape(-1, 3)
    result_faces = np.asarray(result_faces, dtype=np.int64).reshape(-1, 3)
    if len(result_faces):
        triangles = result_vertices[result_faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        orientation = np.einsum("ij,ij->i", normals, np.asarray(result_normals))
        if np.any(orientation <= 0) or not np.all(np.isfinite(orientation)):
            raise ValueError("Unsupported support: reconciliation produced degenerate or inverted triangles.")
    if not np.all(np.isfinite(result_vertices)):
        raise ValueError("Reconciliation produced nonfinite vertices.")
    return result_vertices, result_faces


def reconcile_planar_contact(controller_vertices, controller_faces,
                             truncated_vertices, truncated_faces,
                             controller_plane, truncated_plane,
                             retained_sign, tolerance=1e-8):
    """Clip a planar target and create matching finite seam EDGE connectivity.

    Plane tuples are ``(unit_normal, offset)`` with ``xyz @ normal = offset``.
    The target retains ``retained_sign * (xyz @ controller_normal - offset) >= 0``.
    Returns copied/new controller vertices/faces, target vertices/faces, and a
    JSON-native report. Input arrays are never mutated. Unused vertices remain.

    Ownership removes discarded target area even without a shared seam. Target
    seams with retained area require full finite controller support, including
    when the patches are disjoint. Distinct parallel planes never join.
    Coincident planes and insufficiently supported seams raise ValueError.
    Controller boundary seams are valid and reported separately. This assumes
    ordinary nonoverlapping triangle patches; it does not repair input topology.
    Tolerance is an absolute spatial tolerance, not a mesh simplification scale.
    Tolerances below float64 coordinate spacing are explicitly unsupported.
    """
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive.")
    if retained_sign not in (-1, 1):
        raise ValueError("retained_sign must be +1 or -1.")
    cv, cf, cn, co = _validate_patch(controller_vertices, controller_faces,
                                     controller_plane, tolerance)
    tv, tf, tn, to = _validate_patch(truncated_vertices, truncated_faces,
                                     truncated_plane, tolerance)
    report = {"status": "no_contact", "seam_segment_count": 0,
              "controller_boundary_seam_segment_count": 0,
              "seam_vertex_count": 0, "controller_faces_before": int(len(cf)),
              "controller_faces_after": int(len(cf)),
              "truncated_faces_before": int(len(tf)),
              "truncated_faces_after": int(len(tf))}
    cross = np.cross(cn, tn)
    sine = np.linalg.norm(cross)
    if sine <= 64 * np.finfo(float).eps:
        separation = abs(to - np.dot(cn, tn) * co)
        if separation <= tolerance:
            raise ValueError("Coincident planes are unsupported.")
        if retained_sign * (np.dot(cn, tn) * to - co) < -tolerance:
            tf = np.empty((0, 3), dtype=np.int64)
        report["truncated_faces_after"] = int(len(tf))
        report["reason"] = "parallel_planes"
        return cv, cf, tv, tf, report
    if not len(tf):
        report["reason"] = "empty_patch"
        return cv, cf, tv, tf, report

    target_residuals = retained_sign * (tv[tf] @ cn - co)
    retained_area = np.any(target_residuals > tolerance, axis=1)
    if np.any(np.all(np.abs(target_residuals) <= tolerance, axis=1)):
        raise ValueError("Unsupported support: a triangle is indistinguishable from the seam at this tolerance.")
    if not np.any(retained_area):
        report.update(reason="target_fully_discarded", truncated_faces_after=0)
        return cv, cf, tv, np.empty((0, 3), dtype=np.int64), report

    direction = cross / sine
    # Anchor near the mesh to avoid unnecessarily large line parameters.
    anchor = cv[cf[0, 0]] if len(cf) else tv[tf[0, 0]]
    normals = np.stack((cn, tn))
    origin = anchor + np.linalg.lstsq(normals, np.array([co, to]) - normals @ anchor,
                                       rcond=None)[0]
    controller_intervals = _line_intervals(cv, cf, tn, to, origin, direction, tolerance)
    geometric_target_intervals = _line_intervals(tv, tf, cn, co, origin, direction, tolerance)
    target_intervals = [interval if retained else None
                        for interval, retained in zip(geometric_target_intervals, retained_area)]
    controller_support = _merge_intervals(controller_intervals, tolerance)
    target_support = _merge_intervals(target_intervals, tolerance)
    shared = _merge_intervals([(max(a, c), min(b, d))
                              for a, b in controller_support for c, d in target_support
                              if min(b, d) - max(a, c) > tolerance], tolerance)
    for low, high in target_support:
        if not any(a <= low + tolerance and b >= high - tolerance
                   for a, b in controller_support):
            raise ValueError("Insufficient controller support for the truncated seam.")
    if not shared:
        if np.any(retained_area & np.any(target_residuals < -tolerance, axis=1)):
            raise ValueError("Insufficient support: truncated seam is below the spatial tolerance.")
        # Separate retained/discarded components need ownership, but no joining.
        tf = tf[retained_area].copy()
        report.update(reason="no_retained_seam" if len(cf) else "empty_patch",
                      truncated_faces_after=int(len(tf)))
        return cv, cf, tv, tf, report

    controller_polygons = []
    parameters = [value for interval in target_intervals if interval is not None
                  for value in interval]
    for face, interval in zip(cf, controller_intervals):
        triangle = cv[face]
        intersects = interval is not None and any(
            min(interval[1], high) - max(interval[0], low) > tolerance
            for low, high in shared)
        if intersects:
            parameters.extend(interval)
            controller_polygons.append([_clip_polygon(triangle, tn, to, sign, tolerance)
                                        for sign in (1, -1)])
        else:
            controller_polygons.append([triangle])
    target_polygons = [[_clip_polygon(tv[face], cn, co, retained_sign, tolerance)] for face in tf]
    knots = []
    for parameter in sorted(parameters):
        if not knots or parameter - knots[-1] > tolerance:
            knots.append(parameter)
    knots = np.asarray(knots)
    new_cv, new_cf = _build_patch(cv, cf, controller_polygons, origin, direction, knots, tolerance)
    new_tv, new_tf = _build_patch(tv, tf, target_polygons, origin, direction, knots, tolerance)

    # Verify actual seam edges, not just matching point coordinates.
    seam_edges = []
    for vertices, faces in ((new_cv, new_cf), (new_tv, new_tf)):
        edges = {}
        for face in faces:
            for i in range(3):
                points = vertices[face[[i, (i + 1) % 3]]]
                line_parameters = (points - origin) @ direction
                if np.any(np.linalg.norm(points - origin - line_parameters[:, None] * direction,
                                         axis=1) > tolerance):
                    continue
                indices = [int(np.argmin(np.abs(knots - p))) for p in line_parameters]
                if all(abs(knots[k] - p) <= tolerance for k, p in zip(indices, line_parameters)):
                    edge = tuple(sorted(indices))
                    edges[edge] = edges.get(edge, 0) + 1
        seam_edges.append(edges)
    expected = {(i, i + 1) for i in range(len(knots) - 1)
                if any(low - tolerance <= knots[i] and knots[i + 1] <= high + tolerance
                       for low, high in shared)}
    if not expected or any(not expected.issubset(edges) for edges in seam_edges):
        raise ValueError("Insufficient support: shared seam edges could not be constructed.")
    if any(b - a > 1 for edges in seam_edges for a, b in edges):
        raise ValueError("Unsupported support: seam contains a T junction.")
    if any(count > 2 for count in seam_edges[0].values()):
        raise ValueError("Nonmanifold controller seam incidence exceeds two faces.")
    if any(count != 1 for count in seam_edges[1].values()):
        raise ValueError("Invalid target seam incidence: each edge must have exactly one face.")
    report.update(status="reconciled", seam_segment_count=int(len(expected)),
                  controller_boundary_seam_segment_count=int(sum(seam_edges[0][edge] == 1
                                                                for edge in expected)),
                  seam_vertex_count=int(len({k for edge in expected for k in edge})),
                  controller_faces_after=int(len(new_cf)),
                  truncated_faces_after=int(len(new_tf)))
    return new_cv, new_cf, new_tv, new_tf, report
