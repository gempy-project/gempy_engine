"""Analytic interior DC meshes, independent of the capping triangulator."""

from itertools import product
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from gempy_engine.modules.dual_contouring._extent_capping import boundary_lattice, cap_mesh
from gempy_engine.modules.dual_contouring import _extent_capping


def plane_mesh(shape, extent, normal, level):
    bounds = np.asarray(extent).reshape(3, 2)
    shape = np.asarray(shape)
    normal = np.asarray(normal)
    step = (bounds[:, 1] - bounds[:, 0]) / shape
    positions = lambda c: bounds[:, 0] + np.asarray(c) * step
    scalar = lambda c: positions(c) @ normal - level
    vertices, cells, lookup = [], [], {}
    for cell in product(*(range(n) for n in shape)):
        intersections = []
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            for offsets in product((0, 1), repeat=2):
                a = np.array(cell)
                a[others] += offsets
                b = a.copy()
                b[axis] += 1
                sa, sb = scalar(a), scalar(b)
                if (sa <= 0) != (sb <= 0):
                    intersections.append(positions(a) + sa / (sa - sb) * (positions(b) - positions(a)))
        if intersections:
            lookup[cell] = len(vertices)
            cells.append(cell)
            vertices.append(np.mean(intersections, axis=0))
    triangles = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        ranges = [range(n + 1) for n in shape]
        ranges[axis] = range(shape[axis])
        for corner in product(*ranges):
            a = np.array(corner)
            b = a.copy()
            b[axis] += 1
            if (scalar(a) <= 0) == (scalar(b) <= 0):
                continue
            ring = []
            for du, dv in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
                c = a.copy()
                c[u] += du
                c[v] += dv
                if tuple(c) in lookup:
                    ring.append(lookup[tuple(c)])
            if len(ring) != 4:
                continue
            for triangle in ((ring[0], ring[1], ring[2]), (ring[0], ring[2], ring[3])):
                p, q, r = [vertices[i] for i in triangle]
                if np.dot(np.cross(q - p, r - p), normal) < 0:
                    triangle = triangle[::-1]
                triangles.append(triangle)
    return SimpleNamespace(vertices=np.asarray(vertices).reshape(-1, 3),
                           edges=np.asarray(triangles, dtype=int).reshape(-1, 3)), np.asarray(cells).reshape(-1, 3)


def assert_closed(mesh):
    report = mesh.capping_report
    assert report["watertight"], report
    assert report["closure_success"], report
    assert report["added_triangles"] == report["added_cap_triangles"] + report["added_transition_triangles"]
    assert report["cap_max_plane_error"] == 0
    assert not report["warnings"], report
    assert len({tuple(sorted(t)) for t in mesh.edges}) == len(mesh.edges)
    triangles = mesh.vertices[mesh.edges]
    assert np.all(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1) > 0)


def test_boundary_lattice_shared_points_and_outward_faces():
    shape, extent = (2, 3, 4), (-7, 2, 10, 16, -2, 6)
    coordinates, positions, metadata = boundary_lattice(shape, extent)
    assert coordinates.dtype.kind == "i"
    assert len(coordinates) == 3 * 4 * 5 - 1 * 2 * 3
    assert len(np.unique(coordinates, axis=0)) == len(coordinates)
    assert metadata["triangles"].shape == (4 * (2 * 3 + 3 * 4 + 4 * 2), 3)
    center = np.asarray(extent).reshape(3, 2).mean(axis=1)
    for triangle in positions[metadata["triangles"]]:
        a, b, c = triangle
        assert np.dot(np.cross(b - a, c - a), triangle.mean(axis=0) - center) > 0
    np.testing.assert_array_equal(positions.min(axis=0), np.asarray(extent)[::2])
    np.testing.assert_array_equal(positions.max(axis=0), np.asarray(extent)[1::2])


@pytest.mark.parametrize("normal,level", [((1, .37, -.21), .731), ((1, 0, 0), .43),
                                        ((1, 1, 1), .71), ((-.3, 1, .7), .52)])
def test_plane_closed_and_original_indices_preserved(normal, level):
    shape, extent = (4, 5, 3), (0, 1, 0, 1, 0, 1)
    mesh, cells = plane_mesh(shape, extent, normal, level)
    original_vertices, original_triangles = mesh.vertices.copy(), mesh.edges.copy()
    # Neither the QEF rows nor boundary sample order is assumed sorted.
    permutation = np.arange(len(cells))[::-1]
    inverse = np.argsort(permutation)
    mesh.vertices = mesh.vertices[permutation]
    mesh.edges = inverse[mesh.edges]
    coordinates, positions, _ = boundary_lattice(shape, extent)
    assert cap_mesh(mesh, cells[permutation], shape, extent, coordinates[::-1],
                    (positions @ normal)[::-1], level) is mesh
    assert_closed(mesh)
    assert mesh.capping_report["open_edges_before"] > 0
    assert mesh.capping_report["non_extent_open_edges_before"] == 0
    assert mesh.capping_report["open_edges_after"] == 0
    assert mesh.capping_report["non_extent_open_edges_after"] == 0
    assert mesh.capping_report["added_cap_triangles"] > 0
    assert mesh.capping_report["added_transition_triangles"] > 0
    np.testing.assert_array_equal(mesh.vertices[:len(cells)], original_vertices[permutation])
    np.testing.assert_array_equal(mesh.edges[:len(original_triangles)], inverse[original_triangles])
    new_triangles = mesh.edges[len(original_triangles):]
    assert np.count_nonzero(np.all(new_triangles >= len(cells), axis=1)) == mesh.capping_report["added_cap_triangles"]
    assert np.count_nonzero(np.any(new_triangles < len(cells), axis=1)) == mesh.capping_report["added_transition_triangles"]
    assert np.all(mesh.vertices >= 0) and np.all(mesh.vertices <= 1)
    # Shared box edges have one vertex per position, not one per face.
    new = mesh.vertices[len(cells):]
    assert len(np.unique(new, axis=0)) == len(new)
    t = mesh.vertices[mesh.edges]
    assert np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))) > 0


def test_exact_extent_nonunit_plane():
    shape, extent = (3, 4, 5), (-2, 4, 10, 14, -8, -3)
    normal, level = (1, .31, .17), 3.19
    mesh, cells = plane_mesh(shape, extent, normal, level)
    coordinates, positions, _ = boundary_lattice(shape, extent)
    cap_mesh(mesh, cells, shape, extent, coordinates, positions @ normal, level)
    assert_closed(mesh)
    np.testing.assert_array_equal(mesh.vertices.min(axis=0), [-2, 10, -8])
    np.testing.assert_array_equal(mesh.vertices.max(axis=0)[1:], [14, -3])


@pytest.mark.parametrize("level", [1., 1.5])
def test_corner_ties_are_shared_and_limitations_reported(level):
    shape, extent = (4, 4, 4), (0, 1, 0, 1, 0, 1)
    mesh, cells = plane_mesh(shape, extent, (1, 1, 1), level)
    coordinates, positions, _ = boundary_lattice(shape, extent)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        cap_mesh(mesh, cells, shape, extent, coordinates, positions.sum(axis=1), level)
    new = mesh.vertices[len(cells):]
    assert len(np.unique(new, axis=0)) == len(new)
    assert mesh.capping_report["added_triangles"] > 0
    assert mesh.capping_report["watertight"] or (emitted and mesh.capping_report["warnings"])


@pytest.mark.parametrize("inside", [False, True])
def test_no_boundary_contour_adds_no_shell(inside):
    # An enclosed sphere represented by an outward-oriented octahedron.
    vertices = np.vstack((np.eye(3), -np.eye(3))) * .2 + .5
    triangles = []
    for x, y, z in product((0, 3), (1, 4), (2, 5)):
        a, b, c = vertices[[x, y, z]]
        t = [x, y, z]
        if np.dot(np.cross(b - a, c - a), a - .5) < 0:
            t.reverse()
        triangles.append(t)
    mesh = SimpleNamespace(vertices=vertices.copy(), edges=np.asarray(triangles))
    coordinates, positions, _ = boundary_lattice((10, 10, 10), (0, 1, 0, 1, 0, 1))
    values = np.linalg.norm(positions - .5, axis=1) - .2
    cap_mesh(mesh, np.floor(vertices * 10).astype(int), (10, 10, 10), (0, 1, 0, 1, 0, 1),
             coordinates, -values if inside else values, 0)
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.edges, triangles)
    assert mesh.capping_report["added_triangles"] == 0
    assert mesh.capping_report["contour_edges"] == 0
    assert_closed(mesh)


def test_missing_escaped_qef_and_internal_hole_are_not_repaired():
    shape, extent = (5, 5, 5), (0, 1, 0, 1, 0, 1)
    mesh, cells = plane_mesh(shape, extent, (1, .37, -.21), .73)
    mesh.vertices[0] = [-2, -2, -2]
    # Remove an interior triangle, not a boundary feature.
    mesh.edges = mesh.edges[1:].copy()
    original = mesh.edges.copy()
    coordinates, positions, _ = boundary_lattice(shape, extent)
    with pytest.warns(RuntimeWarning, match="best effort"):
        cap_mesh(mesh, cells, shape, extent, coordinates, positions @ (1, .37, -.21), .73)
    report = mesh.capping_report
    assert report["escaped_qef"] == 1
    assert report["missing_qef"] > 0
    assert report["boundary_edges"] > 0
    assert not report["watertight"]
    np.testing.assert_array_equal(mesh.edges[:len(original)], original)
    assert all(np.any(t >= len(cells)) for t in mesh.edges[len(original):])


def test_no_available_qef_discards_all_cap_components():
    mesh = SimpleNamespace(vertices=np.empty((0, 3)), edges=np.empty((0, 3), dtype=int))
    coordinates, positions, _ = boundary_lattice((2, 2, 2), (0, 1, 0, 1, 0, 1))
    with pytest.warns(RuntimeWarning, match="missing_qef"):
        cap_mesh(mesh, np.empty((0, 3)), (2, 2, 2), (0, 1, 0, 1, 0, 1),
                 coordinates, positions[:, 0], .3)
    assert mesh.capping_report["discarded_components"] > 0
    assert mesh.vertices.shape == (0, 3)
    assert mesh.edges.shape == (0, 3)


def test_nonfinite_boundary_is_diagnostic_not_geometry():
    mesh = SimpleNamespace(vertices=np.empty((0, 3)), edges=np.empty((0, 3), dtype=int))
    coordinates, positions, _ = boundary_lattice((1, 1, 1), (0, 1, 0, 1, 0, 1))
    values = positions[:, 0].copy()
    values[0] = np.nan
    with pytest.warns(RuntimeWarning, match="Nonfinite"):
        cap_mesh(mesh, np.empty((0, 3)), (1, 1, 1), (0, 1, 0, 1, 0, 1), coordinates, values, .3)
    assert mesh.capping_report["unsupported_topology"] == 1
    assert mesh.capping_report["added_triangles"] == 0


def test_only_component_connected_to_available_qef_is_retained():
    shape, extent = (2, 2, 2), (0, 1, 0, 1, 0, 1)
    coordinates, positions, _ = boundary_lattice(shape, extent)
    values = np.ones(len(coordinates))
    values[np.all(coordinates == 0, axis=1) | np.all(coordinates == 2, axis=1)] = -1
    mesh = SimpleNamespace(vertices=np.asarray([[.1, .1, .1]]),
                           edges=np.empty((0, 3), dtype=int))
    with pytest.warns(RuntimeWarning, match="missing_qef"):
        cap_mesh(mesh, np.asarray([[0, 0, 0]]), shape, extent, coordinates, values, 0)
    assert mesh.capping_report["discarded_components"] == 1
    assert mesh.capping_report["watertight"]
    assert not mesh.capping_report["closure_success"]
    assert np.all(mesh.vertices <= .25)


def test_internal_hole_without_missing_qef_is_preserved():
    shape, extent = (6, 6, 6), (0, 1, 0, 1, 0, 1)
    mesh, cells = plane_mesh(shape, extent, (1, .37, -.21), .731)
    interior = np.flatnonzero(np.all((cells[mesh.edges] > 0) & (cells[mesh.edges] < 5), axis=(1, 2)))
    assert len(interior)
    removed = mesh.edges[interior[0]].copy()
    mesh.edges = np.delete(mesh.edges, interior[0], axis=0)
    coordinates, positions, _ = boundary_lattice(shape, extent)
    with pytest.warns(RuntimeWarning, match="boundary_edges: 3"):
        cap_mesh(mesh, cells, shape, extent, coordinates, positions @ (1, .37, -.21), .731)
    assert mesh.capping_report["boundary_edges"] == 3
    assert mesh.capping_report["missing_qef"] == 0
    assert mesh.capping_report["escaped_qef"] == 0
    assert mesh.capping_report["non_extent_open_edges_before"] == 3
    assert mesh.capping_report["non_extent_open_edges_after"] == 3
    assert not mesh.capping_report["closure_success"]
    assert tuple(sorted(removed)) not in {tuple(sorted(t)) for t in mesh.edges}


def test_prebuilt_geometry_and_single_batched_strict_crossing(monkeypatch):
    shape, extent = (3, 4, 5), (0, 1, 0, 1, 0, 1)
    geometry = boundary_lattice(shape, extent)
    coordinates, positions, metadata = geometry
    original_coordinates, original_positions = coordinates.copy(), positions.copy()
    original_triangles, original_cells = metadata["triangles"].copy(), metadata["cells"].copy()
    mesh, cells = plane_mesh(shape, extent, (1, 0, 0), .43)
    # Every QEF on this plane is inset from the actual extent planes, including
    # the rim. Its owning cell, not its position, identifies an extent opening.
    assert np.all((mesh.vertices > 0) & (mesh.vertices < 1))
    original_crossing = _extent_capping.scalar_crossing_parameters
    calls = []

    def crossing(start, end, iso, *, xp=np):
        calls.append(start.shape)
        return original_crossing(start, end, iso, xp=xp)

    def unexpected_lattice(*args):
        pytest.fail("prebuilt boundary geometry must not be regenerated")

    monkeypatch.setattr(_extent_capping, "boundary_lattice", unexpected_lattice)
    monkeypatch.setattr(_extent_capping, "scalar_crossing_parameters", crossing)
    # Tiny but nonzero scalar differences must not get tolerance-snapped.
    values = (positions[:, 0] - .43) * 1e-200
    cap_mesh(mesh, cells, shape, extent, coordinates, values, 0, boundary_geometry=geometry)
    assert calls == [metadata["triangles"].shape]
    assert_closed(mesh)
    assert mesh.capping_report["open_edges_before"] > 0
    assert mesh.capping_report["non_extent_open_edges_before"] == 0
    new = mesh.vertices[len(cells):]
    assert np.any(np.isclose(new[:, 0], .43, atol=1e-15))
    np.testing.assert_array_equal(coordinates, original_coordinates)
    np.testing.assert_array_equal(positions, original_positions)
    np.testing.assert_array_equal(metadata["triangles"], original_triangles)
    np.testing.assert_array_equal(metadata["cells"], original_cells)


def test_crossing_arithmetic_overflow_is_reported():
    shape, extent = (1, 1, 1), (0, 1, 0, 1, 0, 1)
    coordinates, positions, _ = boundary_lattice(shape, extent)
    values = np.where(positions[:, 0] == 0, -1e308, 1e308)
    mesh = SimpleNamespace(vertices=np.empty((0, 3)), edges=np.empty((0, 3), dtype=int))
    with pytest.warns(RuntimeWarning, match="Nonfinite scalar crossing interpolation arithmetic"):
        cap_mesh(mesh, np.empty((0, 3)), shape, extent, coordinates, values, 0)
    assert mesh.capping_report["unsupported_topology"] == 1
    assert not mesh.capping_report["closure_success"]
    assert mesh.capping_report["added_triangles"] == 0


@pytest.mark.parametrize("defect,field", [
    ("nonfinite", "nonfinite_vertices"),
    ("negative_index", "invalid_triangles"),
    ("out_of_range", "invalid_triangles"),
    ("fractional_index", "invalid_triangles"),
    ("nonfinite_index", "invalid_triangles"),
    ("degenerate", "degenerate_triangles"),
    ("escaped", "escaped_qef"),
    ("unsupported", "unsupported_topology"),
])
def test_original_geometry_defects_never_report_success(defect, field):
    shape, extent = (3, 3, 3), (0, 1, 0, 1, 0, 1)
    vertices = np.asarray([[.2, .2, .2], [.7, .2, .2], [.2, .7, .2], [.2, .2, .7]])
    cells = np.floor(vertices * 3).astype(int)
    triangles = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    if defect == "nonfinite":
        vertices[0, 0] = np.nan
    elif defect == "negative_index":
        triangles[0, 0] = -1
    elif defect == "out_of_range":
        triangles[0, 0] = len(vertices)
    elif defect in ("fractional_index", "nonfinite_index"):
        triangles = triangles.astype(float)
        triangles[0, 0] = .5 if defect == "fractional_index" else np.nan
    elif defect == "degenerate":
        vertices[0] = vertices[1]
    elif defect == "escaped":
        vertices[0, 0] = -.1
    elif defect == "unsupported":
        cells[0] = cells[1]
    mesh = SimpleNamespace(vertices=vertices.copy(), edges=triangles.copy())
    coordinates, positions, _ = boundary_lattice(shape, extent)
    with pytest.warns(RuntimeWarning, match=field):
        cap_mesh(mesh, cells, shape, extent, coordinates, np.ones(len(positions)), 0)
    report = mesh.capping_report
    assert report[field] > 0
    assert not report["closure_success"]
    assert report["added_triangles"] == 0
    np.testing.assert_array_equal(mesh.vertices, vertices)
    np.testing.assert_array_equal(mesh.edges, triangles)
    if defect in ("nonfinite", "degenerate", "escaped", "unsupported"):
        assert report["watertight"]  # Topological closure alone is insufficient.
