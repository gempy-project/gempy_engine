"""Phase 1: passing characterizations of legacy behavior, not a new mesh mode."""

from copy import deepcopy
from itertools import combinations

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.modules.dual_contouring.dual_contouring_v2 import compute_dual_contouring_v2
from gempy_engine.modules.dual_contouring.overlapping import average_overlapping_vertices
from gempy_engine.modules.dual_contouring.weighted_qef_setup_multicore import (
    find_and_inject_multi_surface_constraints_multicore,
)
from tests.fixtures.contact_cases import build_contact_case


NAMES = ("planar_erosion", "onlap", "parallel_false_overlap", "three_way_junction")
TOL = 1e-8  # Legacy crossing perturbation and QEF arithmetic, unit cube float64.


@pytest.fixture(autouse=True)
def numpy_extraction(monkeypatch):
    previous = dict(engine_backend=BackendTensor.engine_backend, use_gpu=BackendTensor.use_gpu,
                    use_pykeops=BackendTensor.use_pykeops, dtype=BackendTensor.dtype,
                    grads=BackendTensor.COMPUTE_GRADS)
    pykeops_enabled = BackendTensor.pykeops_enabled
    BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype="float64")
    BackendTensor.pykeops_enabled = False
    monkeypatch.setenv("GEMPY_SKIP_TRIANGULATION", "0")
    monkeypatch.setenv("DUAL_CONTOURING_MULTITHREAD", "False")
    try:
        yield
    finally:
        BackendTensor._change_backend(**previous)
        BackendTensor.pykeops_enabled = pykeops_enabled


def shared_rows(case, i, j):
    a, b = case.active_cells[i], case.active_cells[j]
    # Independent cell lookup, not the production packed-code overlap function.
    lookup = {tuple(cell): row for row, cell in enumerate(b)}
    pairs = [(row, lookup[tuple(cell)]) for row, cell in enumerate(a) if tuple(cell) in lookup]
    return np.asarray(pairs, dtype=int).reshape(-1, 2).T


def boundary_segments(mesh):
    edges = np.concatenate([mesh.edges[:, [0, 1]], mesh.edges[:, [1, 2]], mesh.edges[:, [2, 0]]])
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return mesh.vertices[edges[counts == 1]]


def assert_valid_mesh(mesh):
    assert len(mesh.vertices) and len(mesh.edges)
    assert np.isfinite(mesh.vertices).all()
    assert mesh.edges.dtype.kind in "iu"
    assert mesh.edges.min() >= 0 and mesh.edges.max() < len(mesh.vertices)
    assert len(np.unique(np.sort(mesh.edges, axis=1), axis=0)) == len(mesh.edges)
    triangles = mesh.vertices[mesh.edges]
    assert np.all(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                         triangles[:, 2] - triangles[:, 0]), axis=1) > 1e-12)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("resolution", [6, (10, 8, 6)])
def test_raw_extraction_against_independent_planes(name, resolution):
    case = build_contact_case(name, resolution)
    meshes = compute_dual_contouring_v2(case.dc_data, max_workers=1)
    for i, mesh in enumerate(meshes):
        assert_valid_mesh(mesh)
        np.testing.assert_allclose(mesh.vertices @ case.normals[i], case.levels[i], atol=TOL, rtol=0)
        assert np.all((mesh.vertices >= 0) & (mesh.vertices <= 1))
        # Independent regular-grid oracle for these axis-aligned planes.
        axis = np.flatnonzero(case.normals[i])[0]
        expected_cells = np.prod([n for k, n in enumerate(case.resolution) if k != axis])
        expected_faces = 2 * np.prod([n - 1 for k, n in enumerate(case.resolution) if k != axis])
        assert len(mesh.vertices) == expected_cells
        assert len(mesh.edges) == expected_faces
        for (a, b), line in case.seams.items():
            np.testing.assert_allclose(line @ case.normals[a], case.levels[a], atol=1e-15)
            np.testing.assert_allclose(line @ case.normals[b], case.levels[b], atol=1e-15)
    if case.junction is not None:
        np.testing.assert_allclose(case.normals @ case.junction, case.levels, atol=1e-15)


@pytest.mark.parametrize("name", ["planar_erosion", "onlap"])
def test_legacy_contact_coincidence_does_not_clip_or_create_boundary_seam(name, record_property):
    case = build_contact_case(name)
    meshes = compute_dual_contouring_v2(case.dc_data, max_workers=1)
    original = deepcopy(meshes)
    average_overlapping_vertices(meshes, case.active_cells, case.resolution, case.surface_to_stack)
    rows_a, rows_b = shared_rows(case, 0, 1)
    assert len(rows_a) == 6
    expected = (original[0].vertices[rows_a] + original[1].vertices[rows_b]) / 2
    for mesh, rows in zip(meshes, (rows_a, rows_b)):
        np.testing.assert_allclose(mesh.vertices[rows], expected, atol=1e-15)
    controller, truncated, sign = case.contacts[0]
    assert (controller, truncated, sign) == ((0, 1, -1) if name == "planar_erosion" else (1, 0, 1))
    # Expected sides are explicit geometry: below erosion, above substrate onlap.
    centroids = meshes[truncated].vertices[meshes[truncated].edges].mean(axis=1)
    retained = sign * (centroids[:, 2] - .47)
    # Compact legacy reference: wrong-side faces remain, rather than being clipped.
    assert np.count_nonzero(retained < -TOL) == (30 if name == "planar_erosion" else 20)
    assert np.count_nonzero(retained > TOL) > 0
    seam = np.array([.43, .47])
    residual = np.max(np.abs(expected[:, [0, 2]] - seam), axis=1)
    np.testing.assert_allclose(residual.max(), 2 / 75, atol=TOL, rtol=0)
    record_property("contact_max_coordinate_residual", float(residual.max()))
    record_property("wrong_side_triangle_count", int(np.count_nonzero(retained < -TOL)))
    for i, mesh in enumerate(meshes):
        assert_valid_mesh(mesh)
        np.testing.assert_array_equal(mesh.edges, original[i].edges)
        segments = boundary_segments(mesh)
        on_line = np.all(np.abs(segments[:, :, [0, 2]] - seam) < TOL, axis=(1, 2))
        assert not np.any(on_line)  # No paired boundary segments on the true seam.
        other = np.setdiff1d(np.arange(len(mesh.vertices)), (rows_a, rows_b)[i])
        np.testing.assert_array_equal(mesh.vertices[other], original[i].vertices[other])


def test_parallel_same_cell_false_join_and_same_stack_isolation():
    case = build_contact_case("parallel_false_overlap")
    meshes = compute_dual_contouring_v2(case.dc_data, max_workers=1)
    original = deepcopy(meshes)
    a, b = shared_rows(case, 0, 1)
    assert len(a) == 36
    np.testing.assert_allclose(original[1].vertices[b, 0] - original[0].vertices[a, 0], .03, atol=TOL)
    average_overlapping_vertices(meshes, case.active_cells, case.resolution, case.surface_to_stack)
    np.testing.assert_allclose(meshes[0].vertices[a], meshes[1].vertices[b], atol=1e-15)
    for i, mesh in enumerate(meshes):
        assert_valid_mesh(mesh)
        np.testing.assert_allclose(mesh.vertices[:, 0], .445, atol=TOL)
        np.testing.assert_allclose(np.linalg.norm(mesh.vertices - original[i].vertices, axis=1), .015, atol=TOL)
        np.testing.assert_array_equal(mesh.edges, original[i].edges)
    isolated = deepcopy(original)
    average_overlapping_vertices(isolated, case.active_cells, case.resolution, (0, 0))
    for actual, before in zip(isolated, original):
        np.testing.assert_array_equal(actual.vertices, before.vertices)
        np.testing.assert_array_equal(actual.edges, before.edges)


@pytest.mark.parametrize("same_stack,absent_fault_matrix", [(False, False), (True, False), (True, True)])
def test_inherited_qef_distorts_parallel_planes_before_averaging(same_stack, absent_fault_matrix, record_property):
    case = build_contact_case("parallel_false_overlap")
    raw = compute_dual_contouring_v2(case.dc_data, max_workers=1)
    find_and_inject_multi_surface_constraints_multicore(
        case.dc_data, case.active_cells, case.resolution, max_workers=1,
        surface_to_stack=(0, 0) if same_stack else case.surface_to_stack,
        faults_relations=None if absent_fault_matrix else np.zeros((2, 2), dtype=bool),
    )
    weighted = compute_dual_contouring_v2(case.dc_data, max_workers=1)
    for i, (before, after) in enumerate(zip(raw, weighted)):
        assert case.dc_data[i].extra_weights is not None
        assert np.count_nonzero(case.dc_data[i].extra_weights) == 36 * 4
        assert_valid_mesh(after)
        displacement = np.max(np.abs(after.vertices[:, 0] - before.vertices[:, 0]))
        record_property(f"surface_{i}_qef_max_displacement", float(displacement))
        # Four own rows plus one mass-point bias row; four partner rows at weight 10.
        np.testing.assert_allclose(displacement, .03 * 40 / 45, atol=TOL, rtol=0)
        np.testing.assert_allclose(after.vertices[:, 1:], before.vertices[:, 1:], atol=TOL, rtol=0)
        np.testing.assert_array_equal(after.edges, before.edges)
    # The same-stack variant intentionally captures empty-set -> None filtering.


def test_three_way_pairwise_averaging_is_inconsistent_and_order_dependent(record_property):
    case = build_contact_case("three_way_junction")
    raw = compute_dual_contouring_v2(case.dc_data, max_workers=1)
    junction_cell = tuple(np.floor(np.array([.43, .46, .47]) * 6).astype(int))
    rows = [next(i for i, cell in enumerate(cells) if tuple(cell) == junction_cell)
            for cells in case.active_cells]
    initial = np.array([mesh.vertices[row] for mesh, row in zip(raw, rows)])
    meshes = deepcopy(raw)
    average_overlapping_vertices(meshes, case.active_cells, case.resolution, case.surface_to_stack)
    actual = np.array([mesh.vertices[row] for mesh, row in zip(meshes, rows)])
    ab = (initial[0] + initial[1]) / 2
    ac = (ab + initial[2]) / 2
    bc = (ab + ac) / 2
    np.testing.assert_allclose(actual, [ac, bc, bc], atol=1e-15)
    spread = max(np.linalg.norm(actual[a] - actual[b]) for a, b in combinations(range(3), 2))
    assert spread > .01
    record_property("junction_vertex_spread", float(spread))
    reordered = deepcopy(raw[::-1])
    average_overlapping_vertices(reordered, case.active_cells[::-1], case.resolution, case.surface_to_stack[::-1])
    reverse = np.array([mesh.vertices[row] for mesh, row in zip(reordered[::-1], rows)])
    order_delta = np.max(np.abs(reverse - actual))
    np.testing.assert_allclose(order_delta, 1 / 150, atol=TOL, rtol=0)
    record_property("junction_order_max_coordinate_delta", float(order_delta))
    assert np.max(np.linalg.norm(actual - np.array([.43, .46, .47]), axis=1)) > .01
    for before, after in zip(raw, meshes):
        np.testing.assert_array_equal(after.edges, before.edges)


def test_resolved_parallel_planes_have_no_overlap_or_movement():
    case = build_contact_case("parallel_false_overlap", resolution=40)
    meshes = compute_dual_contouring_v2(case.dc_data, max_workers=1)
    before = deepcopy(meshes)
    assert len(shared_rows(case, 0, 1)[0]) == 0
    average_overlapping_vertices(meshes, case.active_cells, case.resolution, case.surface_to_stack)
    for mesh, original in zip(meshes, before):
        np.testing.assert_array_equal(mesh.vertices, original.vertices)
        np.testing.assert_array_equal(mesh.edges, original.edges)


def test_factory_fresh_inputs_and_invalid_arguments():
    first = build_contact_case("planar_erosion")
    second = build_contact_case("planar_erosion")
    first.dc_data[0].xyz_on_edge[:] = -1
    assert np.all(second.dc_data[0].xyz_on_edge >= 0)
    with pytest.raises(ValueError, match="Unknown contact case"):
        build_contact_case("unknown")
    for resolution in (0, 2.5, (2, 3), (2, -1, 3)):
        with pytest.raises(ValueError, match="resolution"):
            build_contact_case("onlap", resolution)
