from itertools import product

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import InterpolationOptions
from gempy_engine.core.data.options.evaluation_options import TriangulationMethod
from gempy_engine.modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge
from gempy_engine.modules.dual_contouring.fancy_triangulation import triangulate
from gempy_engine.modules.dual_contouring.quad_triangulation import triangulate_quads


@pytest.fixture(params=['numpy', 'PYTORCH'])
def backend(request):
    if request.param == 'PYTORCH':
        pytest.importorskip('torch')
    old = (BackendTensor.engine_backend, BackendTensor.use_gpu, BackendTensor.dtype,
           BackendTensor.use_pykeops, BackendTensor.COMPUTE_GRADS, BackendTensor.pykeops_enabled)
    BackendTensor._change_backend(AvailableBackends[request.param], use_gpu=False,
                                 dtype='float64', use_pykeops=False, grads=True)
    BackendTensor.pykeops_enabled = False
    yield request.param
    BackendTensor._change_backend(old[0], use_gpu=old[1], dtype=old[2], use_pykeops=old[3], grads=old[4])
    BackendTensor.pykeops_enabled = old[5]


def plane_data(coords, axis=0, sign=1, iso=1.4):
    t = BackendTensor.t
    coords = t.array(coords, dtype='int64')
    corners = coords[:, None, :] + t.array(list(product((0, 1), repeat=3)), dtype='float64')
    _, valid = find_intersection_on_edge(corners.reshape(-1, 3),
                                         sign * corners[:, :, axis].reshape(-1),
                                         t.array([sign * iso], dtype='float64'))
    valid = valid.reshape(-1, 12)
    normals = t.zeros((len(coords), 12, 3), dtype='float64')
    normals[:, :, axis] = sign
    normals[~valid] = 0
    active = t.any(valid, axis=1)
    vertices = t.array(coords[active], dtype='float64') + 0.5
    vertices[:, axis] = iso
    return coords, valid, normals, vertices


def canonical_faces(faces):
    faces = np.sort(faces, axis=1)
    return faces[np.lexsort(faces.T[::-1])]


def oriented_faces(faces):
    rotated = np.take_along_axis(faces, (np.argmin(faces, axis=1)[:, None] + np.arange(3)) % 3, axis=1)
    return rotated[np.lexsort(rotated.T[::-1])]


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('sign', [-1, 1])
def test_complete_quads_legacy_parity_and_winding(backend, axis, sign):
    t = BackendTensor.t
    coords = np.array(list(product(range(3), range(4), range(5))), dtype=np.int64)
    np.random.default_rng(42).shuffle(coords)
    coords, valid, normals, vertices = plane_data(coords, axis, sign)
    report = {}
    faces = triangulate_quads(coords, valid, normals, vertices, (3, 4, 5), report=report)
    active = t.any(valid, axis=1)
    legacy = triangulate(coords[active], valid[active], 1, normals[active], vertices, (3, 4, 5))
    faces, legacy, vertices = map(t.to_numpy, (faces, legacy, vertices))
    np.testing.assert_array_equal(canonical_faces(faces), canonical_faces(legacy))
    np.testing.assert_array_equal(oriented_faces(faces), oriented_faces(legacy))
    assert len(faces) == 2 * report['quad_count'] > 0
    assert len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces)
    normal = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                      vertices[faces[:, 2]] - vertices[faces[:, 0]])
    assert np.all(normal[:, axis] * sign > 0)
    assert faces.dtype == np.int64
    assert report['physical_boundary_edge_count'] > 0
    assert report['unknown_boundary_edge_count'] == 0


@pytest.mark.parametrize('kind', ['masked', 'refinement', 'unknown'])
def test_sparse_missing_quad_classification(backend, kind):
    t = BackendTensor.t
    full = np.array(list(product([1], [1, 2], [1, 2])), dtype=np.int64)
    coords, valid, normals, vertices = plane_data(full[:-1])
    generated = None if kind == 'unknown' else t.array(full if kind == 'masked' else full[:-1], dtype='int64')
    report = {}
    faces = triangulate_quads(coords, valid, normals, vertices, (4, 4, 4), generated, report)
    assert faces.shape == (0, 3)  # Three vertices must not produce a partial quad.
    assert report['quad_count'] == 0
    assert report['physical_boundary_edge_count'] == 0
    key = {'masked': 'mask_boundary_edge_count', 'refinement': 'internal_refinement_boundary_edge_count',
           'unknown': 'unknown_boundary_edge_count'}[kind]
    assert report[key] > 0
    if kind == 'masked':
        # Some other neighbors were never generated, rather than masked out.
        assert report['internal_refinement_boundary_edge_count'] > 0


def test_sparse_large_domain_and_row_order(backend):
    t = BackendTensor.t
    # Large theoretical extent, only four generated cells and no dense domain allocation.
    coords = np.array(list(product([2 ** 32], [1, 2], [1, 2])), dtype=np.int64)
    results = []
    for order in ([0, 1, 2, 3], [3, 0, 2, 1]):
        cells, valid, normals, vertices = plane_data(coords[order], iso=2 ** 32 + 0.4)
        report = {}
        faces = triangulate_quads(cells, valid, normals, vertices, (2 ** 32 + 2, 4, 4), cells, report)
        assert faces.shape == (2, 3)
        assert report == dict(crossing_edge_count=9, quad_count=1, missing_incident_cell_count=20,
                              physical_boundary_edge_count=0, mask_boundary_edge_count=0,
                              internal_refinement_boundary_edge_count=8, unknown_boundary_edge_count=0)
        results.append(t.to_numpy(vertices[faces]))
    np.testing.assert_array_equal(*results)


def test_winding_uses_shared_edge_gradient(backend):
    t = BackendTensor.t
    coords = np.array(list(product([1], [1, 2], [1, 2])), dtype=np.int64)
    coords, valid, normals, vertices = plane_data(coords)
    normals[:, :, 0] = -100
    # The four copies of the central crossing point have positive x gradients.
    for cell, edge in enumerate([3, 2, 1, 0]):
        normals[cell, edge, 0] = 1
    faces = triangulate_quads(coords, valid, normals, vertices, (4, 4, 4))
    xyz = t.to_numpy(vertices[faces])
    assert np.all(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])[:, 0] > 0)


@pytest.mark.parametrize('case', ['empty', 'no_crossings', 'isolated', 'boundary'])
def test_empty_and_incomplete(backend, case):
    t = BackendTensor.t
    coords = [] if case == 'empty' else [[0, 0, 0] if case == 'boundary' else [1, 1, 1]]
    coords = np.array(coords, dtype=np.int64).reshape(-1, 3)
    coords, valid, normals, vertices = plane_data(coords, iso=10 if case == 'no_crossings' else 0.4 if case == 'boundary' else 1.4)
    report = {'stale': True}
    faces = triangulate_quads(coords, valid, normals, vertices, (4, 4, 4), report=report)
    assert faces.shape == (0, 3)
    assert t.to_numpy(faces).dtype == np.int64
    assert 'stale' not in report
    if case == 'boundary':
        assert report['physical_boundary_edge_count'] > 0
    if backend == 'PYTORCH':
        assert faces.device == coords.device


def test_duplicate_consistency(backend):
    coords = np.array(list(product([1], [1, 2], [1, 2])), dtype=np.int64)
    coords, valid, normals, vertices = plane_data(coords)
    valid[0, 3] = False  # Shared central x edge disagrees with three other cells.
    with pytest.raises(ValueError, match='Inconsistent crossing'):
        triangulate_quads(coords, valid, normals, vertices, (4, 4, 4))
    # Also check duplicates from retained cells that have no QEF vertex at all.
    valid[0, :] = False
    with pytest.raises(ValueError, match='Inconsistent crossing'):
        triangulate_quads(coords, valid, normals, vertices[1:], (4, 4, 4))
    coords, valid, normals, vertices = plane_data([[1, 1, 1], [1, 1, 1]])
    with pytest.raises(ValueError, match='Duplicate cell'):
        triangulate_quads(coords, valid, normals, vertices, (4, 4, 4))


def test_tolerant_crossing_rule(backend):
    coords = np.array(list(product([1], [1, 2], [1, 2])), dtype=np.int64)
    # Both endpoint values are below the isovalue, but inside the shipped tolerance.
    coords, valid, normals, vertices = plane_data(coords, iso=2.005)
    assert BackendTensor.t.all(valid[:, :4])
    faces = triangulate_quads(coords, valid, normals, vertices, (4, 4, 4))
    assert faces.shape == (2, 3)


def test_bounds_and_overflow(backend):
    coords, valid, normals, vertices = plane_data([[-1, 1, 1]])
    with pytest.raises(ValueError, match='theoretical domain'):
        triangulate_quads(coords, valid, normals, vertices, (4, 4, 4))
    with pytest.raises(OverflowError):
        triangulate_quads(coords, valid, normals, vertices, (2 ** 32, 2 ** 32, 2))


def test_selector_serialization():
    options = InterpolationOptions.from_args(range=1., c_o=1.)
    assert options.evaluation_options.triangulation_method is TriangulationMethod.LEGACY
    options.evaluation_options.triangulation_method = TriangulationMethod.QUADS
    restored = InterpolationOptions.model_validate_json(options.model_dump_json())
    assert restored.evaluation_options.triangulation_method is TriangulationMethod.QUADS


def test_public_model_mesh_parity(backend, monkeypatch):
    from gempy_engine.API.model.model_api import compute_model
    from tests.fixtures.simple_models import simple_model_interpolation_input_factory

    monkeypatch.setenv('GEMPY_FLAT_STACKS', 'False')
    monkeypatch.setenv('GEMPY_SKIP_TRIANGULATION', '0')
    results = []
    for method in (TriangulationMethod.LEGACY, TriangulationMethod.QUADS):
        interp, options, descriptor = simple_model_interpolation_input_factory()
        options.evaluation_options.number_octree_levels = 2
        options.evaluation_options.triangulation_method = method
        results.append(compute_model(interp, options, descriptor))
    assert len(results[0].dc_meshes) == len(results[1].dc_meshes) > 0
    for old, new in zip(results[0].dc_meshes, results[1].dc_meshes):
        np.testing.assert_allclose(old.vertices, new.vertices, atol=1e-8)
        np.testing.assert_array_equal(canonical_faces(old.edges), canonical_faces(new.edges))
        np.testing.assert_array_equal(oriented_faces(old.edges), oriented_faces(new.edges))
        assert len(new.edges) > 0
        assert new.dc_data.triangulation_method is TriangulationMethod.QUADS
        assert new.dc_data.generated_cell_coordinates is not None
        assert new.dc_data.triangulation_report['unknown_boundary_edge_count'] == 0
        assert new.dc_data.triangulation_report['quad_count'] * 2 == len(new.edges)
