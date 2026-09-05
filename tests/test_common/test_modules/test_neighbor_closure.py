from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.options.evaluation_options import EvaluationOptions
from gempy_engine.modules.octrees_topology._neighbor_closure import close_refinement_mask
from gempy_engine.modules.octrees_topology._octree_common import _generate_next_level_centers
from gempy_engine.modules.octrees_topology._octree_internals import compute_next_octree_locations
from gempy_engine.modules.dual_contouring.fancy_triangulation import get_left_right_array
from gempy_engine.modules.dual_contouring.fancy_triangulation import triangulate
from gempy_engine.modules.dual_contouring._support_report import mesh_support_report


@pytest.fixture(params=['numpy', 'PYTORCH'])
def backend(request):
    if request.param == 'PYTORCH':
        pytest.importorskip('torch')
    old = BackendTensor.engine_backend
    old_gpu = BackendTensor.use_gpu
    old_dtype = BackendTensor.dtype
    old_keops = BackendTensor.use_pykeops
    BackendTensor._change_backend(engine_backend=getattr(AvailableBackends, request.param), use_gpu=False, dtype='float64')
    yield
    BackendTensor._change_backend(engine_backend=old, use_gpu=old_gpu, dtype=old_dtype, use_pykeops=old_keops)


@pytest.mark.parametrize('mode,seed,expected', [
    ('fast', (2, 2, 2), 1), ('balanced', (2, 2, 2), 7),
    ('precise', (2, 2, 2), 27), ('precise', (0, 2, 2), 18),
    ('precise', (0, 0, 2), 12), ('precise', (0, 0, 0), 8),
])
def test_stencil(backend, mode, seed, expected):
    t = BackendTensor.t
    coords = t.array(list(product(range(5), range(6), range(7))), dtype='int64')
    primary = (coords == t.array(seed)).all(axis=1)
    support, missing = close_refinement_mask(coords, primary, (5, 6, 7), mode)
    assert int((support | primary).sum()) == expected
    assert not bool((support & primary).any())
    assert missing == 0


def test_sheet_overlap_and_empty(backend):
    t = BackendTensor.t
    coords = t.array(list(product(range(7), repeat=3)), dtype='int64')
    primary = coords[:, 0] == 3
    support, missing = close_refinement_mask(coords, primary, (7, 7, 7), 'precise')
    np.testing.assert_array_equal(t.to_numpy(support | primary), t.to_numpy(abs(coords[:, 0] - 3) <= 1))
    assert missing == 0
    support, missing = close_refinement_mask(coords[:0], primary[:0], (7, 7, 7), 'precise')
    assert len(support) == missing == 0


def test_missing_and_deep_coordinates(backend):
    t = BackendTensor.t
    coords = t.array([[255, 1, 1], [256, 1, 1], [257, 1, 1]], dtype='int64')
    primary = t.array([True, False, False], dtype=bool)
    support, missing = close_refinement_mask(coords, primary, (1024, 3, 3), 'balanced')
    np.testing.assert_array_equal(t.to_numpy(support), [False, True, False])
    assert missing == 5


def test_lineage(backend):
    t = BackendTensor.t
    root = RegularGrid([0, 300, 0, 2, 0, 2], [300, 2, 2])
    selected = root.integer_coordinates[:, 0] == 255
    xyz, bits = _generate_next_level_centers(root.values[selected], root.dxdydz)
    child = RegularGrid.from_octree_level(xyz, root, selected, bits)
    expected = 2 * np.repeat(t.to_numpy(root.integer_coordinates[selected]), 8, axis=0) + t.to_numpy(bits)
    np.testing.assert_array_equal(t.to_numpy(child.integer_coordinates), expected)
    assert len(np.unique(expected, axis=0)) == len(expected)
    codes, bounds = get_left_right_array([SimpleNamespace(grid=SimpleNamespace(octree_grid=child))])
    assert bounds == (600, 4, 4)
    np.testing.assert_array_equal(t.to_numpy(codes), expected)
    geometric = (t.to_numpy(xyz) - t.to_numpy(root.orthogonal_extent)[::2]) / np.array([float(v) for v in child.dxdydz]) - 0.5
    np.testing.assert_allclose(geometric, expected, atol=1e-6)


@pytest.mark.parametrize('mode,count', [('fast', 1), ('balanced', 7), ('precise', 27)])
def test_refinement_and_surface_depth(backend, mode, count):
    t = BackendTensor.t
    grid = RegularGrid([0, 5, 0, 5, 0, 5], [5, 5, 5])
    ids = t.zeros((125, 8))
    ids[62, 0] = 1
    prev = SimpleNamespace(litho_faults_ids_corners_grid=ids.reshape(-1),
                           dxdydz=grid.dxdydz, grid=SimpleNamespace(octree_grid=grid), outputs=[])
    options = EvaluationOptions(_number_octree_levels=4, _number_octree_levels_surface=2,
                                octree_min_level=0, octree_refinement_mode=mode, verbose=True)
    result = compute_next_octree_locations(prev, options, 0).octree_grid
    assert int(result.active_cells.sum()) == count
    assert len(result.values) == count * 8
    assert result.refinement_debug['primary_surface_count'] == 1
    assert result.refinement_debug['support_only_count'] == count - 1
    result = compute_next_octree_locations(prev, options, 1).octree_grid
    assert int(result.active_cells.sum()) == 1


def test_support_classification(backend):
    t = BackendTensor.t
    coords = np.array(list(product(range(3), repeat=3)))
    corners = np.array(list(product((0, 1), repeat=3)))
    scalar = (coords[:, None, :] + corners)[..., 0].astype(float)
    report = mesh_support_report(t.array(coords), t.array(scalar), 1.5, (3, 3, 3))
    assert report['crossing_edge_count'] == 16
    assert report['physical_boundary_edge_count'] == 12
    assert report['internal_refinement_boundary_edge_count'] == 0
    retained = ~np.all(coords == [1, 1, 1], axis=1)
    report = mesh_support_report(t.array(coords), t.array(scalar), 1.5, (3, 3, 3), t.array(retained))
    assert report['mask_boundary_edge_count'] == 4
    assert report['internal_refinement_boundary_edge_count'] == 0
    report = mesh_support_report(t.array(coords[retained]), t.array(scalar[retained]), 1.5, (3, 3, 3))
    assert report['internal_refinement_boundary_edge_count'] == 4


def test_closed_sphere_connectivity(backend):
    t = BackendTensor.t
    corners = t.array(list(product((0, 1), repeat=3)), dtype='int64')
    grid = RegularGrid([0, 8, 0, 8, 0, 8], [8, 8, 8])
    options = EvaluationOptions(_number_octree_levels=3, _number_octree_levels_surface=3,
                                octree_min_level=0, octree_refinement_mode='precise')
    for level in range(3):
        points = (grid.integer_coordinates[:, None, :] + corners) / (2 ** level)
        scalar = ((points - 4.13) ** 2).sum(axis=2) - 2.37 ** 2
        if level < 2:
            prev = SimpleNamespace(litho_faults_ids_corners_grid=t.array(scalar >= 0, dtype='int64').reshape(-1),
                                   dxdydz=grid.dxdydz, grid=SimpleNamespace(octree_grid=grid), outputs=[])
            grid = compute_next_octree_locations(prev, options, level).octree_grid
    report = mesh_support_report(grid.integer_coordinates, scalar, 0., grid.regular_grid_shape)
    assert report['internal_refinement_boundary_edge_count'] == 0
    assert report['physical_boundary_edge_count'] == 0
    pairs = [(0, 4), (1, 5), (2, 6), (3, 7), (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 1), (2, 3), (4, 5), (6, 7)]
    valid = t.stack([(scalar[:, a] >= 0) != (scalar[:, b] >= 0) for a, b in pairs], axis=1)
    active = valid.any(axis=1)
    coords = grid.integer_coordinates[active]
    vertices = t.array(coords, dtype='float64') + 0.5
    triangles = triangulate(coords, valid[active], 3, t.ones((len(coords), 12, 3)),
                            vertices, tuple(int(n) for n in grid.regular_grid_shape))
    triangles = t.to_numpy(triangles)
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert len(counts) > 0
    assert np.all(counts == 2)


@pytest.mark.parametrize('mode', ['balanced', 'precise'])
def test_fault_model_mode_integration(graben_fault_model, mode):
    from gempy_engine.API.model.model_api import compute_model

    interpolation_input, structure, options = graben_fault_model
    options.evaluation_options.number_octree_levels = 4
    options.evaluation_options.octree_refinement_mode = mode
    options.debug = True
    solutions = compute_model(interpolation_input, options, structure)
    assert solutions.dc_meshes
    for mesh in solutions.dc_meshes:
        assert mesh.support_report is not None
        assert 'stack_index' in mesh.support_report


def test_mode_serialization():
    from gempy_engine.core.data import InterpolationOptions, OctreeRefinementMode

    options = InterpolationOptions.from_args(range=1., c_o=1.)
    assert options.evaluation_options.octree_refinement_mode == OctreeRefinementMode.FAST
    options.evaluation_options.octree_refinement_mode = OctreeRefinementMode.PRECISE
    restored = InterpolationOptions.model_validate_json(options.model_dump_json())
    assert restored.evaluation_options.octree_refinement_mode == OctreeRefinementMode.PRECISE
