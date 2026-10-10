"""Pure-array unit tests of the joint extraction modules (no compute_model)."""

import numpy as np
import pytest

from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.dual_contouring.joint_cell_branches import CORNERS
from gempy_engine.modules.dual_contouring.joint_cell_complex import build_adaptive_complex
from gempy_engine.modules.dual_contouring.joint_contact_relations import build_contact_relations
from gempy_engine.modules.dual_contouring.joint_edge_decisions import crossing_normals, edge_crossing_decisions
from gempy_engine.modules.dual_contouring.joint_field_queries import ExactPointCache, TargetedFieldBatch
from gempy_engine.modules.dual_contouring.joint_lattice import (field_tolerances, lattice_frame, leaf_corner_nodes,
                                                                lookup_node_fields, minimal_edge_nodes,
                                                                shared_node_fields, tile_nodes)
from gempy_engine.modules.dual_contouring.joint_ownership import (borrowed_hermite_rows, borrowed_leaves,
                                                                  check_composite_controllers,
                                                                  check_missing_controllers, contact_pairs,
                                                                  controllers_from_ownership, fallback_region,
                                                                  fault_merge_targets, ordinary_controllers)
from gempy_engine.modules.dual_contouring.joint_triangle_plan import plan_edge_rows


def _uniform(n=2):
    origins = np.array([(i, j, k) for i in range(n) for j in range(n) for k in range(n)])
    return origins, np.ones(len(origins), dtype=int), np.array([n, n, n])


# region joint_lattice

def test_lattice_nodes_lie_on_leaf_corners():
    origins, spans, domain = _uniform()
    complex_ = build_adaptive_complex(origins, spans, domain)
    corners = leaf_corner_nodes(origins, spans, CORNERS)
    assert corners.shape == (8, 8, 3)
    known = {tuple(c) for c in corners.reshape(-1, 3)}
    tiles = tile_nodes(complex_['faces'])
    edges = minimal_edge_nodes(complex_['edges'])
    assert tiles.shape == (len(complex_['faces']), 4, 3) and edges.shape == (len(complex_['edges']), 2, 3)
    assert {tuple(c) for c in tiles.reshape(-1, 3)} <= known
    assert {tuple(c) for c in edges.reshape(-1, 3)} <= known
    # Every minimal edge has unit length along its axis on a uniform grid.
    assert np.all(np.abs(edges[:, 1]-edges[:, 0]).sum(axis=1) == 1)


def test_shared_node_lookup_and_disagreement():
    origins, spans, domain = _uniform()
    corners = leaf_corner_nodes(origins, spans, CORNERS)
    fields = corners[..., 2][None].astype(float)-.5
    tolerances = field_tolerances(fields+.5, np.array([.5]))
    lattice = shared_node_fields(corners, fields, domain, tolerances)
    np.testing.assert_array_equal(lookup_node_fields(np.array([[0, 0, 2], [1, 1, 0]]), *lattice)[0], [1.5, -.5])
    coarse = leaf_corner_nodes(np.zeros((1, 3), dtype=int), np.array([2]), CORNERS)
    coarse_lattice = shared_node_fields(coarse, np.ones((1, 1, 8)), domain, tolerances)
    with pytest.raises(ValueError, match='unsupported_adaptive_node'):
        lookup_node_fields(np.array([[1, 1, 1]]), *coarse_lattice)
    broken = fields.copy()
    broken[0, 0, 7] += 1e-6
    with pytest.raises(ValueError, match='inconsistent_corner_samples'):
        shared_node_fields(corners, broken, domain, tolerances)


def test_lattice_frame_rejects_invalid_extent():
    bounds, spacing = lattice_frame((0, 2, 0, 4, 0, 8), (2, 2, 2))
    np.testing.assert_array_equal(spacing, [1, 2, 4])
    with pytest.raises(ValueError, match='invalid_extent'):
        lattice_frame((0, 0, 0, 1, 0, 1), (1, 1, 1))

# endregion


# region joint_ownership

def test_controllers_follow_truncations_and_ownership():
    identities = [(0, 0), (1, 0)]
    controllers = ordinary_controllers(identities, {(0, 1)})
    assert controllers == {0: [], 1: [(0, -1)]}
    assert contact_pairs(controllers, np.ones((2, 2), dtype=bool)) == {(0, 1)}
    check_composite_controllers({1: [(0, -1)]}, controllers)
    with pytest.raises(ValueError, match='unsupported_ownership'):
        check_composite_controllers({1: [(0, 1)]}, controllers)
    fields = np.stack([np.linspace(-1, 1, 8)[None].repeat(3, 0), np.ones((3, 8))])
    ownership = np.stack([np.ones((3, 8)), -fields[0]])
    assert controllers_from_ownership(ownership, fields, {(0, 1)}) == {0: [], 1: [(0, -1)]}
    with pytest.raises(ValueError, match='unsupported_ownership'):
        controllers_from_ownership(ownership+.5, fields, {(0, 1)})


def test_missing_controller_is_rejected():
    relations = [R.ERODE, R.BASEMENT]
    isovalues = [np.array([.2, .4]), np.array([.5])]
    full = [(0, 0), (0, 1), (1, 0)]
    _, _, truncations = build_contact_relations(np.array([0, 0, 1]), np.array([0, 1, 0]), relations, None, isovalues)
    check_missing_controllers(full, truncations, np.array([0, 1]), np.array([0, 0]), isovalues)
    with pytest.raises(ValueError, match='unsupported_missing_controller'):
        check_missing_controllers(full, truncations, np.array([0, 1]), np.array([1, 0]), isovalues)


def test_fault_merge_borrowing_and_multi_fault_leaves():
    relations = [R.FAULT, R.FAULT, R.BASEMENT]
    targets = fault_merge_targets({0: {2}, 1: {2}}, [0, 1, 2], relations, {(0, 2), (1, 2)})
    assert targets == {2: [0, 1]}
    with pytest.raises(ValueError, match='invalid_fault_merge_contract'):
        fault_merge_targets({0: {2}}, [0, 1, 2], relations, {(0, 2), (1, 2)})
    stricts = np.zeros((3, 3, 12), dtype=bool)
    stricts[0, [0, 1], 0] = stricts[1, [1, 2], 0] = stricts[2, :, 0] = True
    merged, multi, borrowed = borrowed_leaves(targets, stricts)
    np.testing.assert_array_equal(merged[2], [True, False, True])
    np.testing.assert_array_equal(multi[2], [False, True, False])
    assert borrowed == {(2, 0): 0, (2, 2): 1}
    edge_xyz = np.random.default_rng(0).normal(size=(3, 3, 12, 3))
    normals = np.ones_like(edge_xyz)
    rows = borrowed_hermite_rows(0, np.array([0, 1]), {2}, borrowed, edge_xyz, normals, stricts, 10.)
    xyz, _, weights = rows
    assert xyz.shape == (2, 12, 3)
    np.testing.assert_array_equal(xyz[0], edge_xyz[2, 0])
    assert weights[0, 0] == 10. and not weights[1].any()
    assert borrowed_hermite_rows(0, np.array([1]), {2}, borrowed, edge_xyz, normals, stricts, 10.) is None


def test_fallback_region_adds_edge_neighbours():
    edges = [dict(cells=(0, 1, None, None)), dict(cells=(1, 2, 3, None)), dict(cells=(4, 4, 5, 5))]
    assert fallback_region(edges, np.array([True, False, False, False, False, False])) == {0, 1}
    assert fallback_region(edges, np.array([False, True, False, False, False, False])) == {0, 1, 2, 3}

# endregion


# region joint_edge_decisions and edge rows

def test_edge_decisions_keep_owned_interior_crossings():
    # One surface crossing two interior edges; its controller removes the second.
    edge_fields = np.array([[[-1., 1.], [-1., 3.]], [[1., 1.], [-1., -1.]]])
    edge_cells = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    counts = np.ones((2, 4), dtype=int)
    selected, fraction = edge_crossing_decisions(edge_fields, {0: [(1, 1)], 1: []}, edge_cells, counts)
    np.testing.assert_array_equal(selected, [[True, False], [False, False]])
    np.testing.assert_allclose(fraction[0], [.5, .25])
    gradients = np.zeros((2, 2, 2, 3))
    gradients[0, :, 1, 2] = 4.
    np.testing.assert_allclose(crossing_normals(fraction, gradients)[0, :, 2], [2., 1.])
    exterior = np.array([[0, 1, -1, -1], [0, 1, 2, 3]])
    assert not edge_crossing_decisions(edge_fields, {0: [], 1: []}, exterior, counts)[0][0, 0]
    with pytest.raises(ValueError, match='unsupported_hanging_branch'):
        edge_crossing_decisions(edge_fields, {0: [], 1: []}, edge_cells, np.full((2, 4), 2))
    aligned = edge_fields.copy()
    aligned[0, 0, 0] = 0.
    with pytest.raises(ValueError, match='sample_aligned_interface'):
        edge_crossing_decisions(aligned, {0: [], 1: []}, edge_cells, counts)


def test_edge_rows_split_borrowed_rings_and_drop_all_fault_triangles():
    edges = [dict(cells=(0, 1, 2, 3))]

    def regular_key(s, cell):
        return 'regular', ((s, 0),), (cell, 0, 0), 1, 0

    def vertex_key(s, cell):
        return regular_key(0 if (s, cell) in borrowed else s, cell)

    borrowed = {(1, 0): 0, (1, 1): 0, (1, 3): 0}
    normals = np.zeros((2, 1, 3))
    plans, filtered = plan_edge_rows(np.array([[False], [True]]), edges, normals, regular_key, vertex_key,
                                     borrowed, {(0, 0)})
    assert not plans[0] and filtered == 1
    assert len(plans[1]) == 1 and len(plans[1][0]['keys']) == 3
    assert plans[1][0]['regular_keys'] == [regular_key(1, c) for c in range(4)]

# endregion


# region joint_field_queries

def test_exact_point_cache_samples_each_point_once():
    calls = []

    def sample(points):
        calls.append(len(points))
        return points[:, :1].T*np.ones((2, 1)), np.ones((2, len(points), 3))

    cache = ExactPointCache(sample, 2)
    values, _ = cache(np.array([[0., 0, 0], [1, 0, 0]]))
    values2, _ = cache(np.array([[1., 0, 0], [2, 0, 0], [0, 0, 0]]))
    assert calls == [2, 1] and len(cache) == 3
    np.testing.assert_array_equal(values2[0], [1, 2, 0])
    with pytest.raises(ValueError, match='invalid_callback_samples'):
        ExactPointCache(lambda p: (np.zeros((1, len(p))), np.zeros((1, len(p), 3))), 2)(np.zeros((1, 3)))


def test_targeted_batch_validates_shapes_and_counts_points():
    def query(requests):
        return [np.zeros((len(s), len(p)) + ((3,) if k == 'gradient' else ())) for p, s, k in requests]

    batch = TargetedFieldBatch(query)
    scalars, empty, gradients = batch([(np.zeros((4, 3)), [0, 1], 'scalar'), (np.zeros((0, 3)), [0], 'scalar'),
                                       (np.zeros((2, 3)), [1], 'gradient')])
    assert scalars.shape == (2, 4) and empty.shape == (1, 0) and gradients.shape == (1, 2, 3)
    assert batch.sampled == 6
    with pytest.raises(ValueError, match='invalid_callback_samples'):
        TargetedFieldBatch(lambda requests: [np.zeros(1)])([(np.zeros((2, 3)), [0], 'scalar')])

# endregion
