from itertools import permutations

import numpy as np
import pytest

from gempy_engine.modules.dual_contouring.contact_cells import finalize_cell_vertices, reconcile_cell_vertices


@pytest.mark.parametrize('supported', [[False, True], [False, False], [True, True]])
def test_finalize_pair_support_restores_originals_without_mutation(supported):
    inputs = _inputs([0, 8])
    provisional, ids, _ = reconcile_cell_vertices(*inputs)
    faces = [np.array([[0, 0, 0]], dtype=int) if keep else np.empty((0, 3), dtype=int)
             for keep in supported]
    vertices, final_ids, report = finalize_cell_vertices(inputs[0], provisional, ids, faces, inputs[3])
    retained = all(supported)
    for surface in range(2):
        np.testing.assert_array_equal(vertices[surface], provisional[surface] if retained else inputs[0][surface])
        np.testing.assert_array_equal(final_ids[surface], [0 if retained else -1])
        np.testing.assert_array_equal(provisional[surface], [[4, 0, 0]])
        np.testing.assert_array_equal(ids[surface], [0])
    assert report['contact_count'] == int(retained)
    assert report['unsupported_contact_member_count'] == supported.count(False)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_finalize_three_way_mean_uses_supported_originals(dtype):
    inputs = _inputs([0, 2, 10], dtype=dtype)
    provisional, ids, _ = reconcile_cell_vertices(*inputs)
    faces = [np.empty((0, 3), dtype=int), np.array([[0, 0, 0]]), np.array([[0, 0, 0]])]
    vertices, final_ids, report = finalize_cell_vertices(inputs[0], provisional, ids, faces, inputs[3])
    np.testing.assert_array_equal(vertices[0], inputs[0][0])
    for surface in (1, 2):
        np.testing.assert_array_equal(vertices[surface], [[6, 0, 0]])
        assert vertices[surface].dtype == dtype
    assert [array[0] for array in final_ids] == [-1, 0, 0]
    assert report['contact_count'] == 1
    assert report['dissolved_contact_count'] == 0


def test_finalize_does_not_regroup_same_stack_competitors():
    inputs = _inputs([0, 4, 1], [0, 0, 1])
    provisional, ids, _ = reconcile_cell_vertices(*inputs)
    faces = [np.empty((0, 3), dtype=int), np.array([[0, 0, 0]]), np.array([[0, 0, 0]])]
    vertices, final_ids, report = finalize_cell_vertices(inputs[0], provisional, ids, faces, inputs[3])
    for actual, original in zip(vertices, inputs[0]):
        np.testing.assert_array_equal(actual, original)
    assert [array[0] for array in final_ids] == [-1, -1, -1]
    assert report['contact_count'] == 0


def test_finalize_preserves_fault_snap_with_discarded_target():
    inputs = _inputs([4, 0])
    faults = {(0, 1)}
    provisional, ids, _ = reconcile_cell_vertices(*inputs, fault_pairs=faults)
    faces = [np.array([[0, 0, 0]]), np.empty((0, 3), dtype=int)]
    vertices, final_ids, report = finalize_cell_vertices(inputs[0], provisional, ids, faces, inputs[3], faults)
    for actual, expected in zip(vertices, provisional):
        np.testing.assert_array_equal(actual, expected)
    assert [array[0] for array in final_ids] == [0, 0]
    assert report['contact_count'] == 1
    assert report['unsupported_contact_member_count'] == 0


def test_finalize_partial_group_mean_is_surface_order_independent():
    inputs = _inputs([1e16, 1., -1e16, 10.])
    provisional, ids, _ = reconcile_cell_vertices(*inputs)
    faces = [np.array([[0, 0, 0]]) for _ in range(3)] + [np.empty((0, 3), dtype=int)]
    expected = None
    for order in permutations(range(4)):
        positions, final_ids, report = finalize_cell_vertices(
            [inputs[0][i] for i in order], [provisional[i] for i in order],
            [ids[i] for i in order], [faces[i] for i in order], [inputs[3][i] for i in order],
        )
        for index, surface in enumerate(order):
            if surface == 3:
                np.testing.assert_array_equal(positions[index], inputs[0][surface])
                assert final_ids[index][0] == -1
            else:
                if expected is None:
                    expected = positions[index].copy()
                np.testing.assert_array_equal(positions[index], expected)
                assert final_ids[index][0] == ids[surface][0]
        assert report['contact_count'] == 1


def _inputs(xs, groups=None, cells=None, dtype=np.float64):
    vertices = [np.array([[x, 0, 0]], dtype=dtype) for x in xs]
    n = len(vertices)
    groups = list(range(n)) if groups is None else groups
    coordinates = [np.array([[0, 0, 0]], dtype=np.int64) for _ in xs] if cells is None else cells
    ids = [(group, i) for i, group in enumerate(groups)]
    allowed = ~np.eye(n, dtype=bool)
    return vertices, coordinates, groups, ids, allowed


def test_pair_mean_and_no_mutation():
    inputs = _inputs([0, 8])
    snapshots = [[array.copy() for array in inputs[i]] for i in (0, 1)]
    allowed = inputs[4].copy()
    vertices, ids, report = reconcile_cell_vertices(*inputs)
    for i in range(2):
        np.testing.assert_array_equal(vertices[i], [[4, 0, 0]])
        np.testing.assert_array_equal(ids[i], [0])
        assert ids[i].dtype == np.int64
        assert not np.shares_memory(vertices[i], inputs[0][i])
        for j, field in enumerate((0, 1)):
            np.testing.assert_array_equal(inputs[field][i], snapshots[j][i])
    np.testing.assert_array_equal(inputs[4], allowed)
    assert report['contact_count'] == 1
    assert report['conflict_count'] == 0


def test_three_way_mean_uses_original_positions():
    vertices, ids, report = reconcile_cell_vertices(*_inputs([0, 2, 10]))
    for positions, contact_ids in zip(vertices, ids):
        np.testing.assert_array_equal(positions, [[4, 0, 0]])
        np.testing.assert_array_equal(contact_ids, [0])
    assert report['contact_count'] == 1


def test_same_group_competitors_cannot_merge_transitively():
    vertices, ids, report = reconcile_cell_vertices(*_inputs([0, 4, 1], [0, 0, 1]))
    np.testing.assert_array_equal(vertices[0], [[0.5, 0, 0]])
    np.testing.assert_array_equal(vertices[2], vertices[0])
    np.testing.assert_array_equal(vertices[1], [[4, 0, 0]])
    assert [array[0] for array in ids] == [0, -1, 0]
    assert report['nonfault_conflict_count'] == 2
    assert all(conflict['reason'] == 'same_group' for conflict in report['conflicts'])


def test_nonfault_sets_require_all_cross_pairs_allowed():
    inputs = _inputs([0, 1, 3])
    inputs[4][0, 2] = inputs[4][2, 0] = False
    vertices, ids, report = reconcile_cell_vertices(*inputs)
    assert [array[0] for array in ids] == [0, 0, -1]
    np.testing.assert_array_equal(vertices[2], inputs[0][2])
    assert report['conflicts'][0]['reason'] == 'disallowed_cross_pair'


@pytest.mark.parametrize('faults', [(), ((0, 2), (0, 3), (1, 2))])
def test_mesh_and_row_permutations_preserve_positions_and_ids(faults):
    inputs = _inputs([0, 0, 1, 2], [0, 0, 1, 2])
    for i in range(4):
        inputs[0][i] = np.vstack([inputs[0][i], inputs[0][i] + [8, 0, 0]])
        inputs[1][i] = np.array([[5, 0, 0], [-2, 0, 0]])
    expected_vertices, expected_ids, expected_report = reconcile_cell_vertices(*inputs, fault_pairs=faults)
    order = [3, 1, 2, 0]
    row_order = [1, 0]
    reordered = ([inputs[0][i][row_order] for i in order],
                 [inputs[1][i][row_order] for i in order],
                 [inputs[2][i] for i in order], [inputs[3][i] for i in order],
                 inputs[4][np.ix_(order, order)])
    directed = [(order.index(a), order.index(b)) for a, b in reversed(faults)]
    actual_vertices, actual_ids, report = reconcile_cell_vertices(*reordered, fault_pairs=directed)
    for new_i, old_i in enumerate(order):
        np.testing.assert_array_equal(actual_vertices[new_i][row_order], expected_vertices[old_i])
        np.testing.assert_array_equal(actual_ids[new_i][row_order], expected_ids[old_i])
    assert report['conflicts'] == expected_report['conflicts']
    assert report['contact_count'] == 2


def test_fault_direction_and_multiple_targets_use_original_controller():
    inputs = _inputs([4, 0, 10])
    inputs[4][:] = False
    vertices, ids, report = reconcile_cell_vertices(*inputs, fault_pairs=((0, 1), (0, 2)))
    for positions in vertices:
        np.testing.assert_array_equal(positions, [[4, 0, 0]])
    assert [array[0] for array in ids] == [0, 0, 0]
    assert [array.tolist() for array in report['fault_overlap_vertices']] == [[], [0], [0]]
    assert report['fault_conflict_count'] == 0
    assert report['contact_count'] == 1
    np.testing.assert_array_equal(inputs[0][1], [[0, 0, 0]])


def test_competing_fault_controllers_report_rejection_and_all_overlap_targets():
    inputs = _inputs([0, 4, 1])
    inputs[4][:] = False
    vertices, ids, report = reconcile_cell_vertices(*inputs, fault_pairs=((1, 2), (0, 2)))
    assert [array[0] for array in ids] == [0, -1, 0]
    np.testing.assert_array_equal(vertices[1], [[4, 0, 0]])
    np.testing.assert_array_equal(vertices[2], [[0, 0, 0]])
    assert report['conflict_count'] == report['fault_conflict_count'] == 1
    assert report['conflicts'][0]['reason'] == 'competing_controller'
    assert report['fault_overlap_vertices'][2].tolist() == [0]


def test_fault_same_group_targets_compete_without_losing_overlap_metadata():
    inputs = _inputs([0, 1, 4], [0, 1, 1])
    vertices, ids, report = reconcile_cell_vertices(*inputs, fault_pairs=((0, 1), (0, 2)))
    assert [array[0] for array in ids] == [0, 0, -1]
    np.testing.assert_array_equal(vertices[2], [[4, 0, 0]])
    assert report['fault_conflict_count'] == 1
    assert [array.tolist() for array in report['fault_overlap_vertices']] == [[], [0], [0]]


def test_nonfault_contacts_never_overwrite_fault_anchor():
    vertices, ids, report = reconcile_cell_vertices(*_inputs([0, 4, 5]), fault_pairs=((0, 1),))
    np.testing.assert_array_equal(vertices[0], [[0, 0, 0]])
    np.testing.assert_array_equal(vertices[1], vertices[0])
    np.testing.assert_array_equal(vertices[2], [[5, 0, 0]])
    assert [array[0] for array in ids] == [0, 0, -1]
    assert report['nonfault_conflict_count'] == 2
    assert all(c['reason'] == 'fault_anchored' for c in report['conflicts'])


def test_fault_chain_cannot_move_an_existing_controller():
    vertices, ids, report = reconcile_cell_vertices(*_inputs([0, 1, 5]), fault_pairs=((0, 1), (1, 2)))
    np.testing.assert_array_equal(vertices[1], [[0, 0, 0]])
    np.testing.assert_array_equal(vertices[2], [[5, 0, 0]])
    assert [array[0] for array in ids] == [0, 0, -1]
    assert report['fault_conflict_count'] == 1
    assert report['fault_overlap_vertices'][2].tolist() == [0]


@pytest.mark.parametrize('count', [0, 1, 3])
def test_empty_surfaces(count):
    vertices = [np.empty((0, 3), dtype=np.float32) for _ in range(count)]
    coordinates = [np.empty((0, 3), dtype=np.int64) for _ in range(count)]
    new_vertices, ids, report = reconcile_cell_vertices(
        vertices, coordinates, list(range(count)), [(i, 0) for i in range(count)],
        np.zeros((count, count), dtype=bool))
    assert len(new_vertices) == len(ids) == count
    assert report['contact_count'] == report['conflict_count'] == 0
    assert all(array.shape == (0,) and array.dtype == np.int64 for array in ids)


def test_sparse_disjoint_cells_and_empty_graph_leave_vertices_alone():
    inputs = _inputs([0, 2, 4], cells=[np.array([[i * 10**12, -i, 0]]) for i in range(3)])
    for allowed in (inputs[4], np.zeros((3, 3), dtype=bool)):
        vertices, ids, report = reconcile_cell_vertices(*inputs[:4], allowed)
        for original, actual in zip(inputs[0], vertices):
            np.testing.assert_array_equal(original, actual)
        assert all(array.tolist() == [-1] for array in ids)
        assert report['contact_count'] == report['conflict_count'] == 0
    inputs[1][:] = [np.zeros((1, 3), dtype=int) for _ in range(3)]
    _, ids, report = reconcile_cell_vertices(*inputs[:4], np.zeros((3, 3), dtype=bool))
    assert all(array.tolist() == [-1] for array in ids)
    assert report['contact_count'] == 0


def test_float32_preservation():
    vertices, _, _ = reconcile_cell_vertices(*_inputs([0, 2, 10], dtype=np.float32))
    for positions in vertices:
        assert positions.dtype == np.float32
        np.testing.assert_array_equal(positions, [[4, 0, 0]])


def test_hidden_ordinary_vertices_do_not_move_visible_surface():
    inputs = _inputs([0, 2])
    vertices, ids, report = reconcile_cell_vertices(
        *inputs, contact_eligible=[np.array([True]), np.array([False])])
    for original, actual in zip(inputs[0], vertices):
        np.testing.assert_array_equal(original, actual)
    assert [array[0] for array in ids] == [-1, -1]
    assert report['contact_count'] == report['conflict_count'] == 0


def test_mixed_corner_ownership_authorizes_nonfault_contact():
    ownership = [np.array([[True, False, False, False]]),
                 np.array([[False, False, True, False]])]
    vertices, ids, report = reconcile_cell_vertices(
        *_inputs([0, 2]), contact_eligible=[corners.any(axis=1) for corners in ownership])
    for positions in vertices:
        np.testing.assert_array_equal(positions, [[1, 0, 0]])
    assert [array[0] for array in ids] == [0, 0]
    assert report['contact_count'] == 1


def test_hidden_fault_candidates_still_copy_and_report_rejected_overlaps():
    vertices, ids, report = reconcile_cell_vertices(
        *_inputs([0, 4, 1]), fault_pairs=((0, 2), (1, 2)),
        contact_eligible=[np.array([False]) for _ in range(3)])
    np.testing.assert_array_equal(vertices[2], [[0, 0, 0]])
    assert [array[0] for array in ids] == [0, -1, 0]
    assert report['fault_conflict_count'] == 1
    assert report['fault_overlap_vertices'][2].tolist() == [0]


@pytest.mark.parametrize('masks', [[], [np.array([True])],
                                  [np.array([True, False]), np.array([True])],
                                  [np.array([1]), np.array([True])]])
def test_invalid_eligibility_masks(masks):
    with pytest.raises(ValueError, match='contact_eligible'):
        reconcile_cell_vertices(*_inputs([0, 2]), contact_eligible=masks)


@pytest.mark.parametrize('invalid', [
    'length', 'shape', 'nonfinite', 'integer_positions', 'mixed_dtypes', 'cell_shape',
    'float_cells', 'duplicate_cells', 'duplicate_ids', 'bad_ids', 'bad_groups',
    'graph_shape', 'graph_dtype', 'asymmetric_graph', 'fault_range', 'fault_self', 'fault_type'
])
def test_invalid_inputs(invalid):
    inputs = list(_inputs([0, 2]))
    faults = ()
    if invalid == 'length':
        inputs[2] = [0]
    elif invalid == 'shape':
        inputs[0][0] = np.zeros((1, 2))
    elif invalid == 'nonfinite':
        inputs[0][0][0, 0] = np.nan
    elif invalid == 'integer_positions':
        inputs[0][0] = inputs[0][0].astype(int)
    elif invalid == 'mixed_dtypes':
        inputs[0][1] = inputs[0][1].astype(np.float32)
    elif invalid == 'cell_shape':
        inputs[1][0] = np.zeros((2, 3), dtype=int)
    elif invalid == 'float_cells':
        inputs[1][0] = inputs[1][0].astype(float)
    elif invalid == 'duplicate_cells':
        inputs[0][0] = np.repeat(inputs[0][0], 2, axis=0)
        inputs[1][0] = np.repeat(inputs[1][0], 2, axis=0)
    elif invalid == 'duplicate_ids':
        inputs[3] = [(0, 0), (0, 0)]
    elif invalid == 'bad_ids':
        inputs[3] = [(0, 0), (1, 0.5)]
    elif invalid == 'bad_groups':
        inputs[2] = [0, 0.5]
    elif invalid == 'graph_shape':
        inputs[4] = np.zeros((1, 1), dtype=bool)
    elif invalid == 'graph_dtype':
        inputs[4] = inputs[4].astype(int)
    elif invalid == 'asymmetric_graph':
        inputs[4][0, 1] = False
    elif invalid == 'fault_range':
        faults = ((0, 2),)
    elif invalid == 'fault_self':
        faults = ((0, 0),)
    elif invalid == 'fault_type':
        faults = ((0, 1.0),)
    with pytest.raises(ValueError):
        reconcile_cell_vertices(*inputs, fault_pairs=faults)
