"""Supported cell-local attachment, separate from provisional fault grouping."""

from itertools import permutations

import numpy as np
import pytest

from gempy_engine.modules.dual_contouring.contact_cells import attach_fault_junctions


def _junction(dtype="float64"):
    vertices = [np.array(points, dtype=dtype) for points in (
        [[0, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, .1, .05], [1, 0, 0], [0, -1, 0]],
    )]
    cells = [np.array(points, dtype=np.int64) for points in (
        [[2, 2, 2], [2, 3, 3], [2, 2, 3]],
        [[2, 2, 2], [3, 2, 2], [2, 3, 2]],
        [[2, 2, 2], [3, 2, 2], [2, 1, 2]],
    )]
    ids = [np.array(values, dtype=np.int64) for values in ([0, -1, -1], [0, 1, -1], [-1, 1, -1])]
    faces = [np.array([[0, 1, 2]], dtype=np.int64), np.array([[0, 1, 2]], dtype=np.int64),
             np.array([[1, 0, 2]], dtype=np.int64)]
    allowed = np.zeros((3, 3), dtype=bool)
    allowed[1, 2] = allowed[2, 1] = True
    return vertices, ids, faces, cells, [(0, 0), (1, 0), (2, 0)], allowed, {(0, 1)}, {(1, 2)}


def _run(data):
    vertices, ids, faces, cells, surfaces, allowed, faults, truncation = data
    return attach_fault_junctions(vertices, vertices, ids, faces, cells, surfaces, allowed, faults, truncation)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_attachment_copies_anchor_preserves_inputs_and_surface_order(dtype):
    data = _junction(dtype)
    expected_positions, expected_ids, report = _run(data)
    assert report['fault_junction_attachment_count'] == 1
    assert report['fault_junction_rejected_count'] == 0
    np.testing.assert_array_equal(expected_positions[2][0], data[0][0][0])
    assert expected_ids[2][0] == 0
    for surface in range(3):
        np.testing.assert_array_equal(expected_positions[surface][1:], data[0][surface][1:])
        assert not np.shares_memory(expected_positions[surface], data[0][surface])
        assert not np.shares_memory(expected_ids[surface], data[1][surface])
    assert data[1][2][0] == -1
    np.testing.assert_array_equal(data[0][2][0], np.array([0, .1, .05], dtype=dtype))
    for order in permutations(range(3)):
        reordered = tuple([values[i] for i in order] for values in data[:5]) + (
            data[5][np.ix_(order, order)],
            {(order.index(a), order.index(b)) for a, b in data[6]},
            {(order.index(a), order.index(b)) for a, b in data[7]},
        )
        positions, ids, actual = _run(reordered)
        assert actual == report
        for new, old in enumerate(order):
            np.testing.assert_array_equal(positions[new], expected_positions[old])
            np.testing.assert_array_equal(ids[new], expected_ids[old])


@pytest.mark.parametrize("case,reason", [
    ("same_group", "same_group"), ("fault_target", "fault_participant"),
    ("fault_controller", "fault_participant"), ("disallowed", "disallowed_cross_pair"),
    ("collapse", "collapsed_or_inverted_face"), ("flip", "collapsed_or_inverted_face"),
    ("duplicate", "duplicate_shared_patch"),
])
def test_unsafe_junctions_are_rejected(case, reason):
    data = _junction()
    vertices, ids, faces, cells, surfaces, allowed, faults, _ = data
    if case == 'same_group':
        surfaces[2] = (1, 1)
    elif case == 'fault_target':
        faults.add((0, 2))
    elif case == 'fault_controller':
        faults.add((2, 0))
    elif case == 'disallowed':
        allowed[:] = False
    elif case in ('collapse', 'flip'):
        vertices[0][0] = vertices[1][0] = [0, -1 if case == 'collapse' else -2, 0]
    elif case == 'duplicate':
        ids[1][2] = ids[2][2] = 2
        cells[2][2] = cells[1][2]
        vertices[2][2] = vertices[1][2]
    before = [value.copy() for value in vertices]
    positions, final_ids, report = _run(data)
    assert report['fault_junction_attachment_count'] == 0
    assert report['fault_junction_rejected_count'] == 1
    assert report['fault_junction_rejections'][0]['reason'] == reason
    for actual, original, output_ids, original_ids in zip(positions, before, final_ids, ids):
        np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(output_ids, original_ids)


@pytest.mark.parametrize("case", ['missing_edge', 'different_cell', 'unshared_neighbor', 'reverse_relation',
                                  'already_shared', 'no_faults'])
def test_no_attachment_without_existing_directed_seam_evidence(case):
    data = list(_junction())
    if case == 'missing_edge':
        data[2][1] = np.empty((0, 3), dtype=np.int64)
    elif case == 'different_cell':
        data[3][2][0] = [9, 9, 9]
    elif case == 'unshared_neighbor':
        data[1][2][1] = -1
    elif case == 'reverse_relation':
        data[7] = {(2, 1)}
    elif case == 'already_shared':
        data[1][2][0] = 2
    else:
        data[6] = set()
    positions, ids, report = _run(data)
    assert report['fault_junction_attachment_count'] == 0
    assert report['fault_junction_rejected_count'] == 0
    for original, actual, original_ids, actual_ids in zip(data[0], positions, data[1], ids):
        np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(actual_ids, original_ids)


def test_new_attachment_cannot_supply_evidence_for_another_junction():
    data = _junction()
    vertices, ids, faces, cells, _, _, _, _ = data
    for surface in (0, 1):
        vertices[surface] = np.vstack([vertices[surface], [[.1, .2, 0]]])
        ids[surface] = np.append(ids[surface], 3)
        cells[surface] = np.vstack([cells[surface], [[1, 2, 2]]])
    vertices[2] = np.vstack([vertices[2], [[.1, .2, .05]]])
    ids[2] = np.append(ids[2], -1)
    cells[2] = np.vstack([cells[2], [[1, 2, 2]]])
    faces[1] = np.vstack([faces[1], [[0, 3, 2]]])
    faces[2] = np.vstack([faces[2], [[0, 3, 2]]])
    positions, final_ids, report = _run(data)
    assert report['fault_junction_attachment_count'] == 1
    assert final_ids[2][0] == 0
    assert final_ids[2][3] == -1
    np.testing.assert_array_equal(positions[2][3], vertices[2][3])


def test_competing_same_group_attachments_choose_nearest_original_without_transitive_merge():
    data = list(_junction())
    vertices, ids, faces, cells, surfaces, _, faults, truncation = data
    vertices[1] = np.vstack([vertices[1], [[-1, 0, 0]]])
    ids[1] = np.append(ids[1], 2)
    cells[1] = np.vstack([cells[1], [[1, 2, 2]]])
    faces[1] = np.vstack([faces[1], [[3, 0, 2]]])
    vertices.append(np.array([[0, .2, .1], [-1, 0, 0], [0, -1, .1]]))
    ids.append(np.array([-1, 2, -1]))
    cells.append(np.array([[2, 2, 2], [1, 2, 2], [2, 1, 1]]))
    faces.append(np.array([[0, 1, 2]]))
    surfaces.append((2, 1))
    data[5] = np.zeros((4, 4), dtype=bool)
    data[5][1, 2] = data[5][2, 1] = data[5][1, 3] = data[5][3, 1] = True
    truncation.add((1, 3))
    for order in permutations(range(4)):
        reordered = tuple([values[i] for i in order] for values in data[:5]) + (
            data[5][np.ix_(order, order)],
            {(order.index(a), order.index(b)) for a, b in faults},
            {(order.index(a), order.index(b)) for a, b in truncation},
        )
        positions, final_ids, report = _run(reordered)
        assert report['fault_junction_attachment_count'] == 1
        assert report['fault_junction_rejections'][0]['reason'] == 'same_group'
        assert final_ids[order.index(2)][0] == 0
        assert final_ids[order.index(3)][0] == -1
        np.testing.assert_array_equal(positions[order.index(3)], vertices[3])


@pytest.mark.parametrize("cross_allowed", [False, True])
def test_multiple_target_stacks_require_all_ordinary_cross_pairs(cross_allowed):
    data = list(_junction())
    vertices, ids, faces, cells, surfaces, _, _, truncation = data
    vertices.append(np.array([[0, .2, .1], [1, 0, 0], [0, -1, .1]]))
    ids.append(np.array([-1, 1, -1]))
    cells.append(np.array([[2, 2, 2], [3, 2, 2], [2, 1, 1]]))
    faces.append(np.array([[1, 0, 2]]))
    surfaces.append((3, 0))
    data[5] = np.zeros((4, 4), dtype=bool)
    data[5][1, 2] = data[5][2, 1] = data[5][1, 3] = data[5][3, 1] = True
    data[5][2, 3] = data[5][3, 2] = cross_allowed
    truncation.add((1, 3))
    positions, final_ids, report = _run(data)
    assert report['fault_junction_attachment_count'] == 1 + int(cross_allowed)
    assert final_ids[2][0] == 0
    if cross_allowed:
        assert report['fault_junction_rejected_count'] == 0
        assert final_ids[3][0] == 0
        np.testing.assert_array_equal(positions[3][0], vertices[0][0])
    else:
        assert report['fault_junction_rejections'][0]['reason'] == 'disallowed_cross_pair'
        assert final_ids[3][0] == -1


def test_attachment_uses_finalized_anchor_not_original_controller_position():
    data = _junction()
    vertices, ids, faces, cells, surfaces, allowed, faults, truncation = data
    originals = [value.copy() for value in vertices]
    for surface in (0, 1):
        vertices[surface][0] = [.02, 0, 0]
    positions, final_ids, report = attach_fault_junctions(
        originals, vertices, ids, faces, cells, surfaces, allowed, faults, truncation,
    )
    assert report['fault_junction_attachment_count'] == 1
    for surface in (0, 1):
        np.testing.assert_array_equal(positions[surface], vertices[surface])
        np.testing.assert_array_equal(final_ids[surface], ids[surface])
    np.testing.assert_array_equal(positions[2][0], vertices[1][0])
    assert not np.array_equal(positions[2][0], originals[1][0])
