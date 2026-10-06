"""Array-only voxel topology and explicit fault QEF eligibility contracts."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.dual_contouring.contact_topology import (
    build_contact_relations, contact_surface_roles, reconcile_cell_faces,
)
from gempy_engine.modules.dual_contouring import weighted_qef_setup_multicore as qef


def test_relations_boundary_thresholds_and_nonfault_eligibility():
    # Local order is intentionally unrelated to scalar ordering.
    args = ([0, 0, 1, 1, 2], [0, 1, 0, 1, 0], [R.ERODE, R.ERODE, R.BASEMENT],
            np.zeros((3, 3), bool), [np.array([8., 2.]), np.array([4., 1.]), np.array([0.])])
    before = deepcopy(args)
    allowed, faults, truncation = build_contact_relations(*args)
    assert faults == set()
    assert truncation == {(1, 2), (1, 3), (1, 4), (3, 4)}
    assert not allowed[0, 1] and not allowed[2, 3]
    assert allowed[0, 2] and allowed[1, 2]  # Grouping must not transitively merge A/A/B.
    np.testing.assert_array_equal(allowed, allowed.T)
    for actual, original in zip(args[:3], before[:3]):
        assert actual == original
    np.testing.assert_array_equal(args[3], before[3])
    for actual, original in zip(args[4], before[4]):
        np.testing.assert_array_equal(actual, original)


def test_onlap_uses_max_of_controller_even_when_controller_is_erode():
    _, _, pairs = build_contact_relations(
        [0, 1, 1, 2], [0, 0, 1, 0], [R.ONLAP, R.ERODE, R.BASEMENT], None,
        [[0.], [9., 2.], [0.]])
    # Stack 1 max controls stack 0; its min absorbs the max in exclusion of stack 2.
    assert pairs == {(1, 0), (2, 3)}


@pytest.mark.parametrize('relations, expected', [
    ([R.ONLAP, R.ONLAP, R.BASEMENT], {(1, 0), (2, 0), (2, 1)}),
    ([R.ONLAP, R.FAULT, R.BASEMENT], {(2, 0)}),
    ([R.ERODE, R.FAULT, R.BASEMENT], {(0, 2)}),
    ([R.ONLAP, R.ERODE, R.FAULT, R.BASEMENT], {(1, 0), (1, 3)}),
])
def test_mask_chain_and_fault_interruptions(relations, expected):
    n = len(relations)
    _, _, pairs = build_contact_relations(list(range(n)), [0] * n, relations, None, [[0.]] * n)
    assert pairs == expected


def test_directed_faults_all_surfaces_and_unrelated_fault_null_exclusion():
    faults = np.zeros((5, 5), bool)
    faults[0, 2] = True
    allowed, directed, truncated = build_contact_relations(
        [0, 0, 1, 2, 2, 3, 4], [0, 1, 0, 0, 1, 0, 0],
        [R.FAULT, R.FAULT, R.ERODE, R.NULL_SPACE, R.BASEMENT], faults,
        [[0., 1.], [0.], [0., 1.], [0.], [0.]])
    assert directed == {(0, 3), (0, 4), (1, 3), (1, 4)}
    assert not allowed[:3].any() and not allowed[5].any()
    assert allowed[3, 6] and allowed[4, 6]
    assert truncated == {(3, 6)}


def test_unexported_controller_boundary_is_not_replaced_by_other_surface():
    _, _, pairs = build_contact_relations([0, 1], [0, 0], [R.ERODE, R.BASEMENT],
                                         None, [[5., 1.], [0.]])
    assert pairs == set()


def test_nonexported_null_group_and_empty_surface_catalogue():
    allowed, faults, pairs = build_contact_relations(
        [1], [0], [R.NULL_SPACE, R.BASEMENT], None, [[], [.5]],
    )
    assert allowed.shape == (1, 1) and not allowed.any()
    assert not faults and not pairs
    allowed, faults, pairs = build_contact_relations([], [], [R.NULL_SPACE], None, [[]])
    assert allowed.shape == (0, 0) and not faults and not pairs


def test_surface_roles_exclude_fault_and_null_volume_ownership():
    faults = np.zeros((4, 4), dtype=bool)
    faults[1, 3] = True
    fault_interfaces, ordinary = contact_surface_roles(
        [0, 1, 2, 3], [R.FAULT, R.ERODE, R.NULL_SPACE, R.BASEMENT], faults,
    )
    np.testing.assert_array_equal(fault_interfaces, [True, True, False, False])
    np.testing.assert_array_equal(ordinary, [False, False, False, True])


def test_explicit_ownership_target_needs_no_exported_controller():
    triangles = np.array([[0, 1, 2]])
    result, report = reconcile_cell_faces(
        [triangles], [np.full(3, -1)], [np.zeros((3, 8), dtype=bool)], [[]], set(),
        ownership_targets=[0],
    )
    assert result[0].shape == (0, 3)
    assert report['per_surface'][0]['ownership_removed_count'] == 1
    with pytest.raises(ValueError, match='ownership target'):
        reconcile_cell_faces([triangles], [np.full(3, -1)], [np.zeros((3, 8), dtype=bool)],
                             [[]], set(), ownership_targets=[1])


@pytest.mark.parametrize('policy, expected', [('owned', 1), ('hidden', 0), ('mixed', 1)])
def test_corner_ownership_policy_including_raw(policy, expected):
    faces = [np.array([[2, 0, 1]], dtype=np.int32)]
    ownership = np.zeros((3, 8), bool)
    if policy == 'owned':
        ownership[:] = True  # RAW full ownership must not be treated as hidden.
    if policy == 'mixed':
        ownership[1, 6] = True
    controller_faces = np.zeros((0, 3), dtype=int)
    result, report = reconcile_cell_faces(
        faces + [controller_faces], [np.full(3, -1), np.zeros(0, dtype=int)],
        [ownership, np.zeros((0, 8), bool)], [[], []], {(1, 0)})
    assert len(result[0]) == expected
    assert report['removed_count'] == 1 - expected
    assert result[0].dtype == faces[0].dtype
    if expected:
        np.testing.assert_array_equal(result[0], faces[0])
    assert not np.shares_memory(result[0], faces[0])


def test_fault_all_not_any_directed_all_false_controller_ownership():
    triangles = np.array([[0, 1, 2], [0, 2, 3], [3, 4, 5]])
    faces = [triangles, triangles.copy()]
    result, report = reconcile_cell_faces(
        faces, [np.full(6, -1)] * 2, [np.zeros((6, 8), bool), np.ones((6, 8), bool)],
        [[], [0, 1, 2]], set())
    np.testing.assert_array_equal(result[0], triangles)
    np.testing.assert_array_equal(result[1], triangles[1:])
    assert report['per_surface'][1]['fault_removed_count'] == 1


def test_shared_patch_requires_complete_ids_and_retains_controller_winding():
    faces = [np.array([[2, 0, 1]]), np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])]
    ids = [np.array([10, 11, 12]), np.array([12, 10, 11, 13, -1])]
    ownership = [np.ones((3, 8), bool), np.ones((5, 8), bool)]
    overlap = [np.array([], dtype=int), np.array([], dtype=int)]
    before = deepcopy((faces, ids, ownership, overlap))
    result, report = reconcile_cell_faces(faces, ids, ownership, overlap, {(0, 1)})
    np.testing.assert_array_equal(result[0], faces[0])
    np.testing.assert_array_equal(result[1], faces[1][1:])
    assert report['per_surface'][1]['shared_patch_removed_count'] == 1
    for items, originals in zip((faces, ids, ownership, overlap), before):
        for actual, original in zip(items, originals):
            np.testing.assert_array_equal(actual, original)


def test_a_a_b_contact_subset_does_not_delete_competing_same_group_triangle():
    faces = [np.array([[0, 1, 2]]) for _ in range(3)]
    ids = [np.array([10, 11, 12]), np.array([-1, 11, 12]), np.array([12, 10, 11])]
    result, report = reconcile_cell_faces(faces, ids, [np.ones((3, 8), bool)] * 3,
                                         [[], [], []], {(2, 0), (2, 1)})
    assert [len(array) for array in result] == [0, 1, 1]
    assert report['removed_count'] == 1


def test_hidden_truncation_controller_cannot_remove_visible_target_copy():
    faces = [np.array([[0, 1, 2]])] * 2
    result, _ = reconcile_cell_faces(faces + [np.zeros((0, 3), dtype=int)],
                                     [np.arange(3)] * 2 + [np.zeros(0, dtype=int)],
                                     [np.zeros((3, 8), bool), np.ones((3, 8), bool), np.zeros((0, 8), bool)],
                                     [[], [], []], {(0, 1), (2, 0)})
    assert [len(array) for array in result] == [0, 1, 0]


def test_shared_patch_chain_keeps_root_and_is_pair_order_independent():
    faces = [np.array([[2, 0, 1]]) for _ in range(3)]
    args = (faces, [np.arange(3)] * 3, [np.ones((3, 8), bool)] * 3, [[], [], []])
    for pairs in ([(0, 1), (1, 2)], [(1, 2), (0, 1)]):
        result, report = reconcile_cell_faces(*args, pairs)
        assert [len(array) for array in result] == [1, 0, 0]
        assert report['removed_count'] == 2
        np.testing.assert_array_equal(result[0], faces[0])


def test_shared_patch_cycle_without_surviving_controller_is_rejected():
    with pytest.raises(ValueError, match='every copy'):
        reconcile_cell_faces([np.array([[0, 1, 2]])] * 2, [np.arange(3)] * 2,
                             [np.ones((3, 8), bool)] * 2, [[], []], {(0, 1), (1, 0)})


def test_empty_faces_and_invalid_metadata():
    empty = np.zeros((0, 3), dtype=int)
    result, report = reconcile_cell_faces([empty], [np.zeros(0, dtype=int)],
                                         [np.zeros((0, 8), bool)], [[]], set())
    assert result[0].shape == (0, 3) and report['removed_count'] == 0
    with pytest.raises(ValueError, match='ownership'):
        reconcile_cell_faces([empty], [np.zeros(0, dtype=int)], [np.zeros((0, 8))], [[]], set())
    with pytest.raises(ValueError, match='index'):
        reconcile_cell_faces([np.array([[0, 1, 3]])], [np.arange(3)],
                             [np.ones((3, 8), bool)], [[]], set())


def qef_inputs(n=4):
    edges = np.zeros((1, 12), bool)
    edges[0, 0] = True
    data = [SimpleNamespace(valid_voxels=np.array([True]), valid_edges=edges.copy(),
                            xyz_on_edge=np.full((1, 3), float(i)), gradients=np.ones((1, 3)),
                            extra_edge_xyz=None, extra_edge_normals=None, extra_weights=None)
            for i in range(n)]
    # Valid packed-cell input is immaterial here; isolate code generation only.
    return data, [np.array([[0, 0, 0]]) for _ in range(n)]


def test_qef_override_fault_only_both_directions_respects_empty_sets(monkeypatch):
    data, cells = qef_inputs()
    monkeypatch.setattr(qef, '_generate_voxel_codes', lambda *args: [np.array([0])] * 4)
    monkeypatch.setattr(qef, '_build_allowed_partners', lambda *args: pytest.fail('legacy helper called'))
    # Surface 2 shares group with target 1; surface 3 is unrelated nonfault.
    partners = [{1}, {0}, set(), set()]
    before = deepcopy(partners)
    qef.find_and_inject_multi_surface_constraints_multicore(
        data, cells, (2, 2, 2), max_workers=1, surface_to_stack=[0, 1, 1, 2],
        allowed_partners_per_surface=partners)
    assert partners == before
    for i, source in ((0, 1), (1, 0)):
        assert data[i].extra_edge_xyz.shape == (1, 12, 3)
        np.testing.assert_array_equal(data[i].extra_edge_xyz[0, 0], np.full(3, source))
    assert data[2].extra_edge_xyz is None and data[3].extra_edge_xyz is None


@pytest.mark.parametrize('n', [0, 1, 2])
def test_qef_override_length_validated_before_early_return(n):
    with pytest.raises(ValueError, match='partner set'):
        qef.find_and_inject_multi_surface_constraints_multicore([None] * n, [], (2, 2, 2),
                                                               allowed_partners_per_surface=[set()] * (n + 1))


def test_qef_default_none_retains_legacy_helper_and_numerics(monkeypatch):
    monkeypatch.setattr(qef, '_generate_voxel_codes', lambda *args: [np.array([0])] * 4)
    default_data, cells = qef_inputs()
    explicit_data, _ = qef_inputs()
    qef.find_and_inject_multi_surface_constraints_multicore(default_data, cells, (2, 2, 2), max_workers=1)
    qef.find_and_inject_multi_surface_constraints_multicore(
        explicit_data, cells, (2, 2, 2), max_workers=1, allowed_partners_per_surface=None)
    for a, b in zip(default_data, explicit_data):
        assert a.extra_edge_xyz.shape == (1, 36, 3)
        for attr in ('extra_edge_xyz', 'extra_edge_normals', 'extra_weights'):
            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
