"""CPU analytic contracts for branch-aware Hermite/QEF dual extraction."""

from collections import Counter

import numpy as np
import pytest

from gempy_engine.API.dual_contouring.topology_aware_extraction import extract_topology_aware
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices
from gempy_engine.modules.dual_contouring.quad_triangulation import triangulate_quads
from gempy_engine.modules.dual_contouring.topology_extraction import triangle_quality


def model(axes, fields, gradients, groups=None, indices=None, levels=None,
          relations=None, ownership=None, **kwargs):
    n = len(fields)
    groups = list(range(n)) if groups is None else groups
    ns = max(groups) + 1
    indices = [groups[:i].count(g) for i, g in enumerate(groups)] if indices is None else indices
    levels = [np.zeros(groups.count(g)) for g in range(ns)] if levels is None else levels
    samples = np.array(fields)
    gradients = np.array([np.broadcast_to(g, (*samples.shape[1:], 3)) for g in gradients])
    return extract_topology_aware(
        axes, samples, groups, indices,
        [R.ERODE] * (ns - 1) + [R.BASEMENT] if relations is None else relations,
        levels, ownership=np.ones_like(samples) if ownership is None else ownership,
        gradient_samples=gradients, **kwargs,
    )


def edges(faces):
    return Counter(tuple(sorted((int(a), int(b)))) for f in faces
                   for a, b in zip(f, np.roll(f, 1)))


def connected_components(edge_set):
    neighbors = {}
    for a, b in edge_set:
        neighbors.setdefault(a, set()).add(b)
        neighbors.setdefault(b, set()).add(a)
    remaining = set(neighbors)
    components = []
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            vertex = pending.pop()
            if vertex not in component:
                component.add(vertex)
                pending.extend(neighbors[vertex] - component)
        remaining -= component
        components.append(component)
    return components


def assert_oriented(result, gradients):
    for faces, gradient in zip(result['faces'], gradients):
        assert len(faces)
        points = result['vertices'][faces]
        normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
        assert np.all(np.linalg.norm(normals, axis=1) > 1e-12)
        reference = gradient(points.mean(axis=1)) if callable(gradient) else gradient
        assert np.all(np.sum(normals * reference, axis=1) > 1e-12)


def assert_seam(result, controller, target):
    shared = set(np.intersect1d(result['faces'][controller], result['faces'][target]))
    seam = {tuple(sorted(map(int, edge))) for edge in result['seam_edges']}
    assert len(seam) == 7  # Eight joint cells, with open complete-quad cropping.
    assert shared == set(result['seam_edges'].ravel())
    counts_controller, counts_target = edges(result['faces'][controller]), edges(result['faces'][target])
    assert all(counts_controller[e] == 2 and counts_target[e] == 1 for e in seam)
    assert {e for e in counts_target if set(e) <= shared} == seam
    assert len(connected_components(seam)) == 1
    degrees = Counter(v for edge in seam for v in edge)
    assert sorted(degrees.values()) == [1, 1] + [2] * 6
    assert all(result['vertex_keys'][i][0] == 'joint' for i in shared)
    assert result['diagnostics']['post_reconciliation'] is False
    return result['vertices'][sorted(shared)]


def assert_unaffected_reference(result):
    reference = result['reference']
    reference_ids = {key: i for i, key in enumerate(reference['vertex_keys'])}
    for faces, ref_faces, affected, ref_affected in zip(
            result['faces'], reference['faces'], result['affected_faces'], reference['affected_faces']):
        assert affected.any() and (~affected).any()
        np.testing.assert_array_equal(affected, ref_affected)
        # Global IDs change when joint keys replace regular keys; compare exact
        # oriented faces through stable identities, not coincidental ID offsets.
        mapped = np.array([[reference_ids[result['vertex_keys'][i]] for i in face]
                           for face in faces[~affected]])
        np.testing.assert_array_equal(mapped, ref_faces[~affected])
        np.testing.assert_array_equal(result['vertices'][faces[~affected]],
                                      reference['vertices'][ref_faces[~affected]])
    for i, key in enumerate(result['vertex_keys']):
        if key[0] == 'regular':
            np.testing.assert_array_equal(result['vertices'][i], reference['vertices'][reference_ids[key]])
    # Also compare against native production triangulate_quads, independently
    # of the experimental emitter used for the optional support-matched reference.
    for surface, data in enumerate(result['hermite_data']):
        native_vertices = generate_dual_contouring_vertices(data)
        dense_normals = np.zeros((*data.valid_edges.shape, 3))
        dense_normals[data.valid_edges] = data.gradients
        native_faces = triangulate_quads(data.left_right_codes, data.valid_edges, dense_normals,
                                         native_vertices, data.base_number)
        native_face_set = set(map(tuple, native_faces))
        cell_to_native = {tuple(c): i for i, c in enumerate(data.left_right_codes[data.valid_voxels])}
        for face in result['faces'][surface][~result['affected_faces'][surface]]:
            native = [cell_to_native[result['vertex_keys'][i][2]] for i in face]
            assert tuple(native) in native_face_set
            np.testing.assert_array_equal(result['vertices'][face], native_vertices[native])


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('sign', [-1, 1])
def test_plane_matches_direct_production_hermite_qef_and_quads(axis, sign):
    axes = [np.linspace(0, 1, 9)] * 3
    xyz = np.meshgrid(*axes, indexing='ij')
    normal = np.eye(3)[axis] * sign
    result = model(axes, [sign * (xyz[axis] - .43)], [normal])
    data = result['hermite_data'][0]
    vertices = generate_dual_contouring_vertices(data)
    normals = np.zeros((*data.valid_edges.shape, 3))
    normals[data.valid_edges] = data.gradients
    faces = triangulate_quads(data.left_right_codes, data.valid_edges, normals, vertices,
                              data.base_number, generated_coordinates=data.generated_cell_coordinates)
    np.testing.assert_array_equal(result['vertices'], vertices)
    np.testing.assert_array_equal(result['faces'][0], faces)
    np.testing.assert_allclose(vertices[:, axis], .43, atol=1e-12)
    assert len(vertices) == 64 and len(faces) == 98
    assert not result['affected_faces'][0].any()
    assert result['diagnostics']['algorithm'] == 'hermite_qef_dual_contouring'
    assert_oriented(result, [normal])


def test_erosion_shared_seam_and_exact_unaffected_reference():
    axes = [np.linspace(0, 1, 9)] * 3
    x, _, z = np.meshgrid(*axes, indexing='ij')
    result = model(axes, [x - .57, z - .43], [(1, 0, 0), (0, 0, 1)],
                   relations=[R.ERODE, R.BASEMENT], ownership=[np.ones_like(x), .57 - x],
                   include_reference=True)
    points = assert_seam(result, controller=0, target=1)
    np.testing.assert_allclose(points[:, 0], .57, atol=1e-12)
    np.testing.assert_allclose(points[:, 2], .43, atol=1e-12)
    np.testing.assert_allclose(np.sort(points[:, 1]), (np.arange(8) + .5) / 8, atol=1e-12)
    assert np.all(result['vertices'][result['faces'][1], 0] <= .57 + 1e-12)
    assert_unaffected_reference(result)
    assert_oriented(result, [(1, 0, 0), (0, 0, 1)])


def test_onlap_shared_seam_and_exact_unaffected_reference():
    axes = [np.linspace(0, 1, 9)] * 3
    x, _, z = np.meshgrid(*axes, indexing='ij')
    f0, f1 = x - .25 * z, .25 * x + z
    result = model(axes, [f0, f1], [(1, 0, -.25), (.25, 0, 1)], levels=[[.4], [.6]],
                   relations=[R.ONLAP, R.BASEMENT], ownership=[f1 - .6, np.ones_like(x)],
                   include_reference=True)
    points = assert_seam(result, controller=1, target=0)
    np.testing.assert_allclose(points[:, 0] - .25 * points[:, 2], .4, atol=1e-12)
    np.testing.assert_allclose(.25 * points[:, 0] + points[:, 2], .6, atol=1e-12)
    target = result['vertices'][result['faces'][0]]
    assert np.all(.25 * target[..., 0] + target[..., 2] >= .6 - 1e-12)
    assert_unaffected_reference(result)
    assert_oriented(result, [(1, 0, -.25), (.25, 0, 1)])


def test_disjoint_interfaces_and_unrelated_crossing_are_not_welded():
    axes = [np.linspace(0, 1, 9)] * 3
    x, _, z = np.meshgrid(*axes, indexing='ij')
    for fields, gradients, relations in [
            ([z - .43, z - .46], [(0, 0, 1)] * 2, [R.ERODE, R.BASEMENT]),
            ([x - .57, z - .43], [(1, 0, 0), (0, 0, 1)], [R.NULL_SPACE, R.BASEMENT])]:
        result = model(axes, fields, gradients, relations=relations)
        assert not np.intersect1d(*result['faces']).size
        assert not result['seam_edges'].size
        assert not result['junction_cells'].size
        assert all(not mask.any() for mask in result['affected_faces'])
        assert_oriented(result, gradients)


@pytest.mark.parametrize('distance', [.03, 1e-7])
def test_same_stack_near_horizons_remain_distinct(distance):
    axes = [np.linspace(0, 1, 9)] * 3
    _, _, z = np.meshgrid(*axes, indexing='ij')
    result = model(axes, [z, z], [(0, 0, 1)] * 2, groups=[0, 0],
                   levels=[[.43, .43 + distance]])
    assert not np.intersect1d(*result['faces']).size
    assert not result['seam_edges'].size
    for surface, level in enumerate([.43, .43 + distance]):
        np.testing.assert_allclose(result['vertices'][result['faces'][surface], 2], level, atol=1e-12)
    assert_oriented(result, [(0, 0, 1)] * 2)


def test_only_eligible_boundary_contacts_other_stack_not_near_horizon():
    axes = [np.linspace(0, 1, 9)] * 3
    x, _, z = np.meshgrid(*axes, indexing='ij')
    result = model(axes, [z, z, x], [(0, 0, 1), (0, 0, 1), (1, 0, 0)],
                   groups=[0, 0, 1], levels=[[.43, .4300001], [.57]],
                   ownership=[np.ones_like(x), np.ones_like(x), .43 - z])
    assert_seam(result, controller=0, target=2)
    assert not np.intersect1d(result['faces'][0], result['faces'][1]).size
    assert not np.intersect1d(result['faces'][1], result['faces'][2]).size
    assert all(len({identity[0] for identity in key[1]}) == len(key[1])
               for key in result['vertex_keys'])


def test_surface_reordering_preserves_stable_global_ids():
    axes = [np.linspace(0, 1, 9)] * 3
    x, _, z = np.meshgrid(*axes, indexing='ij')
    a = model(axes, [x - .57, z - .43], [(1, 0, 0), (0, 0, 1)],
              ownership=[np.ones_like(x), .57 - x], include_reference=True)
    b = model(axes, [z - .43, x - .57], [(0, 0, 1), (1, 0, 0)], groups=[1, 0], indices=[0, 0],
              ownership=[.57 - x, np.ones_like(x)], include_reference=True)
    assert a['vertex_keys'] == b['vertex_keys']
    np.testing.assert_array_equal(a['vertices'], b['vertices'])
    np.testing.assert_array_equal(a['seam_edges'], b['seam_edges'])
    for key in ['faces', 'affected_faces', 'primal_edges', 'cell_components', 'branch_labels']:
        for i in range(2):
            np.testing.assert_array_equal(a[key][i], b[key][1 - i])
    assert a['reference']['vertex_keys'] == b['reference']['vertex_keys']
    np.testing.assert_array_equal(a['reference']['vertices'], b['reference']['vertices'])


def test_bilinear_extrusion_has_two_branch_specific_qefs_and_disconnected_topology():
    axes = [np.array([-1., 0., 1., 2.]), np.array([-1., 0., 1., 2.]),
            np.array([0., .5, 1., 1.5])]
    x, y, z = np.meshgrid(*axes, indexing='ij')
    field = (x - .5) * (y - .5) - .1
    gradient = np.stack((y - .5, x - .5, np.zeros_like(z)), axis=-1)
    result = model(axes, [field], [gradient])
    repeated = model(axes, [field.copy()], [gradient.copy()])
    np.testing.assert_array_equal(result['vertices'], repeated['vertices'])
    np.testing.assert_array_equal(result['faces'][0], repeated['faces'][0])
    assert result['diagnostics']['ambiguous_face_count'] > 0
    data = result['hermite_data'][0]
    dense_xyz = np.zeros((*data.valid_edges.shape, 3))
    dense_normals = np.zeros_like(dense_xyz)
    dense_xyz[data.valid_edges] = data.xyz_on_edge
    dense_normals[data.valid_edges] = data.gradients
    ids = []
    for cell in np.flatnonzero(np.all(result['cell_coordinates'][:, :2] == [1, 1], axis=1)):
        assert result['cell_components'][0, cell] == 2
        labels = result['branch_labels'][0, cell]
        assert set(labels[data.valid_edges[cell]]) == {0, 1}
        branch_ids = []
        for branch in range(2):
            mask = labels == branch
            assert mask.sum() == 4
            # Independent analytic oracle: the xy=.1 hyperbola has one branch
            # southwest and one northeast of (.5,.5), never southeast/northwest.
            assert set(np.flatnonzero(mask)) in ({0, 1, 4, 5}, {2, 3, 6, 7})
            points, normals = dense_xyz[cell, mask], dense_normals[cell, mask]
            np.testing.assert_allclose((points[:, 0] - .5) * (points[:, 1] - .5), .1, atol=1e-12)
            np.testing.assert_allclose(normals, np.column_stack((points[:, 1] - .5,
                                                                points[:, 0] - .5, np.zeros(4))))
            branch_data = DualContouringData(points, mask[None], data.xyz_on_centers[cell:cell + 1],
                                             data.dxdydz, 1, data.left_right_codes[cell:cell + 1],
                                             gradients=normals, strict_crossings=True)
            expected = generate_dual_contouring_vertices(branch_data)[0]
            key = ('regular', ((0, 0),), tuple(result['cell_coordinates'][cell]), branch)
            vertex_id = result['vertex_keys'].index(key)
            np.testing.assert_array_equal(result['vertices'][vertex_id], expected)
            assert vertex_id in result['faces'][0]
            branch_ids.append(vertex_id)
        assert np.linalg.norm(result['vertices'][branch_ids[0]] - result['vertices'][branch_ids[1]]) > .1
        ids.append(branch_ids)
    components = connected_components(edges(result['faces'][0]))
    assert len(components) == 2
    assert all(not any(set(pair) <= component for component in components) for pair in ids)
    assert_oriented(result, [lambda p: np.column_stack((p[:, 1] - .5, p[:, 0] - .5, np.zeros(len(p))))])


def test_arbitrary_nonaffine_interior_is_explicitly_unsupported():
    axes = [np.array([0., 1.])] * 3
    x, y, z = np.meshgrid(*axes, indexing='ij')
    field = x * y * z - .2
    gradients = np.stack((y * z, x * z, x * y), axis=-1)
    with pytest.raises(ValueError, match='insufficient_interior_topology'):
        model(axes, [field], [gradients])
    with pytest.raises(ValueError, match='insufficient_interior_topology'):
        model(axes, [field], [gradients], interior_samples=np.array([[[[.125 - .2]]]]))


def test_inactive_corner_samples_do_not_certify_hidden_interior_topology():
    axes = [np.array([0., 1.])] * 3
    samples = np.ones((2, 2, 2))
    result = model(axes, [samples], [(0, 0, 0)])
    assert not result['faces'][0].size
    assert result['diagnostics']['interior_topology'] == 'sampled_boundary_only_not_interior_certified'
    with pytest.raises(ValueError, match='insufficient_interior_topology'):
        model(axes, [samples], [(0, 0, 0)], interior_samples=-np.ones((1, 1, 1, 1)))


@pytest.mark.parametrize('offset, expected_pairs', [
    (-.1, [{0, 1, 4, 5}, {2, 3, 6, 7}]),
    (.1, [{0, 1, 6, 7}, {2, 3, 4, 5}]),
])
def test_bilinear_face_decider_matches_both_analytic_hyperbola_branches(offset, expected_pairs):
    axes = [np.array([0., 1.])] * 3
    x, y, z = np.meshgrid(*axes, indexing='ij')
    field = (x - .5) * (y - .5) + offset
    gradient = np.stack((y - .5, x - .5, np.zeros_like(z)), axis=-1)
    result = model(axes, [field], [gradient])
    labels = result['branch_labels'][0, 0]
    actual = [set(np.flatnonzero(labels == b)) for b in range(2)]
    assert all(pair in actual for pair in expected_pairs)


def test_joint_vertex_minimizes_combined_hermite_qef_with_hard_constraints():
    axes = [np.linspace(0, 1, 9)] * 3
    x, _, z = np.meshgrid(*axes, indexing='ij')
    f0 = 2 * (x - .57) + .7 * (z - .43)
    f1 = .4 * (x - .57) + 3 * (z - .43)
    result = model(axes, [f0, f1], [(2, 0, .7), (.4, 0, 3)],
                   ownership=[np.ones_like(x), -f0])
    hard = np.array([[2., 0, .7], [.4, 0, 3.]])
    levels = hard @ [.57, 0, .43]
    for cell in result['junction_cells']:
        points, normals = [], []
        for data in result['hermite_data']:
            dense_points = np.zeros((*data.valid_edges.shape, 3))
            dense_normals = np.zeros_like(dense_points)
            dense_points[data.valid_edges], dense_normals[data.valid_edges] = data.xyz_on_edge, data.gradients
            points.extend(dense_points[cell, data.valid_edges[cell]])
            normals.extend(dense_normals[cell, data.valid_edges[cell]])
        points, normals = np.array(points), np.array(normals)
        mass = points.mean(axis=0)
        a = np.concatenate((normals, np.eye(3)))
        b = np.concatenate((np.sum(normals * points, axis=1), mass))
        kkt = np.block([[a.T @ a, hard.T], [hard, np.zeros((2, 2))]])
        solution = np.linalg.solve(kkt, np.concatenate((a.T @ b, levels)))[:3]
        key = ('joint', ((0, 0), (1, 0)), tuple(result['cell_coordinates'][cell]), 0)
        vertex = result['vertices'][result['vertex_keys'].index(key)]
        np.testing.assert_allclose(vertex, solution, atol=1e-12)


def test_bilinear_junction_still_needs_resolvable_shared_face_approximants():
    axes = [np.array([-1., 0., 1., 2.])] * 3
    x, y, z = np.meshgrid(*axes, indexing='ij')
    field = (x - .5) * (y - .5) - .1
    gradient = np.stack((y - .5, x - .5, np.zeros_like(z)), axis=-1)
    with pytest.raises(ValueError, match='unsupported_junction_branches|unsupported_junction_field'):
        model(axes, [field, z - .43], [gradient, (0, 0, 1)],
              ownership=[np.ones_like(x), -field])


def test_ambiguous_saddle_tie_and_coincident_joint_constraints_are_explicit():
    axes = [np.array([0., 1.])] * 3
    x, y, z = np.meshgrid(*axes, indexing='ij')
    gradient = np.stack((y - .5, x - .5, np.zeros_like(z)), axis=-1)
    with pytest.raises(ValueError, match='ambiguous_face_tie'):
        model(axes, [(x - .5) * (y - .5)], [gradient])
    with pytest.raises(ValueError, match='coincident_interfaces'):
        model(axes, [x - .43, x - .43], [(1, 0, 0)] * 2,
              ownership=[np.ones_like(x), .43 - x])


def test_same_cell_crossings_without_an_in_cell_intersection_do_not_join():
    axes = [np.linspace(0, 1, 9)] * 3
    x, y, z = np.meshgrid(*axes, indexing='ij')
    controller = x + y - .303
    target = x + y + .1 * z - .4473
    result = model(axes, [controller, target], [(1, 1, 0), (1, 1, .1)],
                   ownership=[np.ones_like(x), controller], include_reference=True)
    active = [data.valid_voxels for data in result['hermite_data']]
    assert np.any(active[0] & active[1])
    assert not result['junction_cells'].size
    assert not np.intersect1d(*result['faces']).size


def test_competing_three_interface_cell_and_insufficient_seam_support_reject():
    axes = [np.linspace(0, 1, 9)] * 3
    x, y, z = np.meshgrid(*axes, indexing='ij')
    with pytest.raises(ValueError, match='unsupported_multiway_junction'):
        model(axes, [x - .57, y - .53, z - .43], [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
              relations=[R.ERODE, R.ERODE, R.BASEMENT],
              ownership=[np.ones_like(x), .57 - x, .53 - y])
    axes = [np.array([0., 1.])] * 3
    x, _, z = np.meshgrid(*axes, indexing='ij')
    with pytest.raises(ValueError, match='insufficient_dual_seam_support'):
        model(axes, [x - .57, z - .43], [(1, 0, 0), (0, 0, 1)],
              ownership=[np.ones_like(x), .57 - x])


def test_experimental_generation_never_calls_contact_mesh_reconciliation(monkeypatch):
    from gempy_engine.API.dual_contouring import contact_reconciliation

    def forbidden(*args, **kwargs):
        pytest.fail('Post-mesh reconciliation must not implement joint dual generation')

    monkeypatch.setattr(contact_reconciliation, 'reconcile_contact_meshes', forbidden)
    axes = [np.linspace(0, 1, 9)] * 3
    x, _, z = np.meshgrid(*axes, indexing='ij')
    result = model(axes, [x - .57, z - .43], [(1, 0, 0), (0, 0, 1)],
                   ownership=[np.ones_like(x), .57 - x])
    assert_seam(result, 0, 1)


def test_mixed_boolean_ownership_is_explicitly_unsupported():
    axes = [np.linspace(0, 1, 9)] * 3
    x, _, z = np.meshgrid(*axes, indexing='ij')
    with pytest.raises(ValueError, match='unsupported_ownership'):
        model(axes, [x - .57, z - .43], [(1, 0, 0), (0, 0, 1)],
              ownership=[np.ones_like(x, dtype=bool), x < .57])


@pytest.mark.parametrize('direction', [None, (0, 1), (1, 0)])
@pytest.mark.parametrize('finite_metadata', [False, True])
def test_fault_directions_are_always_unsupported(direction, finite_metadata):
    axes = [np.linspace(0, 1, 9)] * 3
    x, y, z = np.meshgrid(*axes, indexing='ij')
    faults = np.zeros((2, 2), dtype=bool)
    if direction is not None:
        faults[direction] = True
    kwargs = dict(fault_support={0: 1 - y}, fault_slip={0: np.zeros_like(y)}) if finite_metadata else {}
    with pytest.raises(ValueError, match='unsupported_fault_extraction'):
        model(axes, [x - .57, z - .43], [(1, 0, 0), (0, 0, 1)],
              relations=[R.FAULT, R.BASEMENT], faults_relations=faults, **kwargs)


def test_unexported_directed_fault_controller_is_not_silently_ignored():
    axes = [np.linspace(0, 1, 9)] * 3
    _, _, z = np.meshgrid(*axes, indexing='ij')
    with pytest.raises(ValueError, match='unsupported_fault_extraction'):
        model(axes, [z - .43], [(0, 0, 1)], groups=[1], indices=[0],
              levels=[[], [0.]], relations=[R.BASEMENT, R.BASEMENT],
              faults_relations=np.array([[False, True], [False, False]]))


def test_grid_edge_junction_requires_extra_support():
    axes = [np.linspace(0, 1, 9)] * 3
    x, _, z = np.meshgrid(*axes, indexing='ij')
    controller, target = x + z - .7, x - z - .7
    with pytest.raises(ValueError, match='grid_edge_junction'):
        model(axes, [controller, target], [(1, 0, 1), (1, 0, -1)],
              ownership=[np.ones_like(x), -controller])


def test_input_immutability_and_cell_bound():
    axes = [np.linspace(0, 1, 9).copy() for _ in range(3)]
    x, _, z = np.meshgrid(*axes, indexing='ij')
    samples = np.array([x, z])
    ownership = np.array([np.ones_like(x), .57 - x])
    gradients = np.broadcast_to(np.array([(1, 0, 0), (0, 0, 1)])[:, None, None, None],
                               (*samples.shape, 3)).copy()
    groups, indices = np.array([0, 1]), np.array([0, 0])
    levels = [np.array([.57]), np.array([.43])]
    centers = (samples - np.array([.57, .43])[:, None, None, None])
    centers = sum(centers[:, a:a + 8, b:b + 8, c:c + 8]
                  for a in range(2) for b in range(2) for c in range(2)) / 8
    inputs = [*axes, samples, ownership, gradients, groups, indices, *levels, centers]
    originals = [a.copy() for a in inputs]
    result = extract_topology_aware(axes, samples, groups, indices, [R.ERODE, R.BASEMENT], levels,
                                    ownership=ownership, gradient_samples=gradients,
                                    interior_samples=centers, include_reference=True)
    assert_seam(result, 0, 1)
    assert result['diagnostics']['interior_topology'] == 'center_checked_not_topology_certified'
    for array, original in zip(inputs, originals):
        np.testing.assert_array_equal(array, original)
    axes = [np.linspace(0, 1, 18)] * 3
    _, _, z = np.meshgrid(*axes, indexing='ij')
    with pytest.raises(ValueError, match='unsupported_bound'):
        model(axes, [z - .43], [(0, 0, 1)])
    axes = [np.linspace(0, 1, 17)] * 3
    _, _, z = np.meshgrid(*axes, indexing='ij')
    assert model(axes, [z - .43], [(0, 0, 1)])['faces'][0].size


def test_triangle_quality_numeric_formula_and_percentiles():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [.5, np.sqrt(3) / 2, 0.],
                         [0., 0., 0.], [3., 0., 0.], [0., 4., 0.]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    quality = triangle_quality(vertices, faces)
    minimum_angles = [60., np.degrees(np.arctan(3 / 4))]
    # Longest edge / smallest altitude = longest edge squared / twice area.
    aspects = [2 / np.sqrt(3), 25 / 12]
    assert quality['count'] == 2
    assert quality['percentile_levels'] == [0, 5, 50, 95, 100]
    assert quality['minimum_angle_degrees'] == pytest.approx(min(minimum_angles))
    np.testing.assert_allclose(quality['angle_percentiles'], np.percentile(minimum_angles, [0, 5, 50, 95, 100]))
    np.testing.assert_allclose(quality['aspect_percentiles'], np.percentile(aspects, [0, 5, 50, 95, 100]))
    scaled = triangle_quality(vertices * 7 + [2, -3, 5], faces[:, ::-1])
    np.testing.assert_allclose(scaled['angle_percentiles'], quality['angle_percentiles'])
    np.testing.assert_allclose(scaled['aspect_percentiles'], quality['aspect_percentiles'])
    empty = triangle_quality(vertices, np.empty((0, 3), dtype=int))
    assert empty == dict(count=0, minimum_angle_degrees=None, angle_percentiles=None, aspect_percentiles=None)


@pytest.fixture
def curved_input():
    axes = [np.linspace(0, 1, 9)] * 3
    x, y, z = np.meshgrid(*axes, indexing='ij')
    controller = z - .43 - .25 * (x - .5) ** 2 - .15 * (y - .5) ** 2
    gradient = np.stack((-.5 * (x - .5), -.3 * (y - .5), np.ones_like(z)), axis=-1)
    fields = np.array([controller, x - .57])
    gradients = np.array([gradient, np.broadcast_to((1, 0, 0), gradient.shape)])
    return axes, fields, gradients, np.array([np.ones_like(z), -controller])


@pytest.mark.parametrize('cross_term', [False, True])
def test_ordinary_curved_cells_use_actual_normals_and_exact_production_qef(cross_term):
    axes = [np.linspace(0, 1, 9)] * 3
    x, y, z = np.meshgrid(*axes, indexing='ij')
    field = z - .4317 - .15 * (x - .5) ** 2 - .1 * (y - .5) ** 2
    gradient = np.stack((-.3 * (x - .5), -.2 * (y - .5), np.ones_like(z)), axis=-1)
    if cross_term:
        field = field - .1 * x * y
        gradient = gradient - np.stack((.1 * y, .1 * x, np.zeros_like(z)), axis=-1)
    result = model(axes, [field], [gradient])
    data = result['hermite_data'][0]
    assert np.ptp(data.gradients[:, 0]) > .1
    assert np.ptp(data.gradients[:, 1]) > .1
    points = data.xyz_on_edge
    actual = np.column_stack((-.3 * (points[:, 0] - .5), -.2 * (points[:, 1] - .5), np.ones(len(points))))
    if cross_term:
        actual -= np.column_stack((.1 * points[:, 1], .1 * points[:, 0], np.zeros(len(points))))
    np.testing.assert_allclose(data.gradients, actual, atol=1e-12)
    native_vertices = generate_dual_contouring_vertices(data)
    dense_normals = np.zeros((*data.valid_edges.shape, 3))
    dense_normals[data.valid_edges] = data.gradients
    native_faces = triangulate_quads(data.left_right_codes, data.valid_edges, dense_normals,
                                     native_vertices, data.base_number)
    np.testing.assert_array_equal(result['vertices'], native_vertices)
    np.testing.assert_array_equal(result['faces'][0], native_faces)
    assert np.ptp(result['vertices'][result['faces'][0], 2]) > .02
    assert not result['affected_faces'][0].any()
    assert result['diagnostics']['hermite_normals'] == 'supplied_interpolated_not_forced_to_corner_fit'


def test_curved_erosion_has_real_seam_winding_no_duplicates_and_ordinary_parity(curved_input):
    axes, fields, gradients, ownership = curved_input
    result = model(axes, fields, gradients, ownership=ownership, include_reference=True)
    assert [len(faces) for faces in result['faces']] == [122, 42]
    points = assert_seam(result, 0, 1)
    np.testing.assert_allclose(points[:, 0], .57, atol=1e-12)
    # Approximate sampled seam, not exact analytic paraboloid projection.
    residual = points[:, 2] - .43 - .25 * (points[:, 0] - .5) ** 2 - .15 * (points[:, 1] - .5) ** 2
    assert np.max(np.abs(residual)) > 1e-4
    assert_unaffected_reference(result)
    assert_oriented(result, [lambda p: np.column_stack((-.5 * (p[:, 0] - .5),
                                                        -.3 * (p[:, 1] - .5), np.ones(len(p)))), (1, 0, 0)])
    face_sets = []
    for faces in result['faces']:
        keys = [tuple(sorted(map(int, triangle))) for triangle in faces]
        assert len(set(keys)) == len(keys)
        face_sets.append(set(keys))
        assert max(edges(faces).values()) <= 2
    assert not face_sets[0] & face_sets[1]
    assert sum(mask.sum() for mask in result['affected_faces']) == 42
    assert sum((~mask).sum() for mask in result['affected_faces']) == 122


def test_curved_output_and_ids_are_deterministic_under_surface_reordering(curved_input):
    axes, fields, gradients, ownership = curved_input
    a = model(axes, fields, gradients, ownership=ownership)
    repeated = model(axes, fields.copy(), gradients.copy(), ownership=ownership.copy())
    b = model(axes, fields[::-1], gradients[::-1], ownership=ownership[::-1],
              groups=[1, 0], indices=[0, 0])
    assert a['vertex_keys'] == b['vertex_keys'] == repeated['vertex_keys']
    np.testing.assert_array_equal(a['vertices'], b['vertices'])
    np.testing.assert_array_equal(a['vertices'], repeated['vertices'])
    np.testing.assert_array_equal(a['seam_edges'], b['seam_edges'])
    for i in range(2):
        np.testing.assert_array_equal(a['faces'][i], b['faces'][1 - i])
        np.testing.assert_array_equal(a['faces'][i], repeated['faces'][i])


def test_curved_same_group_horizons_do_not_merge_directly_or_transitively(curved_input):
    axes, fields, gradients, ownership = curved_input
    result = model(axes, [fields[0], fields[0], fields[1]],
                   [gradients[0], gradients[0], gradients[1]], groups=[0, 0, 1],
                   levels=[[0., .003], [0.]], ownership=[ownership[0], ownership[0], ownership[1]])
    assert_seam(result, 0, 2)
    assert not np.intersect1d(result['faces'][0], result['faces'][1]).size
    assert not np.intersect1d(result['faces'][1], result['faces'][2]).size
    assert all(len({identity[0] for identity in key[1]}) == len(key[1]) for key in result['vertex_keys'])


@pytest.mark.parametrize('scalar_weight', [1., 10.])
def test_curved_joint_qef_optimality_uses_noncoincident_tangent_rows(curved_input, scalar_weight):
    from gempy_engine.modules.dual_contouring.topology_extraction import CORNERS

    axes, fields, gradients, ownership = curved_input
    fields, gradients, ownership = fields.copy(), gradients.copy(), ownership.copy()
    fields[0] *= scalar_weight
    gradients[0] *= scalar_weight
    ownership[1] *= scalar_weight
    result = model(axes, fields, gradients, ownership=ownership)
    improvements, deviations = [], []
    for cell in result['junction_cells']:
        coordinate = result['cell_coordinates'][cell]
        nodes = coordinate + CORNERS
        xyz = np.array([[axes[a][node[a]] for a in range(3)] for node in nodes])
        sample_values = fields[:, nodes[:, 0], nodes[:, 1], nodes[:, 2]]
        # Independent world-coordinate corner approximants specify the same
        # constrained line. The objective below uses the ACTUAL Hermite rows.
        coefficients = np.linalg.lstsq(np.column_stack((xyz, np.ones(8))), sample_values.T, rcond=None)[0].T
        hard, rhs = coefficients[:, :3], -coefficients[:, 3]
        points, normals = [], []
        for data in result['hermite_data']:
            dense_points = np.zeros((*data.valid_edges.shape, 3))
            dense_normals = np.zeros_like(dense_points)
            dense_points[data.valid_edges], dense_normals[data.valid_edges] = data.xyz_on_edge, data.gradients
            points.extend(dense_points[cell, data.valid_edges[cell]])
            normals.extend(dense_normals[cell, data.valid_edges[cell]])
        points, normals = np.array(points), np.array(normals)
        mass = points.mean(axis=0)
        a = np.concatenate((normals, np.eye(3)))
        b = np.concatenate((np.sum(normals * points, axis=1), mass))
        system = np.block([[a.T @ a, hard.T], [hard, np.zeros((2, 2))]])
        optimum = np.linalg.solve(system, np.concatenate((a.T @ b, rhs)))[:3]
        key = ('joint', ((0, 0), (1, 0)), tuple(coordinate), 0)
        vertex = result['vertices'][result['vertex_keys'].index(key)]
        np.testing.assert_allclose(vertex, optimum, atol=1e-12)
        assert np.all(vertex >= xyz.min(axis=0) - 1e-12) and np.all(vertex <= xyz.max(axis=0) + 1e-12)
        assert np.max(np.abs(normals @ vertex - np.sum(normals * points, axis=1))) > 1e-4
        mass_only = mass - hard.T @ np.linalg.solve(hard @ hard.T, hard @ mass - rhs)
        deviations.append(np.linalg.norm(vertex - mass_only))
        improvements.append(np.sum((a @ mass_only - b) ** 2) - np.sum((a @ vertex - b) ** 2))
    assert max(deviations) > 1e-8
    assert max(improvements) > 1e-14


def test_true_curved_centers_are_evidence_not_exact_affine_constraints(curved_input):
    axes, fields, gradients, ownership = curved_input
    centers = (axes[0][:-1] + axes[0][1:]) / 2
    x, y, z = np.meshgrid(centers, centers, centers, indexing='ij')
    interior = np.array([z - .43 - .25 * (x - .5) ** 2 - .15 * (y - .5) ** 2, x - .57])
    result = model(axes, fields, gradients, ownership=ownership, interior_samples=interior)
    assert_seam(result, 0, 1)
    assert result['diagnostics']['center_approximant_max_residual'] > 1e-3
    assert result['diagnostics']['interior_topology'] == 'center_checked_not_topology_certified'


def test_nonaffine_curved_junction_remains_explicitly_unsupported():
    axes = [np.linspace(0, 1, 9)] * 3
    x, y, z = np.meshgrid(*axes, indexing='ij')
    field = z - .4317 - .15 * (x - .5) ** 2 - .1 * (y - .5) ** 2 - .1 * x * y
    gradient = np.stack((-.3 * (x - .5) - .1 * y, -.2 * (y - .5) - .1 * x, np.ones_like(z)), axis=-1)
    with pytest.raises(ValueError, match='unsupported_junction_field'):
        model(axes, [field, x - .57], [gradient, (1, 0, 0)], ownership=[np.ones_like(z), -field])


def test_fully_shared_curved_triangle_is_rejected_in_incidence_plan():
    from gempy_engine.modules.dual_contouring.topology_extraction import validate_joint_incidence

    joints = [('joint', ((0, 0), (1, 0)), cell, 0) for cell in ((0, 0, 0), (0, 1, 0), (0, 1, 1))]
    regular = ('regular', ((0, 0),), (0, 0, 1), 0)
    # A curved seam can make three joint representatives noncollinear. Do not
    # rely on a zero-area test to catch the resulting fully shared patch later.
    plans = [[dict(keys=[joints[0], joints[1], regular, joints[2]])]]
    with pytest.raises(ValueError, match='unsupported_shared_patch'):
        validate_joint_incidence(plans, {}, {}, np.array([[0, 0, 0]]), (2, 2, 2))
