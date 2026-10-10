"""Standalone bank-conditioned fields on genuine mixed-resolution leaves."""

from collections import Counter
from itertools import product

import numpy as np
import pytest

import gempy_engine.API.dual_contouring.joint_topology as api
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.dual_contouring.joint_cell_branches import CORNERS
from gempy_engine.modules.dual_contouring.joint_triangle_plan import (
    plan_adaptive_junctions, validate_adaptive_geometry,
)


@pytest.fixture(params=[AvailableBackends.numpy, AvailableBackends.PYTORCH], autouse=True)
def backend(request):
    if request.param is AvailableBackends.PYTORCH:
        pytest.importorskip('torch')
    old = BT.engine_backend, BT.use_gpu, BT.dtype, BT.use_pykeops
    BT._change_backend(request.param, use_gpu=False, dtype='float64')
    yield request.param
    BT._change_backend(old[0], use_gpu=old[1], dtype=old[2], use_pykeops=old[3])


def bank_case(bank, erosion=None, geometry=None):
    origins, spans = [], []
    for origin in product(range(0, 8, 2), repeat=3):
        if origin[1] < 4:
            origins.append(origin)
            spans.append(2)
        else:
            for offset in product(range(2), repeat=3):
                origins.append(np.array(origin) + offset)
                spans.append(1)
    origins, spans = np.asarray(origins), np.asarray(spans)
    separator = 0 if erosion is None else 1
    groups = [0, 1, 1] if erosion is None else [0, 1, 2, 2]
    indices = [0, 0, 1] if erosion is None else [0, 0, 0, 1]
    relations = [R.FAULT, R.BASEMENT] if erosion is None else [R.ERODE, R.FAULT, R.BASEMENT]
    # Original full metadata is retained, including the nonzero fault isovalue.
    levels = [[9.], [10., 20.]] if erosion is None else [[0.], [9.], [10., 20.]]
    faults = np.zeros((len(relations), len(relations)), dtype=bool)
    faults[groups[separator], groups[-1]] = True
    targets = [separator + 1, separator + 2]
    controllers = {t: [(separator, -1)] for t in targets}
    if erosion is not None:
        controllers = {t: [(0, -1), (separator, -1)] for t in targets}
    contract = dict(separator=separator, bank=bank,
                    fault_pairs={(separator, t) for t in targets}, controllers=controllers)

    def sample(points):
        x, y, z = points.T
        sign = -1. if bank == 0 else 1.
        heights = [3.4 + .2*bank, (5.4 if erosion is None else 7.4) + .2*bank]
        values = [sign*(x-3.3), z-heights[0]+10., z-heights[1]+20.]
        gradients = np.broadcast_to(np.array([[sign, 0., 0.], [0., 0., 1.], [0., 0., 1.]])[:, None],
                                     (3, len(points), 3)).copy()
        if geometry == 'curved':
            values[0] += .01*(y-4)**2
            gradients[0, :, 1] = .02*(y-4)
        elif geometry == 'gradient':
            gradients[0, :, 0] += .01*y
        elif geometry == 'scaled_gradient':
            gradients[0] *= .5
        elif geometry == 'reversed_gradient':
            gradients[0] *= -1.
        elif geometry == 'perpendicular_gradient':
            gradients[0] = [0., 1., 0.]
        elif geometry == 'near_endpoint':
            values[0] = sign*(x-3.999)
        elif geometry == 'aligned_separator':
            values[0] = sign*(x-3.)
        elif geometry == 'aligned_target':
            values[1] = z-3.+10.
        if erosion is not None:
            values.insert(0, z-erosion)
            gradients = np.concatenate((np.broadcast_to([0., 0., 1.], (1, len(points), 3)), gradients))
        values = np.asarray(values)
        if BT.engine_backend is AvailableBackends.PYTORCH:
            import torch
            return torch.as_tensor(values, dtype=torch.float64), torch.as_tensor(gradients, dtype=torch.float64)
        return values, gradients

    corners = origins[:, None] + spans[:, None, None]*CORNERS
    samples = sample(corners.reshape(-1, 3))[0].reshape(len(groups), -1, 8)
    args = (origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples,
            groups, indices, relations, levels)
    return args, dict(sample_fields=sample, faults_relations=faults, separator_contacts=contract)


@pytest.mark.parametrize('bank', [0, 1])
@pytest.mark.parametrize('geometry', [None, 'scaled_gradient'])
def test_two_bank_coarsefine_seams_and_same_stack_isolation(bank, geometry):
    args, kwargs = bank_case(bank, geometry=geometry)
    result = api.extract_adaptive_topology(*args, **kwargs)
    assert result['bank'] == bank
    assert result['diagnostics']['contact_type'] == 'explicit_fault_separator'
    assert result['diagnostics']['strict_crossings']
    assert result['diagnostics']['crossing_contract'] == 'strict_scalar_sides'
    assert result['diagnostics']['coarsefine_seam_edge_count'] == 2
    assert set(result['leaf_spans'][result['junction_cells']]) == {1, 2}
    counts = [Counter(tuple(sorted((int(a), int(b)))) for tri in faces
                      for a, b in zip(tri, np.roll(tri, -1))) for faces in result['faces']]
    seam_targets = Counter()
    for a, b in result['seam_edges']:
        edge = tuple(sorted((a, b)))
        assert counts[0][edge] == 2
        target = 1 if counts[1][edge] else 2
        seam_targets[target] += 1
        assert counts[target][edge] == 1
        assert counts[3-target][edge] == 0
    assert set(seam_targets) == {1, 2}
    assert not np.intersect1d(result['faces'][1], result['faces'][2]).size
    for target in (1, 2):
        x = result['vertices'][result['faces'][target]][..., 0]
        assert np.all(x >= 3.3-1e-10) if bank == 0 else np.all(x <= 3.3+1e-10)
    triangles = result['vertices'][result['faces'][0]]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    assert np.all(normals[:, 0]*(-1 if bank == 0 else 1) > 0)
    assert all(len(k[1]) == 2 and k[1][0][0] != k[1][1][0]
               for k in result['vertex_keys'] if k[0] == 'joint')


@pytest.mark.parametrize('bank', [0, 1])
def test_composite_erosion_skips_wholly_hidden_fault_contact(bank):
    args, kwargs = bank_case(bank, erosion=4.3)
    result = api.extract_adaptive_topology(*args, **kwargs)
    assert result['diagnostics']['coarsefine_seam_edge_count'] == 1
    assert len(result['faces'][2]) > 0
    assert len(result['faces'][3]) == 0
    assert all((2, 1) not in key[1] for key in result['vertex_keys'] if key[0] == 'joint')


@pytest.mark.parametrize('case', ['ordinary', 0, 1])
def test_deferred_plan_has_no_emission_and_matches_default_exactly(case, monkeypatch):
    from gempy_engine.modules.dual_contouring import joint_triangle_plan as topology

    args, kwargs = bank_case(1 if case == 'ordinary' else case)
    if case == 'ordinary':
        args = list(args)
        args[7] = [R.ERODE, R.BASEMENT]
        args[8] = [[0.], [10., 20.]]
        kwargs = dict(sample_fields=kwargs['sample_fields'])
    default = api.extract_adaptive_topology(*args, **kwargs)
    emit = topology.emit_adaptive_triangles
    forbid_emission(monkeypatch)
    monkeypatch.setattr(topology, 'emit_adaptive_triangles', api.emit_adaptive_triangles)
    prepared = api.extract_adaptive_topology(*args, **kwargs, defer_emission=True)
    assert prepared['faces'] is None and prepared['affected_faces'] is None
    assert set(prepared) == set(default) | {'_triangle_plan', '_key_to_id'}
    assert '_triangle_plan' not in default and '_key_to_id' not in default
    for key in ('vertices', 'seam_edges', 'junction_cells', 'leaf_origins', 'leaf_spans'):
        np.testing.assert_array_equal(prepared[key], default[key])
    assert prepared['vertex_keys'] == default['vertex_keys']
    assert prepared['diagnostics'] == default['diagnostics']
    if case != 'ordinary':
        assert prepared['bank'] == default['bank'] == case
    faces, affected = emit(prepared['_triangle_plan'], prepared['_key_to_id'])
    for actual, expected in zip(faces, default['faces']):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(affected, default['affected_faces']):
        np.testing.assert_array_equal(actual, expected)


def test_deferred_reference_rejected_before_sampling_or_emission(monkeypatch):
    args, kwargs = bank_case(0)
    forbid_emission(monkeypatch)
    kwargs['sample_fields'] = lambda points: pytest.fail('incompatible deferred reference reached sampling')
    with pytest.raises(ValueError, match='unsupported_deferred_reference'):
        api.extract_adaptive_topology(*args, **kwargs, defer_emission=True, include_reference=True)


def test_deferred_plan_still_requires_validated_geometry(monkeypatch):
    args, kwargs = bank_case(0, geometry='reversed_gradient')
    forbid_emission(monkeypatch)
    with pytest.raises(ValueError, match='unsupported_separator_orientation'):
        api.extract_adaptive_topology(*args, **kwargs, defer_emission=True)


@pytest.mark.parametrize('bank', [0, 1])
def test_bank_uses_strict_primitive_not_tolerance_extrapolated_crossings(bank, monkeypatch):
    args, kwargs = bank_case(bank, geometry='near_endpoint')
    calls = []
    primitive = api.find_intersection_on_edge

    def recorded(*args, **kwargs):
        calls.append(kwargs.get('strict_crossings', False))
        return primitive(*args, **kwargs)

    monkeypatch.setattr(api, 'find_intersection_on_edge', recorded)
    result = api.extract_adaptive_topology(*args, **kwargs)
    assert calls == [True]*3
    assert result['diagnostics']['coarsefine_seam_edge_count'] == 2
    joint = result['vertices'][np.unique(result['seam_edges'])]
    np.testing.assert_allclose(joint[:, 0], 3.999, atol=1e-12)
    # The unchanged ordinary path still rejects the legacy primitive's
    # same-sign extrapolated intersections close to an edge endpoint.
    if bank == 1:
        ordinary = list(args)
        ordinary[7] = [R.ERODE, R.BASEMENT]
        ordinary[8] = [[0.], [10., 20.]]
        with pytest.raises(ValueError, match='unsupported_geometric_flags'):
            api.extract_adaptive_topology(*ordinary, sample_fields=kwargs['sample_fields'])
        assert calls[-1] is False


@pytest.mark.parametrize('bank', [0, 1])
@pytest.mark.parametrize('geometry', ['aligned_separator', 'aligned_target'])
def test_bank_genuine_zero_aligned_corners_rejected_before_emission(bank, geometry, monkeypatch):
    args, kwargs = bank_case(bank, geometry=geometry)
    forbid_emission(monkeypatch)
    with pytest.raises(ValueError, match='sample_aligned_interface'):
        api.extract_adaptive_topology(*args, **kwargs)


def forbid_emission(monkeypatch):
    def forbidden(*args):
        pytest.fail('invalid contract or junction reached triangle emission')
    monkeypatch.setattr(api, 'emit_adaptive_triangles', forbidden)


@pytest.mark.parametrize('case', ['reverse', 'missing', 'extra', 'self', 'same_stack', 'controller', 'sign', 'none'])
def test_unauthorized_directed_fault_edges_and_controller_overrides(case, monkeypatch):
    args, kwargs = bank_case(1)
    contract = kwargs['separator_contacts']
    if case == 'reverse':
        kwargs['faults_relations'] = kwargs['faults_relations'].T
    elif case == 'missing':
        contract['fault_pairs'].remove((0, 2))
    elif case in ('extra', 'self', 'same_stack'):
        contract['fault_pairs'].add({'extra': (1, 0), 'self': (0, 0), 'same_stack': (1, 2)}[case])
    elif case == 'controller':
        contract['controllers'][1].append((2, -1))
    elif case == 'none':
        contract['controllers'] = None
    else:
        contract['controllers'][1] = [(0, 1)]
    forbid_emission(monkeypatch)
    with pytest.raises(ValueError, match='unauthorized_fault_pairs|unsupported_ownership'):
        api.extract_adaptive_topology(*args, **kwargs)


@pytest.mark.parametrize('pairs', [
    {(0., 1.), (0., 2.)}, {(False, 1), (False, 2)}, {(0, True), (0, 2)},
    {(0, -1)}, {(0, 3)}, {(0,)}, {0},
])
def test_malformed_fault_pair_indices_rejected_before_emission(pairs, monkeypatch):
    args, kwargs = bank_case(0)
    kwargs['separator_contacts']['fault_pairs'] = pairs
    forbid_emission(monkeypatch)
    with pytest.raises(ValueError, match='invalid_fault_pairs'):
        api.extract_adaptive_topology(*args, **kwargs)


def test_unrepresented_second_fault_label_rejected_before_emission(monkeypatch):
    args, kwargs = bank_case(0)
    args = list(args)
    args[7] = [R.FAULT, R.BASEMENT, R.FAULT]
    args[8] = args[8] + [[4.3]]
    faults = np.zeros((3, 3), dtype=bool)
    faults[0, 1] = True
    kwargs['faults_relations'] = faults
    forbid_emission(monkeypatch)
    with pytest.raises(ValueError, match='unsupported_multiple_faults'):
        api.extract_adaptive_topology(*args, **kwargs)


@pytest.mark.parametrize('fault_contract', [False, True])
def test_known_unrepresented_erosion_controller_rejected_before_emission(fault_contract, monkeypatch):
    args, kwargs = bank_case(0)
    args = list(args)
    args[5] = [1, 2, 2]
    args[7] = [R.ERODE, R.FAULT if fault_contract else R.BASEMENT, R.BASEMENT]
    args[8] = [[4.3], [9. if fault_contract else 0.], [10., 20.]]
    faults = np.zeros((3, 3), dtype=bool)
    faults[1, 2] = True
    kwargs['faults_relations'] = faults
    if not fault_contract:
        kwargs = dict(sample_fields=kwargs['sample_fields'])
    forbid_emission(monkeypatch)
    with pytest.raises(ValueError, match='unsupported_missing_controller.*0/0'):
        api.extract_adaptive_topology(*args, **kwargs)


@pytest.mark.parametrize('kind', ['erode_min', 'onlap_max'])
def test_known_unrepresented_extremum_in_present_stack_rejected_before_emission(kind, monkeypatch):
    args, kwargs = bank_case(1)
    args = list(args)
    if kind == 'erode_min':
        args[6] = [1, 0, 1]
        args[7] = [R.ERODE, R.BASEMENT]
        args[8] = [[-1., 0.], [10., 20.]]
    else:
        args[7] = [R.ONLAP, R.BASEMENT]
        args[8] = [[0.], [10., 20., 30.]]
    forbid_emission(monkeypatch)
    with pytest.raises(ValueError, match='unsupported_missing_controller'):
        api.extract_adaptive_topology(*args, sample_fields=kwargs['sample_fields'])


def test_unrepresented_null_space_is_not_an_ordinary_controller():
    args, kwargs = bank_case(0)
    args = list(args)
    args[5] = [1, 2, 2]
    args[7] = [R.NULL_SPACE, R.FAULT, R.BASEMENT]
    args[8] = [[4.3], [9.], [10., 20.]]
    faults = np.zeros((3, 3), dtype=bool)
    faults[1, 2] = True
    kwargs['faults_relations'] = faults
    result = api.extract_adaptive_topology(*args, **kwargs)
    assert result['diagnostics']['coarsefine_seam_edge_count'] == 2


@pytest.mark.parametrize('bank', [0, 1])
@pytest.mark.parametrize('geometry', ['reversed_gradient', 'perpendicular_gradient'])
def test_separator_actual_gradient_direction_rejected_before_emission(bank, geometry, monkeypatch):
    args, kwargs = bank_case(bank, geometry=geometry)
    forbid_emission(monkeypatch)
    with pytest.raises(ValueError, match='unsupported_separator_orientation'):
        api.extract_adaptive_topology(*args, **kwargs)


@pytest.mark.parametrize('geometry', ['curved', 'gradient'])
def test_nonplanar_or_drifting_separator_rejected(geometry, monkeypatch):
    args, kwargs = bank_case(0, geometry=geometry)
    forbid_emission(monkeypatch)
    with pytest.raises(ValueError, match='unsupported_separator_geometry'):
        api.extract_adaptive_topology(*args, **kwargs)


@pytest.mark.parametrize('bank', [0, 1])
def test_competing_triple_rejected_before_emission(bank, monkeypatch):
    args, kwargs = bank_case(bank, erosion=3.7)
    forbid_emission(monkeypatch)
    with pytest.raises(ValueError, match='unsupported_multiway_junction|unsupported_ownership_tile'):
        api.extract_adaptive_topology(*args, **kwargs)


def test_actual_joint_representative_obeys_nonparticipating_controller():
    key = ('joint', ((0, 0), (1, 0)), (0, 0, 0), 1, 0)
    triangles = [[dict(keys=(key, key, key))], [], []]
    with pytest.raises(ValueError, match='unowned_qef_vertex'):
        validate_adaptive_geometry(triangles, set(), {0: dict(key=key, pair=(0, 1))}, {},
                                   {key: np.array([0., 0., 1.])}, np.full(3, 1e-10),
                                   {0: [(1, -1), (2, -1)], 1: [], 2: []})


@pytest.mark.parametrize('participant', [0, 1])
@pytest.mark.parametrize('decision', ['excluded', 'competing', 'mismatch'])
def test_candidate_evidence_checks_both_participants_before_joint(participant, decision):
    # The same third controller must be considered on the controller itself,
    # not just on the nominal target of the represented contact equality.
    face = dict(cells=(0, None))
    tile_fields = np.array([[[-.3, .7, -.3, .7]], [[-.4, -.4, .6, .6]],
                            [[-1., -1., -1., -1.]]])
    leaf_fields = np.ones((3, 1, 8))
    leaf_fields[2] = -1.
    if decision == 'competing':
        tile_fields[2, 0] = [-1., 1., -1., 1.]
        leaf_fields[2, 0, 1] = 1.
    elif decision == 'mismatch':
        tile_fields[2] = 1.
    kwargs = dict(controllers={participant: [(2, 1)]}, leaf_fields=leaf_fields)
    args = ([face], np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]]),
            tile_fields, {(0, 1)}, [(0, 0), (1, 0), (2, 0)], np.array([[0, 0, 0]]),
            np.array([1]), None, None, None, None)
    if decision == 'excluded':
        assert plan_adaptive_junctions(*args, **kwargs) == ({}, [])
    else:
        with pytest.raises(ValueError, match='unsupported_multiway_junction|unsupported_ownership_tile'):
            plan_adaptive_junctions(*args, **kwargs)


@pytest.mark.parametrize('mixed', [False, True], ids=['full_refine', 'mixed_leaves'])
@pytest.mark.parametrize('backend', [AvailableBackends.numpy], indirect=True)
def test_actual_model7_root7_two_levels_strict_banks(mixed, monkeypatch):
    import importlib
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.core.data.regular_grid import RegularGrid
    from tests.fixtures.model7_combination import model7_combination_factory

    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=True)
    root = RegularGrid(inputs.grid.octree_grid.orthogonal_extent, [7, 7, 7])
    inputs.grid.octree_grid = root
    options.evaluation_options.octree_min_level = 0
    options.evaluation_options.mesh_extraction_overlap = 'joint'
    internals = importlib.import_module('gempy_engine.modules.octrees_topology._octree_internals')
    mark = internals._mark_voxel

    def select(corner_ids):
        shifts, _ = mark(corner_ids)
        mask = np.ones(len(root.values), dtype=bool)
        if mixed:
            # Isolated contacts in the two displaced banks retain their actual
            # production parents; every other parent refines normally.
            extent = np.asarray(root.orthogonal_extent)
            indices = np.floor((root.values-extent[::2]) /
                               ((extent[1::2]-extent[::2])/7)).astype(int)
            for parent in ((5, 3, 1), (5, 3, 3)):
                mask[np.all(indices == parent, axis=1)] = False
        return shifts, mask

    monkeypatch.setattr(internals, '_mark_voxel', select)
    calls = []
    extract = api.extract_adaptive_topology

    def audited(*args, **kwargs):
        result = extract(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(importlib.import_module('gempy_engine.API.dual_contouring.joint_fault_banks'),
                        'extract_adaptive_topology', audited)
    original_points = inputs.surface_points.sp_coords.copy()
    original_orientations = inputs.orientations.dip_positions.copy()
    solution = compute_model(inputs, options, descriptor)
    assert len(solution.dc_meshes) == 4
    assert len(calls) == 2 and {r['bank'] for r in calls} == {0, 1}
    for result in calls:
        assert result['diagnostics']['strict_crossings']
        assert set(result['leaf_spans']) == ({1, 2} if mixed else {1})
        assert result['diagnostics']['coarsefine_seam_edge_count'] == (2 if mixed else 0)
        assert len(result['leaf_spans']) == (2730 if mixed else 2744)
        assert len(result['seam_edges']) > 0
        counts = [Counter(tuple(sorted((int(a), int(b)))) for tri in faces
                          for a, b in zip(tri, np.roll(tri, -1))) for faces in result['faces']]
        for a, b in result['seam_edges']:
            edge = tuple(sorted((a, b)))
            assert sorted(count[edge] for count in counts if count[edge]) == [1, 2]
    for mesh in solution.dc_meshes:
        assert len(mesh.edges) > 0
        assert set(mesh.joint_face_bank_ids) == {0, 1}
        for triangle, bank in zip(mesh.edges, mesh.joint_face_bank_ids):
            assert np.all(mesh.joint_bank_ids[triangle] == bank)
    np.testing.assert_array_equal(inputs.surface_points.sp_coords, original_points)
    np.testing.assert_array_equal(inputs.orientations.dip_positions, original_orientations)
