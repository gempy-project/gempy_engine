"""Leaf-native contact seams, including a genuine coarse/fine face tile."""

from collections import Counter
from itertools import product

import numpy as np
import pytest

from gempy_engine.API.dual_contouring.joint_topology import extract_adaptive_topology
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices
from gempy_engine.modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge
from gempy_engine.modules.dual_contouring.joint_cell_branches import CORNERS
from gempy_engine.modules.dual_contouring.joint_triangle_plan import (
    face_pair_root, plan_adaptive_triangles, validate_adaptive_geometry,
)


@pytest.fixture(autouse=True)
def numpy_backend():
    old = BT.engine_backend, BT.use_gpu, BT.dtype, BT.use_pykeops
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, dtype='float64')
    yield
    BT._change_backend(old[0], use_gpu=old[1], dtype=old[2], use_pykeops=old[3])


def mixed_leaves(permutation=(0, 1, 2)):
    origins, spans = [], []
    for origin in product(range(0, 8, 2), repeat=3):
        if origin[1] < 4:
            origins.append(origin)
            spans.append(2)
        else:
            for offset in product(range(2), repeat=3):
                origins.append(np.array(origin)+offset)
                spans.append(1)
    return np.asarray(origins)[:, permutation], np.asarray(spans)


def plane_fields(points, permutation=(0, 1, 2), onlap=False):
    x, _, z = points[:, np.argsort(permutation)].T
    values = np.array([x-3.3, z-3.4])
    normals = np.array([[1., 0., 0.], [0., 0., 1.]])[:, permutation]
    if onlap:
        values, normals = values[::-1], normals[::-1]
    return values, np.broadcast_to(normals[:, None, :], (2, len(points), 3)).copy()


def extract_planes(permutation=(0, 1, 2), onlap=False, **kwargs):
    origins, spans = mixed_leaves(permutation)
    corners = origins[:, None, :]+spans[:, None, None]*CORNERS
    calls = []

    def callback(points):
        calls.extend(map(tuple, points))
        return plane_fields(points, permutation, onlap)

    samples = plane_fields(corners.reshape(-1, 3), permutation, onlap)[0].reshape(2, -1, 8)
    result = extract_adaptive_topology(
        origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples, [0, 1], [0, 0],
        [R.ONLAP if onlap else R.ERODE, R.BASEMENT], [[0.], [0.]],
        sample_fields=callback, **kwargs)
    assert len(calls) == len(set(calls)), 'canonical points must be queried only once for all surfaces'
    return result, samples


def triangle_edges(faces):
    return Counter(tuple(sorted((int(a), int(b)))) for triangle in faces
                   for a, b in zip(triangle, np.roll(triangle, -1)))


@pytest.mark.parametrize('permutation', [(0, 1, 2), (1, 2, 0), (2, 0, 1)])
@pytest.mark.parametrize('onlap', [False, True])
@pytest.mark.parametrize('backend', [AvailableBackends.numpy, AvailableBackends.PYTORCH])
def test_mixed_leaf_joint_seam(permutation, onlap, backend):
    if backend is AvailableBackends.PYTORCH:
        pytest.importorskip('torch')
    BT._change_backend(backend, use_gpu=False, dtype='float64')
    result, _ = extract_planes(permutation, onlap, include_reference=True)
    diagnostics = result['diagnostics']
    assert diagnostics['coarsefine_seam_edge_count'] == 1
    assert diagnostics['no_uniform_fallback'] and diagnostics['no_post_reconciliation']
    assert set(result['leaf_spans']) == {1, 2}
    junctions = result['junction_cells']
    assert np.any(result['leaf_spans'][junctions] == 2)
    assert np.any(result['leaf_spans'][junctions] == 1)
    counts = [triangle_edges(f) for f in result['faces']]
    controller, target = (1, 0) if onlap else (0, 1)
    for a, b in result['seam_edges']:
        assert counts[controller][tuple(sorted((a, b)))] == 2
        assert counts[target][tuple(sorted((a, b)))] == 1
        directed = Counter((int(u), int(v)) for triangle in result['faces'][controller]
                           for u, v in zip(triangle, np.roll(triangle, -1)))
        assert directed[a, b] == directed[b, a] == 1
    positions = result['vertices'][np.unique(result['seam_edges'])][:, np.argsort(permutation)]
    np.testing.assert_allclose(positions[:, 0], 3.3, atol=1e-12)
    np.testing.assert_allclose(positions[:, 2], 3.4, atol=1e-12)
    target_points = result['vertices'][result['faces'][target]][..., np.argsort(permutation)]
    assert np.all(target_points[..., 0] >= 3.3-1e-10) if onlap else np.all(target_points[..., 0] <= 3.3+1e-10)
    reference = dict(zip(result['reference']['vertex_keys'], result['reference']['vertices']))
    ordinary = [(k, p) for k, p in zip(result['vertex_keys'], result['vertices']) if k[0] == 'regular']
    assert ordinary
    for key, position in ordinary:
        np.testing.assert_array_equal(position, reference[key])
    for surface, flags in enumerate(result['affected_faces']):
        actual = [tuple(result['vertex_keys'][v] for v in triangle)
                  for triangle in result['faces'][surface][~flags]]
        retained_keys = {k for k in result['vertex_keys'] if k[0] == 'regular'}
        independent = [tuple(result['reference']['vertex_keys'][v] for v in triangle)
                       for triangle in result['reference']['faces'][surface]
                       if all(result['reference']['vertex_keys'][v] in retained_keys for v in triangle)]
        assert Counter(actual) == Counter(independent)
    for surface, faces in enumerate(result['faces']):
        triangles = result['vertices'][faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        expected = plane_fields(np.zeros((1, 3)), permutation, onlap)[1][surface, 0]
        assert np.all(normals @ expected > 0)


@pytest.mark.parametrize('backend', [AvailableBackends.numpy, AvailableBackends.PYTORCH])
def test_original_production_qef_exact(backend):
    if backend is AvailableBackends.PYTORCH:
        pytest.importorskip('torch')
    BT._change_backend(backend, use_gpu=False, dtype='float64')
    result, samples = extract_planes(include_reference=True)
    origins, spans = mixed_leaves()
    xyz = origins[:, None, :]+spans[:, None, None]*CORNERS

    def tensor(a, boolean=False):
        if backend is AvailableBackends.PYTORCH:
            import torch
            return torch.as_tensor(a, dtype=torch.bool if boolean else torch.float64)
        return np.asarray(a, dtype=bool if boolean else float)

    def array(a):
        return a.detach().numpy() if hasattr(a, 'detach') else a

    reference = dict(zip(result['reference']['vertex_keys'], result['reference']['vertices']))
    for s in range(2):
        crossings, flags = find_intersection_on_edge(tensor(xyz.reshape(-1, 3)),
                                                    tensor(samples[s].reshape(-1)), tensor([0.]))
        flags = array(flags).reshape(-1, 12)
        normals = plane_fields(array(crossings))[1][s]
        dc = DualContouringData(crossings, tensor(flags, True), tensor(xyz.mean(axis=1)),
                                tensor(spans[:, None]*np.ones(3)), 1, tensor(origins), gradients=tensor(normals))
        expected = array(generate_dual_contouring_vertices(dc))
        for cell, position in zip(np.flatnonzero(flags.any(axis=1)), expected):
            key = ('regular', ((s, 0),), tuple(origins[cell]), int(spans[cell]), 0)
            np.testing.assert_array_equal(reference[key], position)


def test_signed_synthetic_ownership_uses_canonical_controller():
    origins, spans = mixed_leaves()
    xyz = origins[:, None, :]+spans[:, None, None]*CORNERS
    fields = plane_fields(xyz.reshape(-1, 3))[0].reshape(2, -1, 8)
    own = np.array([np.ones_like(fields[0]), -fields[0]])
    synthetic, _ = extract_planes(ownership=own)
    geological, _ = extract_planes()
    assert synthetic['vertex_keys'] == geological['vertex_keys']
    np.testing.assert_array_equal(synthetic['vertices'], geological['vertices'])
    for a, b in zip(synthetic['faces'], geological['faces']):
        np.testing.assert_array_equal(a, b)


def test_same_stack_and_unrelated_parallel_surfaces_remain_independent():
    origins, spans = mixed_leaves()
    xyz = origins[:, None, :]+spans[:, None, None]*CORNERS

    def callback(points):
        values = np.array([points[:, 2]-3.3, points[:, 2]-3.6, points[:, 2]-5.3])
        return values, np.broadcast_to([0., 0., 1.], (3, len(points), 3)).copy()

    samples = callback(xyz.reshape(-1, 3))[0].reshape(3, -1, 8)
    result = extract_adaptive_topology(origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples,
                                      [0, 0, 1], [0, 1, 0], [R.ERODE, R.BASEMENT], [[0., 1.], [0.]],
                                      sample_fields=callback, ownership=np.ones_like(samples))
    assert not result['junction_cells'].size
    for a, b in product(range(3), repeat=2):
        if a != b:
            assert not np.intersect1d(result['faces'][a], result['faces'][b]).size


@pytest.mark.parametrize('same_stack', [False, True])
def test_parallel_ordinary_surfaces_without_synthetic_ownership(same_stack):
    origins, spans = mixed_leaves()
    xyz = origins[:, None, :]+spans[:, None, None]*CORNERS

    def callback(points):
        values = np.broadcast_to(points[:, 0], (2, len(points))).copy()
        return values, np.broadcast_to([1., 0., 0.], (2, len(points), 3)).copy()

    samples = callback(xyz.reshape(-1, 3))[0].reshape(2, -1, 8)
    groups = [0, 0] if same_stack else [0, 1]
    indices = [0, 1] if same_stack else [0, 0]
    relations = [R.BASEMENT] if same_stack else [R.ERODE, R.BASEMENT]
    levels = [[3.3, 3.6]] if same_stack else [[3.3], [3.6]]
    result = extract_adaptive_topology(origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples,
                                      groups, indices, relations, levels, sample_fields=callback)
    assert not result['junction_cells'].size
    assert not np.intersect1d(*result['faces']).size


def test_leaf_order_does_not_change_stable_geometry_or_faces():
    baseline, samples = extract_planes()
    origins, spans = mixed_leaves()
    order = np.random.default_rng(17).permutation(len(origins))
    reordered = extract_adaptive_topology(
        origins[order], spans[order], (8, 8, 8), (0, 8, 0, 8, 0, 8), samples[:, order],
        [0, 1], [0, 0], [R.ERODE, R.BASEMENT], [[0.], [0.]], sample_fields=plane_fields)
    assert baseline['vertex_keys'] == reordered['vertex_keys']
    np.testing.assert_array_equal(baseline['vertices'], reordered['vertices'])
    for a, b in zip(baseline['faces'], reordered['faces']):
        np.testing.assert_array_equal(a, b)


def test_curved_monotone_contact_uses_actual_crossing_normals():
    origins, spans = mixed_leaves()
    xyz = origins[:, None, :]+spans[:, None, None]*CORNERS

    def callback(points):
        x, y, z = points.T
        values = np.array([x-3.3-.004*(y-4)**2, z-3.4-.003*(y-4)**2])
        normals = np.zeros((2, len(points), 3))
        normals[0, :, 0], normals[1, :, 2] = 1., 1.
        normals[0, :, 1], normals[1, :, 1] = -.008*(y-4), -.006*(y-4)
        return values, normals

    samples = callback(xyz.reshape(-1, 3))[0].reshape(2, -1, 8)
    result = extract_adaptive_topology(
        origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples, [0, 1], [0, 0],
        [R.ERODE, R.BASEMENT], [[0.], [0.]], sample_fields=callback, include_reference=True)
    assert result['diagnostics']['coarsefine_seam_edge_count'] == 1
    reference = dict(zip(result['reference']['vertex_keys'], result['reference']['vertices']))
    for key, position in zip(result['vertex_keys'], result['vertices']):
        if key[0] == 'regular':
            np.testing.assert_array_equal(reference[key], position)


def test_competing_pairs_rejected_before_emission():
    origins, spans = mixed_leaves()
    xyz = origins[:, None, :]+spans[:, None, None]*CORNERS

    def callback(points):
        values = np.array([points[:, 0]-3.3, points[:, 2]-3.4, points[:, 2]-2.6])
        normals = np.array([[1., 0., 0.], [0., 0., 1.], [0., 0., 1.]])
        return values, np.broadcast_to(normals[:, None, :], (3, len(points), 3)).copy()

    samples = callback(xyz.reshape(-1, 3))[0].reshape(3, -1, 8)
    own = np.array([np.ones_like(samples[0]), -samples[0], -samples[0]])
    with pytest.raises(ValueError, match='unsupported_multiway_junction'):
        extract_adaptive_topology(origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples,
                                  [0, 1, 1], [0, 0, 1], [R.ERODE, R.BASEMENT], [[0.], [0., 1.]],
                                  sample_fields=callback, ownership=own)


def test_aligned_tile_pair_rejected():
    with pytest.raises(ValueError, match='grid_edge_junction'):
        face_pair_root(np.array([[0., 1., 0., 1.], [-.3, -.3, .7, .7]]))


@pytest.mark.parametrize('values', [
    [[-.3, .7, -.3, .7], [-.6, .4, -.6, .4]],
    [[-.3, -.3, .7, .7], [-.6, -.6, .4, .4]],
])
def test_parallel_face_restrictions_have_no_contact(values):
    assert face_pair_root(values) is None


def test_unbalanced_leaves_explicitly_rejected():
    origins = [(0, 0, 0)] + [(x, y, z) for x, y, z in product(range(8), range(4), range(4)) if x >= 4]
    spans = [4]+[1]*(len(origins)-1)
    with pytest.raises(ValueError, match='unsupported_unbalanced_octree'):
        extract_adaptive_topology(origins, spans, (8, 4, 4), (0, 8, 0, 4, 0, 4),
                                  np.ones((1, len(origins), 8)), [0], [0], [R.BASEMENT], [[0.]],
                                  sample_fields=lambda p: None)


def test_float32_and_faults_explicitly_unsupported():
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, dtype='float32')
    with pytest.raises(ValueError, match='unsupported_backend'):
        extract_planes()
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, dtype='float64')
    with pytest.raises(ValueError, match='unsupported_fault_extraction'):
        extract_adaptive_topology([[0, 0, 0]], [1], (1, 1, 1), (0, 1, 0, 1, 0, 1),
                                  np.ones((1, 1, 8)), [0], [0], [R.FAULT], [[0.]],
                                  sample_fields=lambda p: None)


@pytest.mark.parametrize('backend', [AvailableBackends.numpy, AvailableBackends.PYTORCH])
@pytest.mark.parametrize('case, error', [
    ('folded_controller', 'folded_controller_seam'),
    ('unowned_target', 'unowned_qef_vertex'),
    ('offset_mismatch', 'inconsistent_corner_samples'),
    ('hidden_hanging_branch', 'unsupported_hanging_branch'),
])
def test_review_reproductions_rejected_before_face_emission(backend, case, error, monkeypatch):
    import gempy_engine.API.dual_contouring.joint_topology as api

    if backend is AvailableBackends.PYTORCH:
        pytest.importorskip('torch')
    BT._change_backend(backend, use_gpu=False, dtype='float64')
    origins, spans = mixed_leaves()
    xyz = origins[:, None, :]+spans[:, None, None]*CORNERS

    def callback(points):
        x, y, z = points.T
        values = np.array([x-3.3, z-3.4])
        gradients = np.zeros((2, len(points), 3))
        gradients[0, :, 0], gradients[1, :, 2] = 1., 1.
        if case == 'folded_controller':
            values[0] += 2*(z-4)*np.sin(np.pi*y)-.5*(y-4)*np.sin(np.pi*z)
            gradients[0, :, 1] = 2*(z-4)*np.pi*np.cos(np.pi*y)-.5*np.sin(np.pi*z)
            gradients[0, :, 2] = 2*np.sin(np.pi*y)-.5*(y-4)*np.pi*np.cos(np.pi*z)
        elif case == 'unowned_target':
            a, b = 3+1.5*(x-4), -(x-4)-3*(y-4)
            values[1] += a*np.sin(np.pi*x)+b*np.sin(np.pi*y)
            gradients[1, :, 0] = 1.5*np.sin(np.pi*x)+a*np.pi*np.cos(np.pi*x)-np.sin(np.pi*y)
            gradients[1, :, 1] = -3*np.sin(np.pi*y)+b*np.pi*np.cos(np.pi*y)
        elif case == 'offset_mismatch':
            values[0] = 1e12+x-3.3
        else:
            delta = points-np.array([3., 4., 3.])
            values = (np.sum(delta**2, axis=1)-.6**2)[None, :]
            gradients = (2*delta)[None, :, :]
        return values, gradients

    samples = callback(xyz.reshape(-1, 3))[0].reshape(-1, len(origins), 8)
    if case == 'offset_mismatch':
        samples[0] += .1
    n = len(samples)
    levels = [[1e12], [0.]] if case == 'offset_mismatch' else [[0.]]*n

    def forbidden_emission(*args):
        pytest.fail('invalid patch reached final triangle allocation')

    monkeypatch.setattr(api, 'emit_adaptive_triangles', forbidden_emission)
    with pytest.raises(ValueError, match=error):
        api.extract_adaptive_topology(
            origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples,
            list(range(n)), [0]*n, [R.ERODE, R.BASEMENT] if n == 2 else [R.BASEMENT],
            levels, sample_fields=callback)


@pytest.mark.parametrize('backend', [AvailableBackends.numpy, AvailableBackends.PYTORCH])
def test_consistent_large_raw_offset_remains_supported(backend):
    if backend is AvailableBackends.PYTORCH:
        pytest.importorskip('torch')
    BT._change_backend(backend, use_gpu=False, dtype='float64')
    origins, spans = mixed_leaves()
    xyz = origins[:, None, :]+spans[:, None, None]*CORNERS

    def callback(points):
        values, gradients = plane_fields(points)
        values[0] = 1e12+points[:, 0]-3.3
        return values, gradients

    samples = callback(xyz.reshape(-1, 3))[0].reshape(2, -1, 8)
    result = extract_adaptive_topology(
        origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples, [0, 1], [0, 0],
        [R.ERODE, R.BASEMENT], [[1e12], [0.]], sample_fields=callback)
    assert result['diagnostics']['coarsefine_seam_edge_count'] == 1


@pytest.mark.parametrize('case, error', [
    ('geometry', 'third vertices do not straddle seam'),
    ('field', 'third vertices do not straddle target field'),
])
def test_opposite_oriented_seam_still_requires_geometric_and_field_sides(case, error):
    members = ((0, 0), (1, 0))
    a = ('joint', members, (0, 0, 0), 1, 0)
    b = ('joint', members, (1, 0, 0), 1, 0)
    regular = [('regular', ((s, 0),), (i, 1, 0), 1, 0) for i, s in enumerate([0, 0, 1])]
    positions = {a: np.array([0., 0., 0.]), b: np.array([1., 0., 0.]),
                 regular[0]: np.array([0., 1., 0.]),
                 regular[1]: np.array([0., 2. if case == 'geometry' else -1., 0.]),
                 regular[2]: np.array([0., 0., 1.])}
    plans = [[dict(keys=[a, b, regular[0]], normal=np.array([0., 0., 1.])),
              dict(keys=[b, a, regular[1]], normal=np.array([0., 0., -.5 if case == 'geometry' else 1.]))],
             [dict(keys=[a, b, regular[2]], normal=np.array([0., -1., 0.]))]]
    triangles = plan_adaptive_triangles(plans, positions)
    assert triangles[0][0]['keys'][:2] == triangles[0][1]['keys'][:2][::-1]
    vertex_fields = {regular[0]: np.array([0., 1.]),
                     regular[1]: np.array([0., -1. if case == 'geometry' else 1.])}
    with pytest.raises(ValueError, match=error):
        validate_adaptive_geometry(triangles, {(a, b)}, {0: dict(key=a, pair=(0, 1))},
                                   positions, vertex_fields, np.array([1e-10, 1e-10]), {0: [], 1: []})
