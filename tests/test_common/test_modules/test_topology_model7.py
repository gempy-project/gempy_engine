"""Actual gallery Model7: original solve, forced bank drift, and shared DC incidence."""

import copy
from collections import Counter

import numpy as np
import pytest

from examples.topology_aware_model7 import run_case
from gempy_engine.API.dual_contouring.model7_bank_experiment import (
    capture_model7, extract_bank_skin, query_bank,
)
from tests.fixtures.model7_combination import model7_combination_factory


@pytest.fixture(scope='module')
def experiment():
    return run_case(14)


def test_actual_model7_full_surface_counts_and_sampling(experiment):
    report, meshes, _, _ = experiment
    assert report['status'] == 'supported'
    assert report['observations'] == 102 and report['orientations'] == 8
    assert report['original_observations_preserved']
    assert report['production_triangles'] == [126, 112, 236, 344]
    assert report['joint_triangle_counts'] == [[442, 78, 156, 182], [442, 234, 676, 878]]
    assert report['joint_seam_counts'] == [52, 91]
    assert all(len(f) for mesh in meshes for f in mesh['faces'])
    for parity in report['bank_sampling']:
        for stack in parity:
            assert stack['saturated_queries'] > 1000
            assert stack['scalar_max_error'] < 1e-12
            assert stack['gradient_max_error'] < 1e-12
            assert stack['drift_min'] == 0 and stack['drift_max'] == 1
            assert stack['smooth_transition_queries'] == 27
            assert stack['shifted_sigmoid_max_error'] < 1e-12
    assert min(report['dependent_bank_max_difference']) > .05


def test_queries_preserve_weights_and_original_reference_drifts(experiment):
    _, _, _, data = experiment
    xyz = data['xyz'].reshape(-1, 3)[::71]
    for capture in data['captures'][1:]:
        original, weights, _ = capture
        before = copy.deepcopy(original)
        weights_before = weights.copy()
        zero, zero_grad = query_bank(capture, xyz, 0)
        one, one_grad = query_bank(capture, xyz, 1)
        assert np.max(np.abs(one-zero)) > .05
        assert np.ptp(one-zero) < 1e-12
        np.testing.assert_allclose(zero_grad, one_grad, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(weights, weights_before)
        np.testing.assert_array_equal(original.xyz_to_interpolate, before.xyz_to_interpolate)
        for attribute in ('fault_values_everywhere', 'fault_values_on_sp', 'fault_values_ref', 'fault_values_rest'):
            np.testing.assert_array_equal(getattr(original.fault_internal, attribute),
                                          getattr(before.fault_internal, attribute))
        assert isinstance(zero, np.ndarray) and zero.dtype == np.float64
        assert np.isfinite(zero_grad).all()


def test_bank_orientation_is_verified_not_assumed(experiment):
    _, meshes, solution, data = experiment
    fault, _ = query_bank(data['captures'][0], data['xyz'].reshape(-1, 3))
    signed = fault-data['levels'][0][0]
    for bank, mesh in enumerate(meshes):
        capture = data['captures'][1][0]
        slc = solution.root_output.outputs[1].grid.custom_grid_slice
        drift = capture.fault_internal.fault_values_everywhere[0, slc][:len(fault)]
        np.testing.assert_array_equal(drift > .5, signed < 0)
        for faces in mesh['faces'][1:]:
            values, _ = query_bank(data['captures'][0], mesh['vertices'][np.unique(faces)])
            side = (-1 if bank == 0 else 1)*(values-data['levels'][0][0])
            assert side.max() < 1e-10


def test_shared_seams_are_actual_oriented_mesh_incidence(experiment):
    report, meshes, _, _ = experiment
    for mesh, validation in zip(meshes, report['validation']):
        assert validation['actual_shared_edges_match']
        assert validation['noncontact_parity']['passed']
        assert not any(validation['bank_side_violations'])
        assert not any(validation['actual_engine_winding_violations'])
        assert max(validation['older_surface_max_erosion_residual']) < 1e-8
        assert all(not any(d.values()) for d in validation['defects'])
        assert all(sorted(row) == [0, 0, 1, 2] for row in validation['seam_incidence'])
        for faces in mesh['faces']:
            edges = Counter(tuple(sorted((int(a), int(b)))) for face in faces
                            for a, b in zip(face, np.roll(face, -1)))
            assert max(edges.values()) <= 2
        assert np.isfinite(mesh['vertices']).all()
        assert mesh['diagnostics']['post_reconciliation'] is False
        assert mesh['diagnostics']['interior_topology'] == 'center_checked_not_topology_certified'
        assert mesh['diagnostics']['shared_face_max_disagreement'] == 0


def test_same_group_and_cross_bank_identities_cannot_merge(experiment):
    _, meshes, _, _ = experiment
    assert not set(meshes[0]['tagged_vertex_keys']) & set(meshes[1]['tagged_vertex_keys'])
    for bank, mesh in enumerate(meshes):
        assert mesh['surface_ids'] == [(bank, 0, 0), (bank, 1, 0), (bank, 2, 0), (bank, 2, 1)]
        assert mesh['directed_fault_pairs'] == [(0, 1), (0, 2), (0, 3)]
        assert mesh['directed_erosion_pairs'] == [(1, 2), (1, 3)]
        assert mesh['diagnostics']['interface'] == 'bank_side_fault_skin_not_single_conforming_fault_mesh'
        for key in mesh['vertex_keys']:
            stacks = [identity[0] for identity in key[1]]
            assert len(stacks) == len(set(stacks))


def test_production_mesh_and_all_observations_unchanged(experiment):
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.core.data.generic_grid import GenericGrid
    _, _, captured_solution, data = experiment
    inputs, options, descriptor = model7_combination_factory(mesh_extraction=True)
    original = copy.deepcopy(inputs)
    root_queries = data['captures'][0][0].xyz_to_interpolate
    root_grid = captured_solution.root_output.outputs[0].grid
    inputs.grid.custom_grid = GenericGrid(root_queries[root_grid.custom_grid_slice].copy())
    options.evaluation_options.compute_scalar_gradient = True
    baseline = compute_model(inputs, options, descriptor)
    np.testing.assert_array_equal(inputs.surface_points.sp_coords, original.surface_points.sp_coords)
    np.testing.assert_array_equal(inputs.orientations.dip_positions, original.orientations.dip_positions)
    np.testing.assert_array_equal(inputs.orientations.dip_gradients, original.orientations.dip_gradients)
    for actual, expected in zip(captured_solution.dc_meshes, baseline.dc_meshes):
        np.testing.assert_array_equal(actual.vertices, expected.vertices)
        np.testing.assert_array_equal(actual.edges, expected.edges)


def test_default_fault_guard_is_still_active(experiment):
    from gempy_engine.API.dual_contouring.topology_aware_extraction import extract_topology_aware
    from gempy_engine.core.data.stack_relation_type import StackRelationType as R
    _, _, _, data = experiment
    with pytest.raises(ValueError, match='unsupported_fault_extraction'):
        extract_topology_aware(data['axes'], None, [0, 1, 2, 2], [0, 0, 0, 1],
                               [R.FAULT, R.ERODE, R.BASEMENT], data['levels'],
                               ownership=None, gradient_samples=None,
                               faults_relations=np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=bool))


def test_finite_multiple_and_nonplanar_faults_rejected(experiment):
    from gempy_engine.core.data.kernel_classes.faults import FaultsData
    inputs, options, descriptor = model7_combination_factory()
    descriptor.stack_structure.faults_input_data = [FaultsData(finite_fault=object()), None, None]
    with pytest.raises(ValueError, match='unsupported_finite_fault'):
        capture_model7(inputs, options, descriptor)
    descriptor.stack_structure.faults_input_data = None
    descriptor.stack_structure.faults_relations[1, 2] = True
    with pytest.raises(ValueError, match='unsupported_model'):
        capture_model7(inputs, options, descriptor)
    _, _, _, data = experiment
    samples = data['banks'][0]['samples'].copy()
    samples[0, 1, 1, 1] += .01
    with pytest.raises(ValueError, match='unsupported_nonplanar_fault'):
        extract_bank_skin(data['axes'], samples, data['banks'][0]['gradients'], data['levels'], 0)


def test_invalid_bank_rejected(experiment):
    _, _, _, data = experiment
    with pytest.raises(ValueError, match='invalid_bank'):
        query_bank(data['captures'][1], np.zeros((1, 3)), None)
    with pytest.raises(ValueError, match='invalid_bank'):
        query_bank(data['captures'][0], np.zeros((1, 3)), 0)


def test_resolution16_rejects_without_fake_success():
    report, meshes, _, _ = run_case(16)
    assert report['status'] == 'unsupported_actual_model7'
    assert any(mesh is None for mesh in meshes)
    assert any(error is not None for error in report['bank_rejections'])
    assert report['production_triangles'] == [126, 112, 236, 344]
