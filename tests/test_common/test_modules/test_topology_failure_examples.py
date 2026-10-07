"""Characterize the gallery's current rejections; update when support is extended."""

import pytest
import numpy as np

from examples.topology_aware_comparison import parity_report, residual_report, shared_edges
from examples.topology_aware_failures import evaluate_case, extract_case, failure_cases
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor


@pytest.fixture(autouse=True)
def numpy_backend():
    saved = dict(engine_backend=BackendTensor.engine_backend, use_gpu=BackendTensor.use_gpu,
                 use_pykeops=BackendTensor.use_pykeops, dtype=BackendTensor.dtype,
                 grads=BackendTensor.COMPUTE_GRADS)
    keops = BackendTensor.pykeops_enabled
    BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False,
                                  use_pykeops=False, dtype='float64', grads=False)
    yield
    BackendTensor._change_backend(**saved)
    BackendTensor.pykeops_enabled = keops


@pytest.mark.parametrize('case', failure_cases(), ids=lambda case: case['name'])
def test_gallery_outcome(case):
    mesh, report = evaluate_case(case)
    if case['expected'] is None:
        assert report['status'] == 'accepted'
        assert report['triangles'] == sum(len(faces) for faces in mesh['faces'])
        assert all(len(faces) > 0 for faces in mesh['faces'])
        actual = shared_edges(mesh)
        assert set(map(tuple, actual)) == set(map(tuple, np.sort(mesh['seam_edges'], axis=1)))
        assert len(actual) > 0
        expected_triangles = {'control': [70, 126], 'curved': [122, 42]}
        assert [len(faces) for faces in mesh['faces']] == expected_triangles[case['name']]
        assert len(actual) == 7
        assert report['per_surface_triangles'] == expected_triangles[case['name']]
        assert report['shared_seam_edge_count'] == len(actual)
        with_reference = extract_case(case, include_reference=True)
        assert np.array_equal(mesh['vertices'], with_reference['vertices'])
        assert all(np.array_equal(faces, ref_faces) for faces, ref_faces in
                   zip(mesh['faces'], with_reference['faces']))
        assert parity_report(with_reference, with_reference['reference'])['passed']
        residuals = residual_report(mesh, case['fields'])
        assert all(r['shared_seam']['count'] == len(np.unique(actual)) for r in residuals)
        assert all(r['contact']['count'] > 0 for r in residuals)
        for residual, field, faces, affected in zip(
                residuals, case['fields'], mesh['faces'], mesh['affected_faces']):
            for region, selected in [('total', faces), ('contact', faces[affected]),
                                     ('shared_seam', actual)]:
                ids = np.unique(selected)
                assert residual[region]['count'] == len(ids)
                assert residual[region]['percentiles']['100'] == np.abs(field(mesh['vertices'][ids])).max()
    else:
        assert report['status'] == 'rejected'
        assert mesh is None
        assert report['error'].startswith(case['expected'] + ':')
    if case.get('comparison'):
        assert report['equivalent_unscaled_case'] == 'accepted'


@pytest.mark.parametrize('case', failure_cases(), ids=lambda case: case['name'])
def test_unchecked_gallery_emits_actual_meshes(case):
    mesh, report = evaluate_case(case, unsafe_diagnostics=True)
    assert report['status'] == 'unchecked'
    assert mesh['diagnostics']['unsafe_diagnostics']
    assert all(len(faces) > 0 for faces in mesh['faces'])
    assert np.isfinite(mesh['vertices']).all()
    assert bool(report['warnings']) == (case['expected'] is not None)
    if case['name'] == 'grid_aligned':
        assert sum(d['degenerate_triangles'] for d in report['defects']) == 14
    if case['name'] == 'multiway':
        assert report['warnings']['unsupported_multiway_junction: overwriting earlier contact pair'] == 1
    if case['name'] == 'coordinate_scale':
        assert report['warnings']['degenerate_dual_face: emitting triangle below area threshold'] == 98
        assert report['defects'][0]['degenerate_triangles'] == 0
