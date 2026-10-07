"""Real kriging exports and sampled-face joint DC, not manufactured mesh output."""

import copy

import numpy as np
import pytest

from examples.topology_aware_models import run_case
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.modules.dual_contouring.topology_extraction import CORNERS, sampled_face_segment


@pytest.fixture(autouse=True)
def numpy_cpu(monkeypatch):
    saved = dict(engine_backend=BackendTensor.engine_backend, use_gpu=BackendTensor.use_gpu,
                 use_pykeops=BackendTensor.use_pykeops, dtype=BackendTensor.dtype, grads=BackendTensor.COMPUTE_GRADS)
    keops = BackendTensor.pykeops_enabled
    BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
    monkeypatch.setenv('GEMPY_FLAT_STACKS', 'False')
    yield
    BackendTensor._change_backend(**saved)
    BackendTensor.pykeops_enabled = keops


def test_real_complex_pile_all_surfaces_and_verified_seams(monkeypatch):
    from tests.fixtures import simple_models
    from gempy_engine.API.dual_contouring.topology_aware_extraction import extract_topology_aware

    original = simple_models.unconformity_complex_factory()
    before = copy.deepcopy(original)
    monkeypatch.setattr(simple_models, 'unconformity_complex_factory', lambda: original)
    report, joint, solution, data = run_case('complex', 16)
    assert report['status'] == 'supported'
    assert report['mask_parity'] == [True, True, True]
    assert report['surface_count'] == 4
    assert data['surface_to_stack'] == [0, 1, 2, 2]
    assert data['surface_indices'] == [0, 0, 0, 1]
    assert report['triangle_counts'] == report['reference_triangle_counts'] == [450, 682, 180, 214]
    assert report['shared_seam_edges'] == 45
    assert report['reference_shared_seam_edges'] == 0
    assert report['actual_shared_edges_match']
    assert report['seam_incidence'].count([2, 1, 0, 0]) == 15
    assert report['seam_incidence'].count([0, 2, 1, 0]) == 15
    assert report['seam_incidence'].count([0, 2, 0, 1]) == 15
    assert report['noncontact_parity']['passed']
    assert report['quality']['noncontact']['count'] == 1248
    assert report['quality']['noncontact'] == report['reference_quality']['noncontact']
    assert all(d == dict(degenerate_triangles=0, duplicate_triangles=0) for d in report['defects'])
    diagnostics = joint['diagnostics']
    assert diagnostics['junction_cell_count'] == 48
    assert diagnostics['shared_face_count'] == 45
    assert diagnostics['shared_face_max_disagreement'] < 1e-10
    assert not diagnostics['unsafe_diagnostics'] and not diagnostics['unsafe_warnings']
    assert not diagnostics['post_reconciliation']
    assert diagnostics['center_approximant_max_residual'] > 1e-4
    assert len(solution.root_output.outputs) == 3
    for output in solution.root_output.outputs:
        slc = output.grid.custom_grid_slice
        assert slc.stop-slc.start == 17**3 + 16**3
        for values in (output.exported_fields.scalar_field, output.exported_fields.gx_field,
                       output.exported_fields.gy_field, output.exported_fields.gz_field):
            assert np.isfinite(values[slc]).all()
    np.testing.assert_array_equal(original[0].surface_points.sp_coords, before[0].surface_points.sp_coords)
    np.testing.assert_array_equal(original[0].orientations.dip_positions, before[0].orientations.dip_positions)
    np.testing.assert_array_equal(original[0].orientations.dip_gradients, before[0].orientations.dip_gradients)
    assert original[0].grid.custom_grid is None
    assert original[1] == before[1]
    np.testing.assert_array_equal(original[2].stack_structure.number_of_surfaces_per_stack,
                                  before[2].stack_structure.number_of_surfaces_per_stack)
    # Default experimental contract still rejects this conjunction, not a silent fallback.
    with pytest.raises(ValueError, match='unsupported_ownership:'):
        extract_topology_aware(**data)
    snapshot = copy.deepcopy(data)
    repeated = extract_topology_aware(**data, include_reference=True, sampled_junctions=True)
    np.testing.assert_array_equal(joint['vertices'], repeated['vertices'])
    assert joint['vertex_keys'] == repeated['vertex_keys']
    for key in ('scalar_samples', 'ownership', 'gradient_samples', 'interior_samples'):
        np.testing.assert_array_equal(data[key], snapshot[key])


@pytest.mark.parametrize('case,resolution', [('complex', 8), ('unconformity', 16)])
def test_real_piles_reject_competing_cell_participants(case, resolution):
    report, joint, solution, data = run_case(case, resolution)
    assert joint is None
    assert report['status'] == 'rejected'
    assert report['rejection'].startswith('unsupported_multiway_junction:')
    assert all(report['mask_parity'])
    assert len(solution.dc_meshes) == report['surface_count']
    assert sum(report['production_triangles']) > 0


def test_nonaffine_face_roots_are_shared_restrictions_not_cell_fits():
    # Neighbor cells differ away from their shared y-face. Its two restrictions
    # intersect at x=.4/(1+.2), z=.3+.1*x, exactly on both cells' face data.
    segments = []
    for offset in (0, 1):
        x, y, z = (CORNERS + [0, offset, 0]).T
        fields = np.array([x*(1+.2*y)-.4, z-.3-.1*x])
        segment = sampled_face_segment(fields)
        assert segment.shape == (2, 3)
        segment[:, 1] += offset
        segments.append(segment)
    a = segments[0][np.isclose(segments[0][:, 1], 1)][0]
    b = segments[1][np.isclose(segments[1][:, 1], 1)][0]
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
    np.testing.assert_allclose(a, [.4/1.2, 1., .3+.1*.4/1.2], atol=1e-12, rtol=0)


def test_sampled_face_coincident_restrictions_reject():
    x, y, z = CORNERS.T
    with pytest.raises(ValueError, match='degenerate_sampled_face:'):
        sampled_face_segment(np.array([x-.4, 2*(x-.4)]))


@pytest.mark.parametrize('level,error', [(.04, 'unsupported_sampled_junction:'), (0., 'degenerate_sampled_face:')])
def test_sampled_face_multiple_or_tangent_roots_reject(level, error):
    x, y, z = CORNERS.T
    with pytest.raises(ValueError, match=error):
        sampled_face_segment(np.array([x-y+.01*z, (x-.5)*(y-.5)-level]))
