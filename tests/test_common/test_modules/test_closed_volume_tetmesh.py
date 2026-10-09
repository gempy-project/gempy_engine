"""Closed unit volumes to conforming unit tetrahedra: needle collapse and TetGen."""

from collections import defaultdict

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.modules.dual_contouring.adaptive_closed_volumes import count_hanging_nodes


@pytest.fixture(autouse=True)
def cpu_float64():
    saved = dict(engine_backend=BT.engine_backend, use_gpu=BT.use_gpu,
                 use_pykeops=BT.use_pykeops, dtype=BT.dtype, grads=BT.COMPUTE_GRADS)
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
    yield
    BT._change_backend(**saved)


@pytest.fixture(scope='module')
def model7_volumes():
    from gempy_engine.API.model.model_api import compute_model
    from tests.test_common.test_modules.test_closed_volumes import _model7

    BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
    return compute_model(*_model7(9, 2, 0)).closed_volumes


def _min_angles(volumes):
    p = volumes.vertices[volumes.triangles]
    angles = []
    for i in range(3):
        u, v = p[:, (i+1) % 3]-p[:, i], p[:, (i+2) % 3]-p[:, i]
        cosine = np.einsum('ij,ij->i', u, v)/np.linalg.norm(u, axis=1)/np.linalg.norm(v, axis=1)
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return np.min(angles, axis=0)


def test_needle_collapse_keeps_closure_and_features(model7_volumes):
    collapsed = model7_volumes.collapse_short_edges()
    report = collapsed.report
    assert report['all_closed'] and report['all_positive'] and report['hanging_nodes'] == 0
    assert report['collapse']['collapsed_edges'] > 0
    assert np.sum(_min_angles(collapsed) < 5) < np.sum(_min_angles(model7_volumes) < 5)/5
    assert collapsed.units == model7_volumes.units
    assert set(collapsed.vertex_keys) <= set(model7_volumes.vertex_keys)
    total = sum(model7_volumes.volumes.values())
    assert max(abs(v) for v in report['collapse']['volume_change'].values()) < 1e-3*total
    # Box faces stay planar: extent vertices never leave the box.
    extent = collapsed.vertices[np.unique(collapsed.triangles[np.array(collapsed.patch_kinds) == 'extent'])]
    low, high = model7_volumes.vertices.min(axis=0), model7_volumes.vertices.max(axis=0)
    on_box = np.isclose(extent, low, atol=1e-12) | np.isclose(extent, high, atol=1e-12)
    assert on_box.any(axis=1).all()


def test_model7_unit_tetrahedra_conform_to_the_interfaces(model7_volumes):
    pytest.importorskip('tetgen')
    mesh = model7_volumes.tetrahedralize()
    surface = mesh.surface
    assert mesh.units == model7_volumes.units
    expected = surface.volumes
    for unit, volume in mesh.unit_volumes.items():
        assert volume == pytest.approx(expected[unit], rel=1e-9)
    p = mesh.nodes[mesh.tetrahedra]
    assert np.all(np.einsum('ij,ij->i', p[:, 1]-p[:, 0], np.cross(p[:, 2]-p[:, 0], p[:, 3]-p[:, 0])) > 0)
    # Conforming: every tetrahedron face is shared by two tetrahedra or lies on
    # the box; faces between different units lie on input triangles whose
    # sides are exactly those two units.
    owners = defaultdict(list)
    for t, tet in enumerate(mesh.tetrahedra.tolist()):
        for face in ((tet[1], tet[2], tet[3]), (tet[0], tet[3], tet[2]), (tet[0], tet[1], tet[3]),
                     (tet[0], tet[2], tet[1])):
            owners[frozenset(face)].append(t)
    patch_of = {frozenset(face): patch for face, patch in zip(mesh.boundary_faces.tolist(),
                                                              mesh.boundary_patches.tolist())}
    for face, tets in owners.items():
        assert len(tets) in (1, 2)
        units = {mesh.units[mesh.tet_units[t]] for t in tets}
        if len(tets) == 1:
            assert surface.patch_kinds[patch_of[face]] == 'extent'
        elif len(units) == 2:
            assert set(surface.unit_sides[patch_of[face]]) == units


def test_self_intersecting_input_is_rejected(model7_volumes):
    pytest.importorskip('tetgen')
    from gempy_engine.core.data.closed_unit_volumes import ClosedUnitVolumes

    # Push one vertex well past its neighbouring surfaces.
    vertices = model7_volumes.vertices.copy()
    vertices[model7_volumes.triangles[0, 0]] += (0., 0., .05)
    broken = ClosedUnitVolumes(vertices, model7_volumes.triangles, model7_volumes.unit_sides,
                               model7_volumes.patch_kinds, model7_volumes.vertex_keys, model7_volumes.report)
    with pytest.raises(ValueError, match='closed_volume_self_intersection'):
        broken.tetrahedralize(collapse_ratio=None, quality=False)


def test_synthetic_layers_tetrahedra_match_analytic_volumes():
    pytest.importorskip('tetgen')
    from gempy_engine.API.dual_contouring.adaptive_joint_extraction import _assemble_closed_volumes
    from tests.test_common.test_modules.test_closed_volumes import CASES, _synthetic

    volumes = _assemble_closed_volumes({0: _synthetic('two_in_stack', True)})
    assert volumes.report['hanging_nodes'] == 0
    mesh = volumes.tetrahedralize()
    analytic = {(0,) + unit[1:]: v for unit, v in CASES['two_in_stack'][3].items()}
    for unit, volume in mesh.unit_volumes.items():
        assert volume == pytest.approx(analytic[unit], rel=1e-9)
    assert count_hanging_nodes(mesh.surface.vertices, mesh.surface.triangles) == 0
