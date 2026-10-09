"""Closed unit volumes from joint extraction: finest rim lattice, shared shells."""

from itertools import product

import numpy as np
import pytest

import gempy_engine.API.dual_contouring.adaptive_topology_extraction as core
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.dual_contouring.adaptive_closed_volumes import (
    OUTSIDE, audit_shells, build_closed_volumes, classify_units, count_hanging_nodes,
)
from gempy_engine.modules.dual_contouring.topology_extraction import CORNERS


@pytest.fixture(autouse=True)
def cpu_float64():
    saved = dict(engine_backend=BT.engine_backend, use_gpu=BT.use_gpu,
                 use_pykeops=BT.use_pykeops, dtype=BT.dtype, grads=BT.COMPUTE_GRADS)
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
    yield
    BT._change_backend(**saved)


# region Synthetic analytic models through the core

def _leaves(mixed):
    origins, spans = [], []
    for origin in product(range(0, 8, 2), repeat=3):
        if mixed and origin[1] < 4:
            origins.append(origin)
            spans.append(2)
        else:
            for offset in product(range(2), repeat=3):
                origins.append(np.array(origin) + offset)
                spans.append(1)
    return np.asarray(origins), np.asarray(spans)


def _plane(c, a=0., b=0.):
    return (lambda x, y, z: z - c - a*x - b*y,
            lambda x, y, z: np.stack([-a + 0*x, -b + 0*x, 1 + 0*x], -1))


CASES = {
    'flat': ([_plane(3.4)], [R.BASEMENT], [[0.]], {(0, 0, 0): 4.6*64, (0, 0, 1): 3.4*64}),
    'tilted': ([_plane(2.5043, .2003, .1985)], [R.BASEMENT], [[0.]],
               {(0, 0, 0): 512 - 64*4.0995, (0, 0, 1): 64*4.0995}),
    'two_in_stack': ([_plane(0., .1)], [R.BASEMENT], [[2.35, 5.15]],
                     {(0, 0, 0): 64*(8 - 5.55), (0, 0, 1): 64*2.8, (0, 0, 2): 64*2.75}),
    'curved': ([(lambda x, y, z: z - 3.042 - .0664*(x-4.13)**2 - .033*(y-3.71)**2,
                 lambda x, y, z: np.stack([-.1328*(x-4.13), -.066*(y-3.71), 1+0*x], -1))],
               [R.BASEMENT], [[0.]], None),
    'erosion': ([_plane(2.0961, .0216, .5001), _plane(3.3778, .0582)], [R.ERODE, R.BASEMENT],
                [[0.], [0.]], None),
}


def _synthetic(case, mixed):
    stacks, relations, isovalues, _ = CASES[case]
    groups = [g for g, iso in enumerate(isovalues) for _ in iso]
    indices = [k for iso in isovalues for k in range(len(iso))]

    def sample_stacks(points):
        x, y, z = np.asarray(points).reshape(-1, 3).T
        return (np.array([f(x, y, z) for f, _ in stacks]),
                np.array([np.broadcast_to(g(x, y, z), (len(x), 3)) for _, g in stacks], dtype=float))

    def sample_fields(points):
        values, gradients = sample_stacks(points)
        return values[groups], gradients[groups]

    origins, spans = _leaves(mixed)
    corners = (origins[:, None] + spans[:, None, None]*CORNERS).reshape(-1, 3)
    samples = sample_fields(corners)[0].reshape(len(groups), -1, 8)
    result = core.extract_adaptive_topology(
        origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples, np.array(groups), np.array(indices), relations,
        [list(map(float, i)) for i in isovalues], sample_fields=sample_fields, closure=True)
    return build_closed_volumes(result['closure'], sample_stacks=sample_stacks, relations=relations,
                                stack_isovalues=[np.array(i, float) for i in isovalues], region_volume=512.)


@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('case', sorted(CASES))
def test_synthetic_unit_shells_close(case, mixed):
    part = _synthetic(case, mixed)
    report = part['report']
    assert report['all_closed'] and report['all_positive'] and report['volumes_sum_to_region']
    assert report['non_interface_triangles'] == 0
    expected = CASES[case][3]
    if expected is not None:  # planes: dual contouring and the rim are exact
        assert set(report['volumes']) == set(expected)
        for unit, volume in expected.items():
            assert report['volumes'][unit] == pytest.approx(volume, rel=1e-9)
    if mixed:
        assert report['fan_triangles'] > 0  # coarse rim tiles close through fans


def test_curved_volume_converges_to_analytic():
    # Analytic volume under z = 3.042 + .0664(x-4.13)^2 + .033(y-3.71)^2 over [0, 8]^2.
    mean = 3.042 + .0664*(64/3 - 8*4.13 + 4.13**2) + .033*(64/3 - 8*3.71 + 3.71**2)
    volume = _synthetic('curved', False)['report']['volumes'][(0, 0, 1)]
    assert volume == pytest.approx(64*mean, rel=2e-3)


def test_erosion_contact_reaching_the_extent():
    report = _synthetic('erosion', True)['report']
    # Eroding stack above, base horizon split into two intervals below it.
    assert set(report['volumes']) == {(0, 0, 0), (0, 1, 0), (0, 1, 1)}
    assert report['rim_interface_edges'] > 0

# endregion


# region Building blocks

def test_audit_detects_an_open_shell():
    part = _synthetic('flat', False)
    unit = next(iter(part['shells']))
    shells = {unit: part['shells'][unit][1:]}
    report = audit_shells(shells, part['positions'])
    assert not report['all_closed'] and report['open_edges'][unit] > 0


def test_unit_classifier_matches_production_masks():
    from gempy_engine.API.model.model_api import compute_model
    from tests.fixtures.model7_combination import model7_combination_factory

    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=False)
    solution = compute_model(inputs, options, descriptor)
    outputs = solution.octrees_output[-1].outputs
    size = len(solution.octrees_output[-1].grid.values)
    values = np.array([BT.t.to_numpy(o.exported_fields.scalar_field)[:size] for o in outputs])
    isovalues = [BT.t.to_numpy(o.scalar_field_at_sp).reshape(-1) for o in outputs]
    masks = np.array([BT.t.to_numpy(o.squeezed_mask_array)[:size] for o in outputs])
    active, interval = classify_units(values, descriptor.stack_structure.masking_descriptor, isovalues)
    np.testing.assert_array_equal(active, np.argmax(masks, axis=0))
    assert interval.min() >= 0

# endregion


# region Production models

def _model7(root, depth, min_level, mode='joint', closed=True):
    from tests.fixtures.model7_combination import model7_combination_factory
    from gempy_engine.core.data.regular_grid import RegularGrid

    inputs, options, descriptor = model7_combination_factory(number_octree_levels=depth, mesh_extraction=True)
    extent = inputs.grid.octree_grid.orthogonal_extent
    extent = (extent.detach().cpu().numpy() if hasattr(extent, 'detach') else np.asarray(extent)).copy()
    inputs.grid.octree_grid = RegularGrid(extent, [root] * 3)
    evaluation = options.evaluation_options
    evaluation.number_octree_levels = depth
    evaluation.number_octree_levels_surface = depth
    evaluation.octree_min_level = min_level
    evaluation.mesh_extraction_overlap = mode
    evaluation.mesh_extraction_closed_volumes = closed
    return inputs, options, descriptor


@pytest.mark.parametrize('case', [(9, 2, 0), (7, 2, 2)])
def test_model7_banked_unit_shells(case):
    from gempy_engine.API.model.model_api import compute_model

    solution = compute_model(*_model7(*case))
    volumes = solution.closed_volumes
    report = volumes.report
    assert report['all_closed']
    assert len(volumes.units) == 8 and {u[0] for u in volumes.units} == {0, 1}
    for bank, bank_report in report['banks'].items():
        assert bank_report['all_closed'] and bank_report['all_positive']
        assert sum(bank_report['volumes'].values()) == pytest.approx(bank_report['region_volume'], rel=1e-9)
    assert sum(report['region_volumes'].values()) == pytest.approx(
        np.prod(np.diff(np.asarray(solution.octrees_output[0].grid.octree_grid.orthogonal_extent).reshape(3, 2))),
        rel=1e-12)
    assert set(volumes.patch_kinds) == {'interface', 'fault_skin', 'extent'}
    # One shared fault interface: no hanging nodes anywhere, and every fault
    # triangle separates a bank-0 unit from a bank-1 unit.
    assert report['hanging_nodes'] == 0
    assert count_hanging_nodes(volumes.vertices, volumes.triangles) == 0
    fault_sides = [sides for sides, kind in zip(volumes.unit_sides, volumes.patch_kinds) if kind == 'fault_skin']
    assert fault_sides and all(sorted(u[0] for u in sides) == [0, 1] for sides in fault_sides)
    # Every unit shell is closed by itself, read back through the public API.
    for unit in volumes.units:
        vertices, faces = volumes.shell(unit)
        directed = {(a, b) for f in faces.tolist() for a, b in zip(f, f[1:] + f[:1])}
        assert all((b, a) in directed for a, b in directed)
        assert len(directed) == 3*len(faces)
    # Surfaces now reach the extent.
    extent = np.asarray(solution.octrees_output[0].grid.octree_grid.orthogonal_extent).reshape(3, 2)
    for mesh in solution.dc_meshes:
        on_rim = np.isclose(mesh.vertices[:, None], extent[None, :, 0]) | np.isclose(mesh.vertices[:, None], extent[None, :, 1])
        assert on_rim.any()


def _square_skin(seam_axis, seam, units, bank_above):
    """A unit-square fault skin in z=0 split by a seam line, as one bank's patches."""
    lo, hi = ('lo', seam_axis), ('hi', seam_axis)
    points = {('c', 0): (0, 0), ('c', 1): (1, 0), ('c', 2): (1, 1), ('c', 3): (0, 1)}
    points[lo] = (seam, 0) if seam_axis == 0 else (0, seam)
    points[hi] = (seam, 1) if seam_axis == 0 else (1, seam)
    if seam_axis == 0:
        halves = [[('c', 0), lo, hi, ('c', 3)], [lo, ('c', 1), ('c', 2), hi]]
    else:
        halves = [[('c', 0), ('c', 1), hi, lo], [lo, hi, ('c', 2), ('c', 3)]]
    patches = []
    for quad, unit in zip(halves, units):
        for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
            patches.append((tri, 'fault_skin', OUTSIDE, unit) if bank_above else (tri, 'fault_skin', unit, OUTSIDE))
    positions = {k: np.array([x, y, 0.]) for k, (x, y) in points.items()}
    return positions, patches


def test_shared_fault_interface_conforms_crossing_seams():
    from gempy_engine.modules.dual_contouring.shared_fault_interface import merge_fault_skins

    below = _square_skin(0, .4, [(0, 1, 0), (0, 1, 1)], bank_above=False)
    above = _square_skin(1, .6, [(1, 1, 0), (1, 1, 1)], bank_above=True)
    # A bank-0 horizon triangle standing on its seam; bank 1's seam crosses it.
    below[0][('h', 0)] = np.array([.4, .5, -1.])
    below[1].append(((('lo', 0), ('hi', 0), ('h', 0)), 'interface', (0, 1, 0), (0, 1, 1)))
    parts = {b: dict(positions=positions, patches=patches) for b, (positions, patches) in enumerate((below, above))}
    patches, positions, report = merge_fault_skins(parts, [0, 0, 1.])
    assert report['constraint_crossings'] == 1 and report['split_triangles'] == 1
    fault = [p for p in patches if p[1] == 'fault_skin']
    area = 0.
    for keys, _, negative, positive in fault:
        a, b, c = (positions[k] for k in keys)
        normal = np.cross(b-a, c-a)
        assert normal[2] > 0  # wound along the fault normal
        area += normal[2]/2
        centroid = (a+b+c)/3
        assert negative == (0, 1, int(centroid[0] > .4)) and positive == (1, 1, int(centroid[1] > .6))
    assert area == pytest.approx(1.)
    keys = sorted({k for p in patches for k in p[0]}, key=repr)
    ids = {k: i for i, k in enumerate(keys)}
    assert count_hanging_nodes(np.array([positions[k] for k in keys]),
                               np.array([[ids[k] for k in p[0]] for p in patches])) == 0
    # The standing triangle is split at the crossing, keeping its area.
    standing = [p[0] for p in patches if p[1] == 'interface']
    areas = [np.linalg.norm(np.cross(positions[b]-positions[a], positions[c]-positions[a]))/2 for a, b, c in standing]
    assert len(standing) == 2 and min(areas) > 0 and sum(areas) == pytest.approx(.5)


def test_model7_option_off_meshes_are_contained():
    from gempy_engine.API.model.model_api import compute_model

    plain = compute_model(*_model7(9, 2, 0, closed=False)).dc_meshes
    closed = compute_model(*_model7(9, 2, 0)).dc_meshes
    assert all(getattr(mesh, 'closed_volumes', None) is None for mesh in plain)
    for before, after in zip(plain, closed):
        keys_before = [before.joint_vertex_keys[i] for i in before.edges.ravel()]
        keys_after = [after.joint_vertex_keys[i] for i in after.edges.ravel()]
        triangles_before = {frozenset(keys_before[i:i+3]) for i in range(0, len(keys_before), 3)}
        triangles_after = {frozenset(keys_after[i:i+3]) for i in range(0, len(keys_after), 3)}
        assert triangles_before < triangles_after


def test_fault_free_production_model_closes():
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.core.data import TensorsStructure
    from gempy_engine.core.data.engine_grid import EngineGrid
    from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
    from gempy_engine.core.data.interpolation_input import InterpolationInput
    from gempy_engine.core.data.kernel_classes.orientations import Orientations
    from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
    from gempy_engine.core.data.options import InterpolationOptions
    from gempy_engine.core.data.regular_grid import RegularGrid
    from gempy_engine.core.data.stacks_structure import StacksStructure

    stacks = StacksStructure(np.array([3]), np.array([1]), np.array([1]), [R.BASEMENT])
    descriptor = InputDataDescriptor(TensorsStructure(np.array([3])), stacks)
    grid = EngineGrid(octree_grid=RegularGrid([0., 1., 0., 1., 0., 1.], [4, 4, 4]))
    inputs = InterpolationInput(SurfacePoints(np.array([[.1, .1, .37], [.9, .1, .37], [.1, .9, .37]])),
                                Orientations(np.array([[.5, .5, .37]]), np.array([[0., 0., 1.]])), grid)
    options = InterpolationOptions.from_args(range=2., c_o=1.)
    options.evaluation_options.number_octree_levels = 2
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.mesh_extraction_overlap = 'joint'
    options.evaluation_options.mesh_extraction_closed_volumes = True
    volumes = compute_model(inputs, options, descriptor).closed_volumes
    assert volumes.report['all_closed'] and len(volumes.units) == 2
    assert sum(volumes.volumes.values()) == pytest.approx(1., rel=1e-12)


@pytest.mark.parametrize('mode, error', [('joint_contacts', NotImplementedError), ('pretty', ValueError)])
def test_closed_volumes_rejected_outside_banked_or_fault_free_joint(mode, error):
    from gempy_engine.API.model.model_api import compute_model

    with pytest.raises(error, match='unsupported_closed_volumes'):
        compute_model(*_model7(7, 2, 2, mode=mode))


def test_closed_volumes_rejected_with_unaffected_stacks():
    """A stack the fault does not cut is not split into banks: no bank-wise shells."""
    from tests.test_common.test_modules.test_adaptive_fault_integration import _fault_case
    from gempy_engine.API.dual_contouring.adaptive_fault_extraction import extract_fault_joint_octree

    descriptor, inputs, options, levels = _fault_case()  # stack 2 is unaffected
    options.evaluation_options.mesh_extraction_closed_volumes = True
    with pytest.raises(ValueError, match='unsupported_closed_volumes'):
        extract_fault_joint_octree(descriptor, inputs, options, levels)


def test_model7_closed_volumes_on_cuda_match_numpy():
    from gempy_engine.API.model.model_api import compute_model

    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    reference = compute_model(*_model7(7, 2, 2)).closed_volumes
    BT._change_backend(AvailableBackends.PYTORCH, use_gpu=True, use_pykeops=False, dtype='float64', grads=False)
    try:
        gpu = compute_model(*_model7(7, 2, 2)).closed_volumes
    finally:
        BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
        torch.set_default_device('cpu')
    assert gpu.report['all_closed'] and gpu.report['hanging_nodes'] == 0
    assert gpu.units == reference.units and set(gpu.vertex_keys) == set(reference.vertex_keys)

    def keyed(volumes, fault):
        return {tuple(volumes.vertex_keys[i] for i in tri) for tri, kind in zip(volumes.triangles.tolist(),
                                                                                volumes.patch_kinds)
                if (kind == 'fault_skin') == fault}
    # Same keyed horizons and caps. The shared fault interface is a geometric
    # (Delaunay) triangulation, so near-cocircular choices may differ.
    assert keyed(gpu, False) == keyed(reference, False)
    assert len(keyed(gpu, True)) == len(keyed(reference, True))
    for unit, volume in reference.volumes.items():
        # Same topology; production NumPy and Torch fields differ slightly
        # (~3e-5 in vertices), so compare against the whole model volume.
        assert gpu.volumes[unit] == pytest.approx(volume, abs=1e-4*sum(reference.volumes.values()))

# endregion
