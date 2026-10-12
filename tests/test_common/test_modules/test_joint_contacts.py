"""joint_contacts: joint erosion/onlap contacts with pretty's fault vertex merge."""

import importlib
from collections import Counter
from itertools import product
from unittest.mock import Mock

import numpy as np
import pytest

import gempy_engine.API.dual_contouring.joint_topology as core
from gempy_engine.config import AvailableBackends, DualContouringOverlap, resolve_dual_contouring_overlap
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.modules.dual_contouring.joint_cell_branches import CORNERS


@pytest.fixture(autouse=True)
def cpu_float64():
    saved = dict(engine_backend=BT.engine_backend, use_gpu=BT.use_gpu,
                 use_pykeops=BT.use_pykeops, dtype=BT.dtype, grads=BT.COMPUTE_GRADS)
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype="float64", grads=False)
    yield
    BT._change_backend(**saved)


# region Synthetic core cases

def merge_case(*, planes=(3.3,), erosion=None, fine=False):
    """Planar faults at ``x = planes``, one horizon with a step throw across each
    (the production drift is a step), optionally an eroder ``z - e0 - ey*y``."""
    origins, spans = [], []
    for origin in product(range(0, 8, 2), repeat=3):
        if origin[1] < 4 and not fine:
            origins.append(origin)
            spans.append(2)
        else:
            for offset in product(range(2), repeat=3):
                origins.append(np.array(origin) + offset)
                spans.append(1)
    origins, spans = np.asarray(origins), np.asarray(spans)
    relations = ([R.ERODE] if erosion else []) + [R.FAULT] * len(planes) + [R.BASEMENT]
    n_stacks = len(relations)
    fault_stacks = [i for i, relation in enumerate(relations) if relation is R.FAULT]
    matrix = np.zeros((n_stacks, n_stacks), dtype=bool)
    matrix[fault_stacks, n_stacks - 1] = True
    groups = np.arange(n_stacks)
    indices = np.zeros(n_stacks, dtype=int)
    levels = [[0.]] * n_stacks

    def sample(points):
        x, y, z = np.asarray(points).T
        values, gradients = [], []
        if erosion:
            values.append(z - erosion[0] - erosion[1]*y)
            gradients.append(np.broadcast_to([0., -erosion[1], 1.], (len(x), 3)))
        for plane in planes:
            values.append(x - plane)
            gradients.append(np.broadcast_to([1., 0., 0.], (len(x), 3)))
        values.append(z - 3.4 - sum(.45*(x > plane) for plane in planes))
        gradients.append(np.broadcast_to([0., 0., 1.], (len(x), 3)))
        return np.asarray(values), np.asarray(gradients, dtype=float)

    samples = sample((origins[:, None] + spans[:, None, None]*CORNERS).reshape(-1, 3))[0]
    args = (origins, spans, (8, 8, 8), (0, 8, 0, 8, 0, 8), samples.reshape(n_stacks, -1, 8),
            groups, indices, relations, levels)
    merge = {int(f): {n_stacks - 1} for f in fault_stacks}
    return args, dict(sample_fields=sample, faults_relations=matrix, fault_merge=merge)


@pytest.mark.parametrize('fine', [False, True])
def test_layer_borrows_fault_vertex_identities(fine):
    args, kwargs = merge_case(fine=fine)
    result = core.extract_adaptive_topology(*args, **kwargs)
    fault, layer = 0, 1
    keys = result['vertex_keys']
    fault_ids = set(np.unique(result['faces'][fault]).tolist())
    layer_ids = set(np.unique(result['faces'][layer]).tolist())
    shared = fault_ids & layer_ids
    assert shared, 'the horizon must end on fault vertices'
    assert all(keys[i][1] == ((fault, 0),) for i in shared)
    # Same identity, hence identical positions by construction; no all-fault layer triangle.
    assert not any(set(tri) <= fault_ids for tri in result['faces'][layer].tolist())
    edges = Counter(tuple(sorted(e)) for tri in result['faces'][layer].tolist()
                    for e in zip(tri, tri[1:] + tri[:1]))
    assert max(edges.values()) <= 2
    diagnostics = result['diagnostics']
    assert diagnostics['contact_type'] == 'pretty_fault_vertex_merge'
    assert diagnostics['borrowed_vertex_count'] > 0
    assert diagnostics['weighted_fault_vertex_count'] > 0
    assert diagnostics['fault_overridden_junction_cells'] == []
    assert 'bank' not in result


def test_weighted_fault_vertex_lies_on_layer_contact():
    """The borrowed fault vertex carries the layer's Hermite rows (pretty's weight)."""
    args, kwargs = merge_case(fine=True)
    result = core.extract_adaptive_topology(*args, **kwargs)
    vertices = np.asarray(result['vertices'])
    keys = result['vertex_keys']
    shared = set(np.unique(result['faces'][0]).tolist()) & set(np.unique(result['faces'][1]).tolist())
    for i in shared:
        x, y, z = vertices[i]
        assert abs(x - 3.3) < 1e-9  # on the fault
        # Between the two throw sides of the horizon, pulled onto the step.
        assert 3.4 - 1e-9 <= z <= 3.85 + 1e-9, keys[i]


def test_two_independent_faults():
    args, kwargs = merge_case(planes=(3.3, 5.7))
    result = core.extract_adaptive_topology(*args, **kwargs)
    keys = result['vertex_keys']
    layer = 2
    sources = {keys[i][1][0][0] for i in np.unique(result['faces'][layer]) if keys[i][1][0][0] != layer}
    assert sources == {0, 1}
    assert result['diagnostics']['filtered_fault_triangle_count'] > 0


def test_leaf_crossed_by_two_faults_keeps_its_own_vertex():
    args, kwargs = merge_case(planes=(3.3, 3.6))
    result = core.extract_adaptive_topology(*args, **kwargs)
    diagnostics = result['diagnostics']
    assert diagnostics['multi_fault_unborrowed_count'] > 0
    assert diagnostics['fallback_region_leaf_count'] > 0
    keys = result['vertex_keys']
    layer = 2
    own = {keys[i] for i in np.unique(result['faces'][layer]) if keys[i][1] == ((layer, 0),)}
    assert own, 'the layer keeps its own vertices'


def test_contact_junction_near_fault_falls_back_and_is_counted():
    args, kwargs = merge_case(erosion=(2.05, .5), fine=True)
    result = core.extract_adaptive_topology(*args, **kwargs)
    diagnostics = result['diagnostics']
    assert diagnostics['fault_overridden_junction_cells']
    assert diagnostics['fallback_region_leaf_count'] == 72
    # Junctions away from the fault are still shared and seam-validated.
    assert len(result['junction_cells']) > 0 and len(result['seam_edges']) > 0
    fallback = set(diagnostics['fault_overridden_junction_cells'])
    assert not fallback & set(result['junction_cells'].tolist())


@pytest.mark.parametrize('contract, match', [
    ({}, 'invalid_fault_merge_contract'),
    ({0: set()}, 'invalid_fault_merge_contract'),
    ({0: {1}, 1: {1}}, 'invalid_fault_merge_contract'),
    (None, 'unsupported_fault_extraction'),
])
def test_fault_merge_contract_is_exact(contract, match):
    args, kwargs = merge_case()
    kwargs['fault_merge'] = contract
    with pytest.raises(ValueError, match=match):
        core.extract_adaptive_topology(*args, **kwargs)


def _targeted(kwargs, log=None):
    """field_query built from the analytic callback; full queries forbidden."""
    full = kwargs['sample_fields']

    def field_query(requests):
        answers = []
        for points, surfaces, kind in requests:
            if log is not None:
                log.append((len(points), tuple(surfaces), kind == 'gradient'))
            values, normals = full(points)
            answers.append(normals[surfaces] if kind == 'gradient' else values[surfaces])
        return answers

    def forbidden(points):
        raise AssertionError('full sample_fields query in field_query mode')

    return dict(kwargs, sample_fields=forbidden, field_query=field_query)


@pytest.mark.parametrize('case', [dict(), dict(fine=True), dict(planes=(3.3, 5.7)),
                                  dict(erosion=(2.05, .5), fine=True)])
def test_targeted_queries_match_full_queries(case):
    args, kwargs = merge_case(**case)
    full = core.extract_adaptive_topology(*args, **kwargs)
    log = []
    targeted = core.extract_adaptive_topology(*args, **_targeted(kwargs, log))
    assert targeted['vertex_keys'] == full['vertex_keys']
    for a, b in zip(targeted['faces'], full['faces']):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(targeted['vertices'], full['vertices'], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(targeted['seam_edges'], full['seam_edges'])
    # Gradients only for single surfaces; all-surface queries are scalar-only.
    assert all(len(surfaces) == 1 for _, surfaces, gradients in log if gradients)
    assert all(not gradients for _, surfaces, gradients in log if len(surfaces) > 1)
    # Fewer point-by-surface evaluations than every surface at every full-query point.
    evaluations = sum(count*len(surfaces) for count, surfaces, _ in log)
    assert evaluations < full['diagnostics']['sampled_point_count']*len(args[5])


def test_targeted_queries_reject_disagreeing_shared_nodes():
    args, kwargs = merge_case()
    samples = np.array(args[4])
    samples[1, 0, 7] += 1e-6  # one leaf's corner disagrees with its neighbours
    args = args[:4] + (samples,) + args[5:]
    with pytest.raises(ValueError, match='inconsistent_corner_samples'):
        core.extract_adaptive_topology(*args, **_targeted(kwargs))


# endregion


# region Mode plumbing

def test_mode_resolution_and_exclusivity():
    assert resolve_dual_contouring_overlap('joint_contacts') is DualContouringOverlap.joint_contacts
    for combined in (DualContouringOverlap.joint_contacts | DualContouringOverlap.pretty,
                     DualContouringOverlap.joint_contacts | DualContouringOverlap.joint):
        with pytest.raises(ValueError):
            resolve_dual_contouring_overlap(combined)


def test_no_fault_model_is_exactly_joint():
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.core.data import TensorsStructure
    from gempy_engine.core.data.engine_grid import EngineGrid
    from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
    from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
    from gempy_engine.core.data.interpolation_input import InterpolationInput
    from gempy_engine.core.data.kernel_classes.orientations import Orientations
    from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
    from gempy_engine.core.data.options import InterpolationOptions
    from gempy_engine.core.data.regular_grid import RegularGrid
    from gempy_engine.core.data.stacks_structure import StacksStructure

    def run(mode):
        normals, isovalues = [np.array([0., 0., 1.]), np.array([1., 0., .2])], [.37, .43]
        functions = [CustomInterpolationFunctions(
            scalar_field_at_surface_points=np.array([iso]),
            implicit_function=lambda xyz, normal=normal: xyz @ normal,
            gx_function=lambda xyz, normal=normal: np.full(len(xyz), normal[0]),
            gy_function=lambda xyz, normal=normal: np.full(len(xyz), normal[1]),
            gz_function=lambda xyz, normal=normal: np.full(len(xyz), normal[2]),
        ) for normal, iso in zip(normals, isovalues)]
        stacks = StacksStructure(np.zeros(2, dtype=int), np.zeros(2, dtype=int), np.ones(2, dtype=int),
                                 [R.ERODE, R.BASEMENT], interp_functions_per_stack=functions)
        descriptor = InputDataDescriptor(TensorsStructure(np.array([], dtype=int)), stacks)
        grid = EngineGrid(octree_grid=RegularGrid([0., 1., 0., 1., 0., 1.], [4, 4, 4]))
        inputs = InterpolationInput(SurfacePoints(np.empty((0, 3))),
                                    Orientations(np.empty((0, 3)), np.empty((0, 3))), grid)
        options = InterpolationOptions.from_args(range=1., c_o=1.)
        options.evaluation_options.number_octree_levels = 2
        options.evaluation_options.number_octree_levels_surface = 2
        options.evaluation_options.mesh_extraction_overlap = mode
        return compute_model(inputs, options, descriptor).dc_meshes

    joint, contacts = run('joint'), run('joint_contacts')
    assert len(joint) == len(contacts) == 2
    for a, b in zip(joint, contacts):
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(a.edges, b.edges)
        assert a.joint_vertex_keys == b.joint_vertex_keys
        np.testing.assert_array_equal(a.joint_seam_edges, b.joint_seam_edges)

# endregion


# region Production models

def _model7(root, depth, min_level, mode):
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
    return inputs, options, descriptor


def _audit_no_solves(monkeypatch):
    """Fail if any weight solve happens while the drift sampler queries."""
    bridge = importlib.import_module('gempy_engine.API.dual_contouring.joint_extraction')
    prepare = bridge.prepare_fault_drift_sampler
    prepared = []

    def audited(*args):
        sampler = prepare(*args)
        no_solve = Mock(side_effect=AssertionError('query attempted a production solve'))
        for name in ('gempy_engine.API.interp_single._interp_scalar_field',
                     'gempy_engine.API.interp_single._stack_ops',
                     'gempy_engine.API.interp_single._interp_single_feature'):
            module = importlib.import_module(name)
            if hasattr(module, 'compute_weights'):
                monkeypatch.setattr(module, 'compute_weights', no_solve)
        prepared.append(sampler)
        return sampler

    monkeypatch.setattr(bridge, 'prepare_fault_drift_sampler', audited)
    return prepared


def _shared_fault_vertices(meshes):
    fault_keys = dict(zip(meshes[0].joint_vertex_keys, meshes[0].vertices))
    shared = {}
    for mesh in meshes[1:]:
        for key, vertex in zip(mesh.joint_vertex_keys, mesh.vertices):
            if key in fault_keys:
                np.testing.assert_array_equal(vertex, fault_keys[key])
                shared[key] = vertex
    return shared


def test_model7_natural_adaptive(monkeypatch):
    from gempy_engine.API.model.model_api import compute_model

    prepared = _audit_no_solves(monkeypatch)
    meshes = compute_model(*_model7(9, 2, 0, 'joint_contacts')).dc_meshes
    assert len(prepared) == 1
    assert [(m.stack_index, m.surface_index) for m in meshes] == [(0, 0), (1, 0), (2, 0), (2, 1)]
    report = meshes[0].contact_report
    assert report['interface'] == 'pretty_fault_vertex_merge_single_fault_mesh'
    assert report['fault_cell_count'] == {0: 414}
    assert report['borrowed_vertex_count'] == 262
    assert report['fault_overridden_junction_cells'] == []
    assert len(meshes[0].joint_seam_edges) > 0
    shared = _shared_fault_vertices(meshes)
    assert len(shared) > 0
    fault_keys = set(meshes[0].joint_vertex_keys)
    for mesh in meshes[1:]:
        borrowed = np.array([key in fault_keys for key in mesh.joint_vertex_keys])
        assert borrowed.any()
        assert not borrowed[mesh.edges].all(axis=1).any()
        edges = Counter(tuple(sorted(e)) for tri in mesh.edges.tolist() for e in zip(tri, tri[1:] + tri[:1]))
        assert max(edges.values()) <= 2


def test_model7_contact_vertices_match_pretty_fault_vertices():
    """Fully refined, so both modes see identical final-level fault cells."""
    from gempy_engine.API.model.model_api import compute_model

    contacts = compute_model(*_model7(7, 2, 2, 'joint_contacts')).dc_meshes
    pretty = compute_model(*_model7(7, 2, 2, 'pretty')).dc_meshes
    shared = np.array(list(_shared_fault_vertices(contacts).values()))
    assert len(shared) > 0
    distance = np.min(np.linalg.norm(shared[:, None] - pretty[0].vertices[None], axis=-1), axis=1)
    assert distance.max() < 1e-9
    assert len(contacts[0].edges) == len(pretty[0].edges)


def test_model7_unsupported_hierarchy_still_rejects():
    from gempy_engine.API.model.model_api import compute_model

    with pytest.raises(ValueError, match='unsupported_hanging_branch'):
        compute_model(*_model7(7, 2, 0, 'joint_contacts'))


def _graben(levels):
    from tests.fixtures.complex_geometries import graben_fault_model
    from gempy_engine.core.data.options import MeshExtractionMaskingOptions

    make = getattr(graben_fault_model, '__wrapped__', graben_fault_model)
    inputs, structure, options = make()
    structure.stack_structure.faults_relations = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
    evaluation = options.evaluation_options
    evaluation.mesh_extraction = True
    evaluation.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    evaluation.number_octree_levels = levels
    evaluation.number_octree_levels_surface = levels
    evaluation.mesh_extraction_overlap = 'joint_contacts'
    return inputs, options, structure


def test_graben_two_independent_faults():
    from gempy_engine.API.model.model_api import compute_model

    meshes = compute_model(*_graben(4)).dc_meshes
    assert len(meshes) == 6 and all(len(m.edges) for m in meshes)
    report = meshes[0].contact_report
    assert report['fault_surfaces'] == [0, 1]
    assert set(report['sampler']['cache_fingerprints']) and len(report['sampler']['drift_rebuild_max_error']) == 2
    fault_keys = {0: set(meshes[0].joint_vertex_keys), 1: set(meshes[1].joint_vertex_keys)}
    for mesh in meshes[2:]:
        sources = {f for f, keys in fault_keys.items() if keys & set(mesh.joint_vertex_keys)}
        assert sources == {0, 1}


def test_graben_coarse_cell_shared_by_both_faults_falls_back():
    from gempy_engine.API.model.model_api import compute_model

    meshes = compute_model(*_graben(3)).dc_meshes
    assert len(meshes) == 6 and all(len(m.edges) for m in meshes)
    assert meshes[0].contact_report['multi_fault_unborrowed_count'] > 0

# endregion


# region Torch, CUDA and KeOps

def _torch_backend(kind):
    torch = pytest.importorskip('torch')
    if kind != 'torch_cpu' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if kind == 'keops_gpu':
        pytest.importorskip('pykeops')
    BT._change_backend(AvailableBackends.PYTORCH, use_gpu=kind != 'torch_cpu',
                       use_pykeops=kind == 'keops_gpu', dtype='float64', grads=False)
    return torch


def _host_meshes(meshes):
    host = lambda a: a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)
    return [(host(m.vertices), host(m.edges), list(m.joint_vertex_keys)) for m in meshes]


def _restore_numpy(torch):
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype='float64', grads=False)
    torch.set_default_device('cpu')


@pytest.mark.parametrize('kind', ['torch_cpu', 'torch_gpu', 'keops_gpu'])
def test_model7_on_torch_backends(monkeypatch, kind):
    from gempy_engine.API.model.model_api import compute_model

    reference = _host_meshes(compute_model(*_model7(7, 2, 2, 'joint_contacts')).dc_meshes)
    torch = _torch_backend(kind)
    try:
        prepared = _audit_no_solves(monkeypatch)
        meshes = compute_model(*_model7(7, 2, 2, 'joint_contacts')).dc_meshes
    finally:
        _restore_numpy(torch)
    assert len(prepared) == 1
    sampler_backend = prepared[0]['diagnostics']['backend']
    assert sampler_backend.startswith('PYTORCH_' + ('cpu' if kind == 'torch_cpu' else 'cuda'))
    result = _host_meshes(meshes)
    if kind == 'keops_gpu':
        # Production KeOps evaluation itself differs from dense kernels (~1e-4 on
        # Model 7), so only the joint_contacts invariants are asserted.
        fault_keys = set(result[0][2])
        for vertices, faces, keys in result[1:]:
            borrowed = np.array([key in fault_keys for key in keys])
            assert borrowed.any() and not borrowed[faces].all(axis=1).any()
        return
    for (v0, f0, k0), (v1, f1, k1) in zip(reference, result):
        assert k0 == k1
        np.testing.assert_array_equal(f0, f1)
        # Same tolerance as production NumPy/Torch field differences on Model 7.
        np.testing.assert_allclose(v0, v1, rtol=0, atol=1e-4)


@pytest.mark.parametrize('kind', ['torch_cpu', 'torch_gpu', 'keops_gpu'])
@pytest.mark.parametrize('mode', ['joint', 'joint_contacts'])
def test_fault_free_production_plane_on_torch_backends(kind, mode):
    """The ordinary (fault-free) joint adapter on every backend."""
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

    def run():
        stacks = StacksStructure(np.array([3]), np.array([1]), np.array([1]), [R.BASEMENT])
        descriptor = InputDataDescriptor(TensorsStructure(np.array([3])), stacks)
        grid = EngineGrid(octree_grid=RegularGrid([0., 1., 0., 1., 0., 1.], [4, 4, 4]))
        inputs = InterpolationInput(SurfacePoints(np.array([[.1, .1, .37], [.9, .1, .37], [.1, .9, .37]])),
                                    Orientations(np.array([[.5, .5, .37]]), np.array([[0., 0., 1.]])), grid)
        options = InterpolationOptions.from_args(range=2., c_o=1.)
        options.evaluation_options.number_octree_levels = 2
        options.evaluation_options.number_octree_levels_surface = 2
        options.evaluation_options.mesh_extraction_overlap = mode
        return _host_meshes(compute_model(inputs, options, descriptor).dc_meshes)

    reference = run()
    torch = _torch_backend(kind)
    try:
        result = run()
    finally:
        _restore_numpy(torch)
    for (v0, f0, k0), (v1, f1, k1) in zip(reference, result):
        assert k0 == k1
        np.testing.assert_array_equal(f0, f1)
        # Production NumPy and Torch fields differ by ~1e-4 here (pretty included).
        np.testing.assert_allclose(v0, v1, rtol=0, atol=2e-4)

# endregion
