"""Fault, micro-point, external-stack and extent-snapping cases found on the server.

Runs the server's mesh path (``joint_contacts``, INTERSECT masking) on NumPy and
on Torch float32 with KeOps and flat stacks. ``known_bug`` tests state the
intended behaviour and fail strictly once they pass, so a fix has to remove the
marker; ``--runxfail`` shows each reproduction. The other tests pin the
supported limits and the references the meshes are measured against.

Models live in the unit cube, with faults off the octree lattice.
"""

import importlib
from typing import NamedTuple, Optional

import numpy as np
import pytest

from gempy_engine.API.model.field_evaluation import prepare_field_evaluator
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data import InterpolationOptions, Orientations, SurfacePoints, TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid, RegularGrid
from gempy_engine.core.data.finite_fault import FiniteFault
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.faults import FaultsData
from gempy_engine.core.data.micro_points import MicroPoints
from gempy_engine.core.data.options.evaluation_options import MeshExtractionMaskingOptions
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.core.exceptions import GemPyEngineInputError
from gempy_engine.modules.dual_contouring._gen_vertices import snap_tolerance

FAULT_X = .47
SECOND_FAULT = (.33, .5)  # y left and right of FAULT_X
STRATA = (.43, .6)  # z left and right of FAULT_X
UNCONFORMITY_Z = .85
CONTACTS = np.array([[.25, .6, STRATA[0] + .06], [.7, .6, STRATA[1] + .06]])  # One per side of the fault
SAMPLER_TOLERANCE = dict(rtol=1e-6, atol=1e-5)  # rtol covers float32 on fields of hundreds, as on the graben

ONE_FAULT = [[0, 1], [0, 0]]
FAULT_ON_FAULT = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
INDEPENDENT_FAULTS = [[0, 0, 1], [0, 0, 1], [0, 0, 0]]
NON_TRANSITIVE_FAULTS = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
ALL_OVERLAPS = ['none', 'pretty', 'joint_contacts']


def known_bug(raises, reason):
    return pytest.mark.xfail(strict=True, raises=raises, reason=reason)


@pytest.fixture(params=['numpy', 'server'])
def backend(request, monkeypatch):
    server = request.param == 'server'
    if server:
        pytest.importorskip('torch')
        pytest.importorskip('pykeops')
    monkeypatch.setenv('GEMPY_FLAT_STACKS', str(server))
    BT._change_backend(AvailableBackends.PYTORCH if server else AvailableBackends.numpy, use_gpu=False,
                       use_pykeops=server, dtype='float32' if server else 'float64', grads=False)
    return request.param


@pytest.fixture
def no_interpolation(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Unsupported input must be rejected before interpolation')

    monkeypatch.setattr('gempy_engine.API.model.model_api.interpolate_n_octree_levels', unexpected)


# region Models

class Stack(NamedTuple):
    points: np.ndarray
    relation: R
    orientation: Optional[tuple] = None  # (position, gradient)
    external: Optional[CustomInterpolationFunctions] = None
    surfaces: int = 1


def _patch(axis, level):
    """3 x 3 points on the plane ``axis = level``."""
    u, v = np.meshgrid(np.linspace(.15, .85, 3), np.linspace(.15, .85, 3))
    points = np.zeros((9, 3))
    points[:, axis] = level
    points[:, [a for a in range(3) if a != axis]] = np.column_stack((u.ravel(), v.ravel()))
    return points


def _stepped_patch(axis, left, right):
    points = _patch(axis, left)
    points[points[:, 0] > FAULT_X, axis] = right
    return points


def _fault():
    return Stack(_patch(0, FAULT_X), R.FAULT, ([FAULT_X, .5, .5], [1., 0., 0.]))


def _second_fault():
    """Offset across the first fault, and flat on each side of it once that fault applies."""
    return Stack(_stepped_patch(1, *SECOND_FAULT), R.FAULT, ([.25, SECOND_FAULT[0], .5], [0., 1., 0.]))


def _strata():
    return Stack(_stepped_patch(2, *STRATA), R.BASEMENT, ([.25, .5, STRATA[0]], [0., 0., 1.]))


def _plane_field(axis, isovalue):
    """External field equal to the ``axis`` coordinate, with its gradients."""

    def gradient(component):
        return lambda xyz: xyz[:, 0] * 0 + float(component == axis)

    return CustomInterpolationFunctions(np.array([isovalue]), lambda xyz: xyz[:, axis],
                                        gradient(0), gradient(1), gradient(2))


def _zero_point_stack(relation, field):
    """As gempy's ``from_structural_frame``: no points, so the stack declares no surfaces."""
    return Stack(np.empty((0, 3)), relation, external=field, surfaces=0)


def _external_fault(with_points):
    field = _plane_field(0, FAULT_X)
    return _fault()._replace(external=field) if with_points else _zero_point_stack(R.FAULT, field)


def _finite_fault():
    return FiniteFault(center=(FAULT_X, .5, .5), strike_radius=.6, dip_radius=.6)


def _contacts(surface):
    return MicroPoints(CONTACTS.copy(), np.repeat(np.eye(3)[None] * 3, 2, axis=0), np.zeros(2), np.full(2, surface))


def _model(stacks, relations, micro=None, finite_fault=None, overlap='joint_contacts', micro_enabled=True):
    orientations = [stack.orientation for stack in stacks if stack.orientation is not None]
    inputs = InterpolationInput(
        SurfacePoints(np.vstack([stack.points for stack in stacks])),
        Orientations(np.array([o[0] for o in orientations]), np.array([o[1] for o in orientations])),
        EngineGrid(octree_grid=RegularGrid([0., 1., 0., 1., 0., 1.], [4, 4, 4])),
        micro_points=micro)
    faults = [None] * len(stacks)
    if finite_fault is not None:
        faults[0] = FaultsData.from_user_input(None, finite_fault)
    externals = [stack.external for stack in stacks]
    descriptor = InputDataDescriptor(
        TensorsStructure(np.concatenate([np.full(s.surfaces, len(s.points) // s.surfaces) for s in stacks if s.surfaces])),
        StacksStructure(np.array([len(stack.points) for stack in stacks]),
                        np.array([int(stack.orientation is not None) for stack in stacks]),
                        np.array([stack.surfaces for stack in stacks]), [stack.relation for stack in stacks],
                        faults_relations=np.array(relations, dtype=bool), faults_input_data=faults,
                        interp_functions_per_stack=externals if any(f is not None for f in externals) else None))
    options = InterpolationOptions.from_args(range=3., c_o=1., uni_degree=1)
    evaluation = options.evaluation_options
    evaluation.number_octree_levels = evaluation.number_octree_levels_surface = 3
    evaluation.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    evaluation.mesh_extraction_overlap = overlap
    options.micro_options.enabled = micro_enabled
    return inputs, options, descriptor


# endregion


# region Measurements

def _host(values):
    return values.detach().cpu().numpy() if hasattr(values, 'detach') else np.asarray(values)


def _mesh(meshes, stack):
    return next(mesh for mesh in meshes if mesh.stack_index == stack)


def _assert_stepped(mesh, axis, levels):
    """The mesh sits on ``levels`` left and right of the fault."""
    vertices = _host(mesh.vertices)
    sides = vertices[vertices[:, 0] < FAULT_X - .12, axis], vertices[vertices[:, 0] > FAULT_X + .12, axis]
    for side, level in zip(sides, levels):
        assert np.abs(side - level).max() < .01


def _contact_misfit(mesh):
    """Vertical distance from each contact to the mesh vertex closest to it in plan."""
    vertices = _host(mesh.vertices)
    nearest = np.argmin(np.linalg.norm(vertices[:, None, :2] - CONTACTS[:, :2], axis=-1), axis=0)
    return np.abs(vertices[nearest, 2] - CONTACTS[:, 2])


def _assert_borrows_fault_vertices(meshes, fault_stack, stack):
    assert set(_mesh(meshes, fault_stack).joint_vertex_keys) & set(_mesh(meshes, stack).joint_vertex_keys)


def _stack_field(solutions, stack):
    return _host(solutions.octrees_output[-1].outputs[stack].exported_fields.scalar_field)


# endregion


# region Supported limits, rejected before interpolating

def test_finite_fault_with_several_surfaces_is_rejected(backend, no_interpolation):
    fault = Stack(np.vstack((_patch(0, FAULT_X), _patch(0, FAULT_X + .1))), R.FAULT, _fault().orientation, surfaces=2)
    with pytest.raises(GemPyEngineInputError, match='exactly one surface'):
        compute_model(*_model([fault, _strata()], ONE_FAULT, finite_fault=_finite_fault()))


def test_micro_points_on_an_external_stack_are_rejected(backend, no_interpolation):
    external = _strata()._replace(external=_plane_field(2, .5))
    with pytest.raises(NotImplementedError, match='external-function stacks'):
        compute_model(*_model([_fault(), external], ONE_FAULT, micro=_contacts(1)))


def test_micro_points_on_a_fault_surface_are_rejected(backend, no_interpolation):
    with pytest.raises(ValueError, match='micro points on fault surfaces'):
        compute_model(*_model([_fault(), _strata()], ONE_FAULT, micro=_contacts(0)))


@pytest.mark.parametrize('with_points', [True, False], ids=['with_points', 'zero_points'])
def test_external_fault_is_rejected(backend, no_interpolation, with_points):
    with pytest.raises(GemPyEngineInputError, match='cannot use an external interpolation function'):
        compute_model(*_model([_external_fault(with_points), _strata()], ONE_FAULT))


# endregion


def test_interpolated_fault_reference(backend):
    """What the known bugs below are measured against: the supported single-fault path."""
    meshes = compute_model(*_model([_fault(), _strata()], ONE_FAULT)).dc_meshes
    _assert_stepped(_mesh(meshes, 1), 2, STRATA)
    _assert_borrows_fault_vertices(meshes, 0, 1)


# region Fault targeting an external stack

def _fault_over_external(relation):
    external = _zero_point_stack(R.ERODE, _plane_field(2, UNCONFORMITY_Z))
    return compute_model(*_model([_fault(), external, _strata()], [[0, relation, 1], [0, 0, 0], [0, 0, 0]]))


@known_bug(AssertionError, 'external stacks never receive fault drift, and nothing rejects the relation')
def test_fault_targeting_an_external_stack_is_not_silently_ignored(backend):
    """Reject the relation or offset the surface; both meet this test."""
    try:
        related = _fault_over_external(1)
    except GemPyEngineInputError:
        return
    assert not np.array_equal(_stack_field(related, 1), _stack_field(_fault_over_external(0), 1))


# endregion


# region Micro points in a faulted model

@pytest.mark.parametrize('finite', [False, True], ids=['infinite', 'finite'])
@pytest.mark.parametrize('overlap', ['pretty', 'joint_contacts'])
def test_micro_contacts_on_a_stack_under_a_fault_reach_the_mesh(backend, overlap, finite):
    meshes = compute_model(*_model([_fault(), _strata()], ONE_FAULT, micro=_contacts(1), overlap=overlap,
                                   finite_fault=_finite_fault() if finite else None)).dc_meshes
    assert (_contact_misfit(_mesh(meshes, 1)) < .015).all()
    if overlap == 'joint_contacts':
        _assert_borrows_fault_vertices(meshes, 0, 1)


def test_uncorrected_mesh_misses_the_contacts(backend):
    """Keeps the misfit bound above meaningful."""
    meshes = compute_model(*_model([_fault(), _strata()], ONE_FAULT, overlap='pretty')).dc_meshes
    assert (_contact_misfit(_mesh(meshes, 1)) > .05).all()


def test_micro_contacts_beside_a_fault_that_does_not_offset_their_stack(backend):
    meshes = compute_model(*_model([_fault(), _strata()], [[0, 0], [0, 0]], micro=_contacts(1))).dc_meshes
    assert (_contact_misfit(_mesh(meshes, 1)) < .015).all()


@pytest.mark.parametrize('state', ['disabled', 'empty'])
def test_inactive_micro_points_do_not_block_joint_contacts(backend, state):
    """Inactive points are ignored everywhere else, so the meshes match a model without them."""
    micro = _contacts(1) if state == 'disabled' else MicroPoints(
        np.empty((0, 3)), np.empty((0, 3, 3)), np.empty(0), np.empty(0, dtype=np.int64))
    stacks = [_fault(), _strata()]
    reference = compute_model(*_model(stacks, ONE_FAULT)).dc_meshes
    meshes = compute_model(*_model(stacks, ONE_FAULT, micro=micro, micro_enabled=state == 'empty')).dc_meshes
    for expected, mesh in zip(reference, meshes):
        np.testing.assert_array_equal(_host(mesh.vertices), _host(expected.vertices))


# endregion


# region Fault offsetting a fault

@pytest.mark.parametrize('overlap', ['pretty', 'joint_contacts'])
def test_fault_offsets_a_younger_fault(backend, overlap):
    meshes = compute_model(*_model([_fault(), _second_fault(), _strata()], FAULT_ON_FAULT, overlap=overlap)).dc_meshes
    _assert_stepped(_mesh(meshes, 1), 1, SECOND_FAULT)
    _assert_stepped(_mesh(meshes, 2), 2, STRATA)
    if overlap == 'joint_contacts':
        _assert_borrows_fault_vertices(meshes, 0, 2)
        _assert_borrows_fault_vertices(meshes, 1, 2)


def test_second_fault_is_not_stepped_without_the_relation(backend):
    """Keeps the step check above meaningful."""
    stacks = [_fault(), _second_fault(), _strata()]
    meshes = compute_model(*_model(stacks, INDEPENDENT_FAULTS, overlap='pretty')).dc_meshes
    with pytest.raises(AssertionError):
        _assert_stepped(_mesh(meshes, 1), 1, SECOND_FAULT)


def test_micro_contacts_under_a_fault_on_fault_chain(backend):
    meshes = compute_model(*_model([_fault(), _second_fault(), _strata()], FAULT_ON_FAULT, micro=_contacts(2))).dc_meshes
    assert (_contact_misfit(_mesh(meshes, 2)) < .015).all()


def test_surface_keeps_its_vertex_where_its_fault_borrows_from_another(backend):
    """A -> B -> strata without A -> strata: where all three cross, B has no vertex to lend."""
    stacks = [_fault(), _second_fault(), _strata()]
    meshes = compute_model(*_model(stacks, NON_TRANSITIVE_FAULTS)).dc_meshes
    assert all(len(mesh.edges) for mesh in meshes)
    assert meshes[0].contact_report['multi_fault_unborrowed_count'] > 0


def test_offset_fault_reaches_the_fault_that_cuts_it(backend):
    meshes = compute_model(*_model([_fault(), _second_fault(), _strata()], FAULT_ON_FAULT)).dc_meshes
    _assert_borrows_fault_vertices(meshes, 0, 1)


# endregion


# region Snapping to the extent

class _DeviceScalar:
    """A 0-D CUDA tensor as NumPy sees it: ``float`` reads it, implicit conversion refuses."""

    def __init__(self, value):
        self.value = value

    def __float__(self):
        return self.value

    def __array__(self, *args, **kwargs):
        raise TypeError("can't convert cuda:0 device type tensor to numpy")


def _plane(level, tilt=(0., 0.)):
    """Horizon at ``level`` in the middle of the model, rising ``tilt`` along x and y."""
    points = _patch(2, level)
    points[:, 2] += tilt[0] * (points[:, 0] - .5) + tilt[1] * (points[:, 1] - .5)
    return Stack(points, R.BASEMENT, ([.5, .5, level], [-tilt[0], -tilt[1], 1.]))


def _snapped(model, snap):
    model[1].evaluation_options.mesh_extraction_snap_to_extent = snap
    return model


def _assert_no_folds(mesh):
    """No triangle lies flat in an extent face."""
    corners = _host(mesh.vertices)[_host(mesh.edges)]
    assert not ((np.abs(corners) < 1e-6) | (np.abs(corners - 1) < 1e-6)).all(axis=1).any()


def test_snap_tolerance_reads_a_grid_spacing_held_on_a_device():
    assert snap_tolerance(tuple(_DeviceScalar(d) for d in (.5, .25, 1.))) == pytest.approx(2.5e-4)


@pytest.mark.parametrize('level', [1e-5, 1 - 1e-5], ids=['bottom', 'top'])
@pytest.mark.parametrize('overlap', ALL_OVERLAPS)
def test_surface_just_inside_a_face_is_not_snapped_away(backend, overlap, level):
    """Closer to the face than the snap tolerance, without crossing it."""
    meshes = [compute_model(*_snapped(_model([_plane(level)], [[0]], overlap=overlap), snap)).dc_meshes[0]
              for snap in (False, True)]
    assert len(meshes[1].edges) == len(meshes[0].edges) > 0
    assert np.abs(_host(meshes[1].vertices)[:, 2] - level).max() < 1e-3


@pytest.mark.parametrize('overlap', ALL_OVERLAPS)
def test_surface_crossing_a_face_ends_on_it_without_folds(backend, overlap):
    mesh = compute_model(*_model([_plane(.1, (.4, .4))], [[0]], overlap=overlap)).dc_meshes[0]
    assert (np.abs(_host(mesh.vertices)[:, 2]) < 1e-6).sum() > 3
    _assert_no_folds(mesh)


def test_contact_leaving_through_a_face_leaves_no_fold(backend):
    """A junction snapped to the face folds like any other snapped vertex."""
    top = _plane(.1, (.5, .25))._replace(relation=R.ERODE)
    meshes = compute_model(*_model([top, _plane(.14, (.1, -.5))], [[0, 0], [0, 0]])).dc_meshes
    assert set(meshes[0].joint_vertex_keys) & set(meshes[1].joint_vertex_keys)
    for mesh in meshes:
        _assert_no_folds(mesh)


@pytest.mark.parametrize('tilt', [0., .3], ids=['flat', 'dipping'])
@pytest.mark.parametrize('overlap', ['pretty', 'joint_contacts'])
def test_fault_vertices_on_the_extent_keep_the_surfaces_that_borrow_them(backend, overlap, tilt):
    """The fault and the horizon meet on the extent face at the horizon's own level."""
    horizon = _plane(STRATA[0], (0., tilt))
    meshes = compute_model(*_model([_fault(), horizon], ONE_FAULT, overlap=overlap)).dc_meshes
    vertices = _host(_mesh(meshes, 1).vertices)
    assert np.abs(vertices[:, 2] - STRATA[0] - tilt * (vertices[:, 1] - .5)).max() < .002
    assert (vertices[:, 1] < 1e-6).any() and (vertices[:, 1] > 1 - 1e-6).any()


# endregion


# region Drift sampler against production

def _independent_faults():
    return _model([_fault(), _second_fault(), _strata()], INDEPENDENT_FAULTS)


def _fault_on_fault():
    return _model([_fault(), _second_fault(), _strata()], FAULT_ON_FAULT)


def _finite_fault_on_fault():
    return _model([_fault(), _second_fault(), _strata()], FAULT_ON_FAULT, finite_fault=_finite_fault())


def _graben():
    """The fixture's own relations; test_joint_contacts._graben overwrites them to independent faults."""
    from tests.fixtures.complex_geometries import graben_fault_model

    inputs, descriptor, options = getattr(graben_fault_model, '__wrapped__', graben_fault_model)()
    assert descriptor.stack_structure.faults_relations[0, 1]
    evaluation = options.evaluation_options
    evaluation.mesh_extraction = True
    evaluation.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    evaluation.number_octree_levels = evaluation.number_octree_levels_surface = 4
    evaluation.mesh_extraction_overlap = 'joint_contacts'
    return inputs, options, descriptor


def _non_transitive_faults():
    return _model([_fault(), _second_fault(), _strata()], NON_TRANSITIVE_FAULTS)


def _micro_under_fault():
    return _model([_fault(), _strata()], ONE_FAULT, micro=_contacts(1))


def _micro_under_finite_fault():
    return _model([_fault(), _strata()], ONE_FAULT, micro=_contacts(1), finite_fault=_finite_fault())


def _micro_under_fault_on_fault():
    return _model([_fault(), _second_fault(), _strata()], FAULT_ON_FAULT, micro=_contacts(2))


def _micro_beside_a_stack_with_the_same_drift():
    """Two stacks with one drift row share KeOps' stacked reduction, so the correction goes through it."""
    upper = Stack(_stepped_patch(2, .7, .85), R.ERODE, ([.25, .5, .7], [0., 0., 1.]))
    return _model([_fault(), upper, _strata()], [[0, 1, 1], [0, 0, 0], [0, 0, 0]], micro=_contacts(2))


def _record_samplers(monkeypatch):
    bridge = importlib.import_module('gempy_engine.API.dual_contouring.joint_extraction')
    prepare, prepared = bridge.prepare_fault_drift_sampler, []
    monkeypatch.setattr(bridge, 'prepare_fault_drift_sampler', lambda *args: prepared.append(prepare(*args)) or prepared[-1])
    return prepared


def _away_from_faults(sampler, outputs, points):
    """Points the drift step cannot flip: on a fault, rounding picks either side, as the sampler's own check allows."""
    faults = sampler['fault_stacks']
    scalars, _ = sampler['query'](points, stacks=faults, gradients=False)
    keep = np.ones(len(points), dtype=bool)
    for scalar, fault in zip(scalars, faults):
        rounding = 64 * np.finfo(BT.dtype).eps * max(np.abs(scalar).max(), 1.)
        keep &= np.abs(scalar - _host(outputs[fault].scalar_field_at_sp).reshape(-1)[0]) > rounding
    return keep


@pytest.mark.parametrize('model', [_independent_faults, _fault_on_fault, _finite_fault_on_fault, _non_transitive_faults,
                                   _graben, _micro_under_fault, _micro_under_finite_fault, _micro_under_fault_on_fault,
                                   _micro_beside_a_stack_with_the_same_drift])
def test_fault_drift_sampler_matches_production(backend, monkeypatch, model):
    """joint_contacts evaluates new points through the sampler, so it has to agree with production.

    Corners are shuffled: planar faults mesh correctly even with drift read by row
    instead of by location, because their gradients do not depend on it.
    """
    prepared = _record_samplers(monkeypatch)
    outputs = compute_model(*model()).octrees_output[-1].outputs
    sampler = prepared[0]
    for stack, output in enumerate(outputs):
        corners = _host(output.grid.corners_grid.values)
        rows = np.random.default_rng(stack).permutation(len(corners))[:2000]
        expected = _host(output.exported_fields.scalar_field[output.grid.corners_grid_slice])[rows]
        keep = _away_from_faults(sampler, outputs, corners[rows])
        values, _ = sampler['query'](corners[rows], stacks=(stack,), gradients=False)
        np.testing.assert_allclose(values[0][keep], expected[keep], **SAMPLER_TOLERANCE)
        # What joint_contacts calls: under KeOps this is the separate, stacked path.
        batched, = sampler['query_batch']([(corners[rows], (stack,), 'scalar')])
        np.testing.assert_allclose(batched[0][keep], expected[keep], **SAMPLER_TOLERANCE)
    points, stacks = corners[rows[:500]], tuple(range(len(outputs)))
    values, gradients = sampler['query'](points, stacks=stacks)
    batched_values, batched_gradients = sampler['query_batch']([(points, stacks, 'scalar'), (points, stacks, 'gradient')])
    np.testing.assert_allclose(batched_values, values, **SAMPLER_TOLERANCE)
    np.testing.assert_allclose(batched_gradients, gradients, rtol=1e-3, atol=1e-3)


# endregion


# region Field evaluation keeps its contract

def test_field_evaluator_rejects_a_fault_offsetting_a_fault(backend):
    inputs, options, descriptor = _fault_on_fault()
    solutions = compute_model(inputs, options, descriptor)
    with pytest.raises(ValueError, match='unsupported_fault_relations'):
        prepare_field_evaluator(solutions, inputs, descriptor, options)


@pytest.mark.parametrize('faulted', [True, False], ids=['faulted', 'no_fault'])
def test_field_evaluator_rejects_micro_points(backend, faulted):
    """The caller's input never has its evaluation points prepared, only compute_model's copy does."""
    model = _micro_under_fault() if faulted else _model([_strata()], [[0]], micro=_contacts(0))
    inputs, options, descriptor = model
    solutions = compute_model(inputs, options, descriptor)
    with pytest.raises(NotImplementedError, match='unsupported_fault_micro_correction'):
        prepare_field_evaluator(solutions, inputs, descriptor, options)


def test_fault_sides_reject_a_fault_offsetting_a_fault(backend, monkeypatch):
    prepared = _record_samplers(monkeypatch)
    compute_model(*_fault_on_fault())
    points = np.array([[.5, .5, .5]])
    with pytest.raises(ValueError, match='fault sides require independent faults'):
        prepared[0]['query_sides'](points, (2,), np.ones((2, 1), dtype=int))

# endregion
