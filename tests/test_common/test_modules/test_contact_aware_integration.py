"""Voxel contact acceptance tests, independent of the superseded planar oracle."""

import importlib
from itertools import permutations, product
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gempy_engine import config
from gempy_engine.API.model.model_api import compute_model
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.dual_contouring_data import DualContouringData
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.options.evaluation_options import MeshExtractionMaskingOptions, MeshExtentCapping
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from tests.fixtures.contact_cases import build_contact_case
from gempy_engine.modules.dual_contouring.dual_contouring_interface import find_intersection_on_edge


dc = importlib.import_module("gempy_engine.API.dual_contouring.multi_scalar_dual_contouring")
contact_api = importlib.import_module("gempy_engine.API.dual_contouring.contact_reconciliation")
ROOT = Path(__file__).resolve().parents[3]


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _descriptor(counts, relations, faults=None):
    n = len(counts)
    return InputDataDescriptor(
        TensorsStructure(np.array([], dtype=int)),
        StacksStructure(np.zeros(n, dtype=int), np.zeros(n, dtype=int),
                        np.array(counts), list(relations),
                        faults_relations=np.zeros((n, n), dtype=bool) if faults is None else faults),
    )


def _assert_report(mesh):
    report = mesh.contact_report
    assert report["cell_vertex_correspondence"] is True
    assert report["differentiable_topology"] is False
    ids = np.asarray(report["contact_ids"])
    assert ids.shape == (len(mesh.vertices),)
    assert np.issubdtype(ids.dtype, np.integer) and np.all(ids >= -1)
    assert report["contact_count"] >= len(np.unique(ids[ids >= 0]))
    assert report["status"] == ("reconciled" if np.any(ids >= 0) else "no_contact")
    assert isinstance(report["topology"], dict)
    for counts in (report, report["topology"]):
        for key, value in counts.items():
            if key.endswith("_count"):
                assert isinstance(value, (int, np.integer)) and value >= 0
    assert isinstance(mesh.vertices, np.ndarray) and isinstance(mesh.edges, np.ndarray)
    assert np.isfinite(mesh.vertices).all()
    assert mesh.edges.ndim == 2 and mesh.edges.shape[1] == 3
    assert np.all((mesh.edges >= 0) & (mesh.edges < len(mesh.vertices)))
    assert len(np.unique(np.sort(mesh.edges, axis=1), axis=0)) == len(mesh.edges)
    return ids


def test_nonexported_null_space_removes_hidden_raw_surface(extraction):
    _, descriptor, inputs, options, levels = extraction()
    descriptor.stack_structure.masking_descriptor[0] = StackRelationType.NULL_SPACE
    levels[0].outputs[0].scalar_fields.stack_relation = StackRelationType.NULL_SPACE
    levels[0].outputs[1].squeezed_mask_array[:] = False
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    assert len(meshes) == 1 and meshes[0].stack_index == 1
    assert len(meshes[0].vertices) > 0 and len(meshes[0].edges) == 0
    counts = meshes[0].contact_report['topology']['per_surface'][0]
    assert counts['ownership_removed_count'] == counts['input_count'] > 0


def test_fault_qef_excludes_fault_to_fault_constraints(monkeypatch):
    faults = np.array([[False, True, False], [False, False, True], [False, False, False]])
    descriptor = _descriptor([1, 1, 1], [StackRelationType.FAULT, StackRelationType.FAULT,
                                       StackRelationType.BASEMENT], faults)
    inject = Mock()
    monkeypatch.setattr(contact_api, 'find_and_inject_multi_surface_constraints_multicore', inject)
    _, fault_pairs, _ = contact_api.prepare_contact_constraints(
        [object()] * 3, [np.zeros((1, 3), dtype=int)] * 3,
        [(0, 0, .5), (1, 0, .5), (2, 0, .5)], descriptor, (1, 1, 1),
    )
    inject.assert_called_once()
    assert inject.call_args.kwargs['allowed_partners_per_surface'] == [set(), {2}, {1}]
    assert fault_pairs == {(0, 1), (1, 2)}


def test_onlap_discarded_contact_row_restores_substrate(extraction):
    # Local window of the 64-cell smoke geometry, with full interior quad support.
    cells = np.array(list(product(range(31, 36), range(8, 12), range(27, 34))))
    offsets = np.array(list(product((0, 1), repeat=3)))
    corners = ((cells[:, None, :] + offsets) / 64).reshape(-1, 3)
    normals = [np.array([1., 0., -.25]), np.array([.25, 0., 1.])]
    substrate = (corners @ normals[1]).reshape(-1, 8)
    owned = [substrate > .6, substrate <= .6]
    data, coordinates, ownership = [], [], []
    for normal, level, mask in zip(normals, [.4, .6], owned):
        selected = mask.any(axis=1)
        xyz, edges = find_intersection_on_edge(corners, corners @ normal,
                                               np.array([level]), masking=selected)
        item = DualContouringData(
            xyz_on_edge=xyz, valid_edges=edges, xyz_on_centers=(cells[selected] + .5) / 64,
            dxdydz=(1 / 64,) * 3, n_surfaces_to_export=0, left_right_codes=cells[selected],
            gradients=np.tile(normal, (len(xyz), 1)), tree_depth=3, base_number=(64, 64, 64),
        )
        data.append(item)
        coordinates.append(item.left_right_codes[item.valid_voxels])
        ownership.append(mask[selected][item.valid_voxels])
    meshes = dc.compute_dual_contouring_v2(data, max_workers=1)
    original = [mesh.vertices.copy() for mesh in meshes]
    descriptor = _descriptor([1, 1], [StackRelationType.ONLAP, StackRelationType.BASEMENT])
    contact_api.reconcile_contact_meshes(meshes, coordinates, [(0, 0, .4), (1, 0, .6)],
                                         descriptor, ownership)
    for z, supported in [(29, False), (30, True)]:
        rows = [int(np.flatnonzero(np.all(points == [33, 9, z], axis=1))[0]) for points in coordinates]
        assert bool(np.any(meshes[0].edges == rows[0])) == supported
        if supported:
            expected = (original[0][rows[0]] + original[1][rows[1]]) / 2
            for mesh, row in zip(meshes, rows):
                np.testing.assert_array_equal(mesh.vertices[row], expected)
                assert mesh.contact_report['contact_ids'][row] >= 0
        else:
            for index, (mesh, row) in enumerate(zip(meshes, rows)):
                np.testing.assert_array_equal(mesh.vertices[row], original[index][row])
                assert mesh.contact_report['contact_ids'][row] == -1
    assert meshes[0].contact_report['dissolved_contact_count'] > 0
    for mesh in meshes:
        shared = np.flatnonzero(mesh.contact_report['contact_ids'] >= 0)
        assert np.all(np.isin(shared, mesh.edges))


@pytest.fixture
def extraction(monkeypatch):
    saved = dict(engine_backend=BackendTensor.engine_backend, use_gpu=BackendTensor.use_gpu,
                 use_pykeops=BackendTensor.use_pykeops, dtype=BackendTensor.dtype,
                 grads=BackendTensor.COMPUTE_GRADS)
    saved_keops = BackendTensor.pykeops_enabled
    monkeypatch.setenv("GEMPY_SKIP_TRIANGULATION", "0")
    monkeypatch.setenv("DUAL_CONTOURING_MULTITHREAD", "False")
    BackendTensor._change_backend(config.AvailableBackends.numpy, use_gpu=False,
                                  use_pykeops=False, dtype="float64", grads=False)
    BackendTensor.pykeops_enabled = False
    monkeypatch.setattr(dc, "DUAL_CONTOURING_VERTEX_OVERLAP", config.DualContouringOverlap.contact_aware)

    def make(name="planar_erosion", dtype="float64", backend=config.AvailableBackends.numpy):
        catalogue_name = ("three_way_junction" if name in ("fault_mixed", "three_stack_erosion", "three_stack_onlap") else "parallel_false_overlap"
                          if name == "fault_parallel" else name
                          if name in ("onlap", "parallel_false_overlap", "three_way_junction") else "planar_erosion")
        case = build_contact_case(catalogue_name, resolution=6)
        if name == "single":
            case.normals, case.levels = case.normals[:1], case.levels[:1]
            case.relations = (StackRelationType.BASEMENT,)
        elif name == "no_overlap":
            case.normals[:] = (1, 0, 0)
            case.levels[:] = (.20, .80)
        elif name == "same_group":
            case.normals[:] = (1, 0, 0)
            case.levels[:] = (.40, .46)
            case.relations = (StackRelationType.BASEMENT,)
        elif name in ("fault", "fault_parallel"):
            case.relations = (StackRelationType.FAULT, StackRelationType.BASEMENT)
        elif name == "fault_mixed":
            case.relations = (StackRelationType.FAULT, StackRelationType.ERODE, StackRelationType.BASEMENT)
        elif name == "three_stack_onlap":
            case.relations = (StackRelationType.ONLAP, StackRelationType.ONLAP, StackRelationType.BASEMENT)
        elif name == "same_group_competition":
            case.normals[:] = [(1, 0, 0), (0, 0, 1)]
            case.levels[:] = [.40, .47]
        BackendTensor._change_backend(backend, use_gpu=False, use_pykeops=False, dtype=dtype, grads=False)
        root = RegularGrid(np.array([0., 1., 0., 1., 0., 1.]), list(case.resolution))
        offsets = np.array(list(product((0, 1), repeat=3)))
        corners = ((_numpy(root.integer_coordinates)[:, None, :] + offsets) / case.resolution).reshape(-1, 3)
        grid = EngineGrid(octree_grid=root, corners_grid=GenericGrid(corners))
        inputs = InterpolationInput(SurfacePoints(np.empty((0, 3))),
                                    Orientations(np.empty((0, 3)), np.empty((0, 3))), grid)
        normals = case.normals[:1] if name == "same_group" else case.normals
        isovalues = [case.levels] if name == "same_group" else [[level] for level in case.levels]
        if name == "same_group_competition":
            isovalues[0] = [.40, .46]
        n = len(normals)
        faults = np.zeros((n, n), dtype=bool)
        if name in ("fault", "fault_parallel", "fault_mixed"):
            faults[0, 1] = True
        descriptor = _descriptor([len(v) for v in isovalues], case.relations, faults)
        options = InterpolationOptions.from_args(10., 1.)
        options.evaluation_options.number_octree_levels = 1
        options.evaluation_options.number_octree_levels_surface = 1
        options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
        xyz = _numpy(grid.values)
        ownership = np.ones((n, len(xyz)), dtype=bool)
        if name == "three_stack_erosion":
            above = xyz @ normals.T >= case.levels
            ownership[0] = above[:, 0]
            ownership[1] = ~above[:, 0] & above[:, 1]
            ownership[2] = ~above[:, 0] & ~above[:, 1]
        elif name == "three_stack_onlap":
            above = xyz @ normals.T >= case.levels
            ownership[0] = above[:, 1] & above[:, 2]
            ownership[1] = ~ownership[0] & above[:, 2]
            ownership[2] = ~above[:, 2]
        elif name == "fault_mixed":
            above = xyz @ normals[1] >= case.levels[1]
            ownership[1], ownership[2] = above, ~above
        elif name == "same_group_competition":
            ownership[0] = xyz @ normals[0] >= case.levels[0]
        if n == 2 and name not in ("fault", "fault_parallel", "no_overlap"):
            controller, truncated, sign = ((1, 0, 1) if case.relations[0] is StackRelationType.ONLAP
                                            else (0, 1, -1))
            controller_field = xyz @ normals[controller]
            if name == "curved" and controller == 0:
                controller_field = controller_field + .2 * xyz[:, 0] ** 2
            ownership[truncated] &= sign * (controller_field - case.levels[controller]) >= 0
        outputs = []
        for index, (normal, values, relation, mask) in enumerate(zip(normals, isovalues, case.relations, ownership)):
            field = xyz @ normal
            if name == "curved" and index == 0:
                field = field + .2 * xyz[:, 0] ** 2
            outputs.append(SimpleNamespace(
                grid=grid, scalar_field_at_sp=BackendTensor.t.array(values, dtype=BackendTensor.dtype_obj),
                exported_fields=SimpleNamespace(scalar_field=BackendTensor.t.array(field, dtype=BackendTensor.dtype_obj)),
                scalar_fields=SimpleNamespace(stack_relation=relation),
                squeezed_mask_array=BackendTensor.t.array(mask, dtype="bool"),
                mask_components=BackendTensor.t.array(mask.copy(), dtype="bool"),
            ))

        def gradients(interpolation_input, options, data_descriptor):
            assert options.evaluation_options.compute_scalar_gradient
            assert not options.evaluation_options.compute_scalar
            points = _numpy(interpolation_input.grid.custom_grid.values)
            data_descriptor.stack_structure.stack_number = n - 1
            result = []
            for index, normal in enumerate(normals):
                gradient = np.tile(normal, (len(points), 1))
                if name == "curved" and index == 0:
                    gradient[:, 0] += .4 * points[:, 0]
                result.append(SimpleNamespace(exported_fields=SimpleNamespace(**{
                    key: BackendTensor.t.array(gradient[:, axis], dtype=BackendTensor.dtype_obj)
                    for axis, key in enumerate(("gx_field", "gy_field", "gz_field"))
                })))
            return result

        monkeypatch.setattr(dc, "interpolate_all_fields_no_octree", gradients)
        return case, descriptor, inputs, options, [OctreeLevel(grid, outputs)]

    try:
        yield make
    finally:
        BackendTensor._change_backend(**saved)
        BackendTensor.COMPUTE_GRADS = saved["grads"]
        BackendTensor.pykeops_enabled = saved_keops


@pytest.mark.parametrize("name", ["planar_erosion", "onlap", "parallel_false_overlap", "curved"])
def test_cell_contacts_use_original_means_and_preserve_caller_state(extraction, monkeypatch, name):
    _, descriptor, inputs, options, levels = extraction(name)
    original_grid, original_weights = inputs.grid, inputs.weights
    original_values = _numpy(inputs.grid.values).copy()
    original_cursor = descriptor.stack_structure.stack_number
    original_options = options.model_dump_json()
    original_fields = [_numpy(o.exported_fields.scalar_field).copy() for o in levels[0].outputs]
    original_masks = [o.squeezed_mask_array.copy() for o in levels[0].outputs]
    snapshots = []
    extract = dc.compute_dual_contouring_v2

    def capture(**kwargs):
        meshes = extract(**dict(kwargs, max_workers=1))
        snapshots.extend((m.vertices, _numpy(m.vertices).copy(), _numpy(m.edges).copy()) for m in meshes)
        return meshes

    extraction_call = Mock(side_effect=capture)
    stage = Mock(wraps=dc.reconcile_contact_meshes)
    monkeypatch.setattr(dc, "compute_dual_contouring_v2", extraction_call)
    monkeypatch.setattr(dc, "reconcile_contact_meshes", stage)
    for helper in ("find_and_inject_multi_surface_constraints_multicore",
                   "average_overlapping_vertices", "remove_fault_overlap_triangles"):
        monkeypatch.setattr(dc, helper, Mock(side_effect=AssertionError("Nonfault contact entered legacy overlap")))
    monkeypatch.setattr(contact_api, "find_and_inject_multi_surface_constraints_multicore",
                        Mock(side_effect=AssertionError("Nonfault contact injected QEF partners")), raising=False)
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    extraction_call.assert_called_once()
    stage.assert_called_once()
    # Explicit extraction metadata, not a plane-fitting object, crosses the API boundary.
    arguments = stage.call_args
    if arguments.args:
        passed_meshes, cells, metadata, copied_descriptor, ownership = arguments.args
    else:
        passed_meshes, cells, metadata, copied_descriptor, ownership = (
            arguments.kwargs[key] for key in
            ("all_meshes", "cell_coordinates", "surface_metadata", "data_descriptor", "corner_ownership"))
    assert passed_meshes is meshes
    assert copied_descriptor is not descriptor
    assert metadata == [(i, 0, float(_numpy(o.scalar_field_at_sp)[0])) for i, o in enumerate(levels[0].outputs)]
    assert len(meshes) == len(cells) == len(ownership) == 2
    for mesh, coordinates, owned, (tensor, before_vertices, _) in zip(meshes, cells, ownership, snapshots):
        assert mesh.vertices_tensor is tensor
        np.testing.assert_array_equal(_numpy(mesh.vertices_tensor), before_vertices)
        assert not np.shares_memory(mesh.vertices, _numpy(mesh.vertices_tensor))
        assert len(mesh.vertices) == len(before_vertices) == len(coordinates)
        assert np.asarray(owned).shape == (len(mesh.vertices), 8)
        assert np.asarray(owned).dtype == np.bool_
        assert np.issubdtype(_numpy(coordinates).dtype, np.integer)
    ids = [_assert_report(mesh) for mesh in meshes]
    common_ids = np.intersect1d(ids[0][ids[0] >= 0], ids[1][ids[1] >= 0])
    if name == 'parallel_false_overlap':
        # Coarse sticking removes the entire redundant target patch. With no
        # surviving attachment, provisional members must not displace its owner.
        assert len(common_ids) == 0
        assert any(len(mesh.edges) == 0 for mesh in meshes)
        assert meshes[0].contact_report['provisional_contact_count'] > 0
        assert meshes[0].contact_report['dissolved_contact_count'] > 0
    else:
        assert len(common_ids) > 0
    for contact_id in common_ids:
        indices = [np.flatnonzero(v == contact_id) for v in ids]
        assert all(len(index) == 1 for index in indices)
        a, b = (int(index[0]) for index in indices)
        np.testing.assert_array_equal(cells[0][a], cells[1][b])
        expected = (snapshots[0][1][a] + snapshots[1][1][b]) / 2
        np.testing.assert_allclose(meshes[0].vertices[a], expected)
        np.testing.assert_array_equal(meshes[0].vertices[a], meshes[1].vertices[b])
    for mesh, vertex_ids, (_, original, _) in zip(meshes, ids, snapshots):
        np.testing.assert_array_equal(mesh.vertices[vertex_ids == -1], original[vertex_ids == -1])
        assert mesh.contact_report["contact_count"] == len(common_ids)
    assert inputs.grid is original_grid and inputs.weights is original_weights
    np.testing.assert_array_equal(_numpy(inputs.grid.values), original_values)
    assert descriptor.stack_structure.stack_number == original_cursor
    assert options.model_dump_json() == original_options
    for output, field, mask in zip(levels[0].outputs, original_fields, original_masks):
        np.testing.assert_array_equal(_numpy(output.exported_fields.scalar_field), field)
        np.testing.assert_array_equal(output.squeezed_mask_array, mask)


@pytest.mark.parametrize("name", ["single", "no_overlap", "same_group"])
def test_unshared_cells_preserve_extraction_and_connectivity(extraction, monkeypatch, name):
    _, descriptor, inputs, options, levels = extraction(name)
    snapshots = []
    extract = dc.compute_dual_contouring_v2

    def capture(**kwargs):
        meshes = extract(**dict(kwargs, max_workers=1))
        snapshots.extend((_numpy(m.vertices).copy(), _numpy(m.edges).copy()) for m in meshes)
        return meshes

    call = Mock(side_effect=capture)
    monkeypatch.setattr(dc, "compute_dual_contouring_v2", call)
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    call.assert_called_once()
    assert len(meshes) == (1 if name == "single" else 2)
    for mesh, (vertices, faces) in zip(meshes, snapshots):
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.edges, faces)
        assert np.all(_assert_report(mesh) == -1)
        assert mesh.contact_report["contact_count"] == 0
    if name == "same_group":
        assert [(m.stack_index, m.surface_index) for m in meshes] == [(0, 0), (0, 1)]
        cells = [m.dc_data.left_right_codes[m.dc_data.valid_voxels] for m in meshes]
        assert set(map(tuple, cells[0])) & set(map(tuple, cells[1]))
        assert not np.array_equal(meshes[0].vertices, meshes[1].vertices)


def _synthetic_reconcile(positions, cells, metadata, descriptor, order=None, faces=None, ownership=None):
    order = list(range(len(positions))) if order is None else list(order)
    meshes = []
    for i in order:
        original = np.asarray(positions[i], dtype=float).reshape(-1, 3).copy()
        # Repeated-index support faces isolate grouping from shared-patch removal;
        # production extraction tests above exercise real triangle geometry.
        triangles = (np.repeat(np.arange(len(original))[:, None], 3, axis=1) if faces is None
                     else np.asarray(faces[i], dtype=int).reshape(-1, 3))
        mesh = DualContouringMesh(original.copy(), triangles.copy())
        mesh.vertices_tensor = original
        meshes.append(mesh)
    coordinates = [np.asarray(cells[i], dtype=np.int64).reshape(-1, 3).copy() for i in order]
    # Mixed ownership permits a cell-local termination without using extraction candidates as ownership.
    owned = [np.tile([True, False] * 4, (len(coordinates[j]), 1)) if ownership is None
             else np.asarray(ownership[i], dtype=bool).copy() for j, i in enumerate(order)]
    snapshots = [(v.copy(), o.copy()) for v, o in zip(coordinates, owned)]
    dc.reconcile_contact_meshes(meshes, coordinates, [metadata[i] for i in order], descriptor, owned)
    for mesh, coordinates_i, owned_i, (before_cells, before_owned) in zip(meshes, coordinates, owned, snapshots):
        _assert_report(mesh)
        np.testing.assert_array_equal(coordinates_i, before_cells)
        np.testing.assert_array_equal(owned_i, before_owned)
        assert not np.shares_memory(mesh.vertices, mesh.vertices_tensor)
    for mesh, i in zip(meshes, order):
        np.testing.assert_array_equal(mesh.vertices_tensor, np.asarray(positions[i]).reshape(-1, 3))
    all_ids = np.concatenate([mesh.contact_report["contact_ids"] for mesh in meshes])
    count = len(np.unique(all_ids[all_ids >= 0]))
    assert all(mesh.contact_report["contact_count"] == count for mesh in meshes)
    return {metadata[i][:2]: mesh for i, mesh in zip(order, meshes)}


def test_three_way_mean_and_contact_identity_are_order_independent(extraction):
    descriptor = _descriptor([1, 1, 1], [StackRelationType.ERODE, StackRelationType.ERODE, StackRelationType.BASEMENT])
    positions = [[[.2, .3, .4]], [[.4, .5, .6]], [[.9, .8, .7]]]
    cells = [[[7, 11, 13]]] * 3
    metadata = [(i, 0, .4 + .1 * i) for i in range(3)]
    expected = np.mean(positions, axis=0)
    baseline = None
    for order in permutations(range(3)):
        result = _synthetic_reconcile(positions, cells, metadata, descriptor, order)
        ids = []
        for mesh in result.values():
            np.testing.assert_allclose(mesh.vertices, expected)
            ids.append(int(mesh.contact_report["contact_ids"][0]))
        assert ids[0] >= 0 and len(set(ids)) == 1
        if baseline is None:
            baseline = ids[0]
        assert ids[0] == baseline


def test_three_stack_extraction_shares_one_original_mean(extraction, monkeypatch):
    _, descriptor, inputs, options, levels = extraction("three_way_junction")
    extract = dc.compute_dual_contouring_v2
    call = Mock(side_effect=lambda **kw: extract(**dict(kw, max_workers=1)))
    monkeypatch.setattr(dc, "compute_dual_contouring_v2", call)
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    call.assert_called_once()
    assert len(meshes) == 3
    ids = [_assert_report(mesh) for mesh in meshes]
    common = set(ids[0][ids[0] >= 0]) & set(ids[1][ids[1] >= 0]) & set(ids[2][ids[2] >= 0])
    assert common
    for contact_id in common:
        indices = [int(np.flatnonzero(vertex_ids == contact_id)[0]) for vertex_ids in ids]
        expected = np.mean([_numpy(mesh.vertices_tensor)[i] for mesh, i in zip(meshes, indices)], axis=0)
        for mesh, i in zip(meshes, indices):
            np.testing.assert_allclose(mesh.vertices[i], expected)


@pytest.fixture
def real_contact_meshes(extraction, monkeypatch):
    def make(name, dtype="float64", backend=config.AvailableBackends.numpy):
        case, descriptor, inputs, options, levels = extraction(name, dtype=dtype, backend=backend)
        snapshots = []
        extract = dc.compute_dual_contouring_v2

        def capture(**kwargs):
            meshes = extract(**dict(kwargs, max_workers=1))
            snapshots.extend((_numpy(m.vertices).copy(), _numpy(m.edges).copy()) for m in meshes)
            return meshes

        monkeypatch.setattr(dc, "compute_dual_contouring_v2", capture)
        stage = Mock(wraps=dc.reconcile_contact_meshes)
        monkeypatch.setattr(dc, "reconcile_contact_meshes", stage)
        meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
        if name in ("three_stack_erosion", "three_stack_onlap"):
            masks = np.array([_numpy(o.squeezed_mask_array) for o in levels[0].outputs])
            np.testing.assert_array_equal(masks.sum(axis=0), 1)
            assert all(mask.any() and not mask.all() for mask in masks)
            owned = stage.call_args.args[4] if stage.call_args.args else stage.call_args.kwargs["corner_ownership"]
            assert any(np.any(rows.any(axis=1) & ~rows.all(axis=1)) for rows in owned)
        cells = [_numpy(m.dc_data.left_right_codes[m.dc_data.valid_voxels]) for m in meshes]
        metadata = [(m.stack_index, m.surface_index,
                     float(_numpy(levels[0].outputs[m.stack_index].scalar_field_at_sp)[m.surface_index]))
                    for m in meshes]
        return case, descriptor, meshes, cells, snapshots, metadata

    return make


REAL_CONTACT_CASES = ["planar_erosion", "onlap", "three_way_junction", "curved", "fault_mixed",
                      "three_stack_erosion", "three_stack_onlap", "same_group_competition"]


@pytest.mark.parametrize("backend", [config.AvailableBackends.numpy, config.AvailableBackends.PYTORCH])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", REAL_CONTACT_CASES)
def test_real_extraction_retains_face_indices_area_winding_and_supported_ids(real_contact_meshes, name, dtype, backend):
    _, _, meshes, cells, snapshots, _ = real_contact_meshes(name, dtype, backend)
    members = {}
    for surface, (mesh, coordinates, (vertices, faces)) in enumerate(zip(meshes, cells, snapshots)):
        ids = _assert_report(mesh)
        assert len(np.unique(ids[ids >= 0])) == np.count_nonzero(ids >= 0)
        np.testing.assert_array_equal(mesh.vertices[ids < 0], vertices[ids < 0])
        assert len(mesh.vertices) == len(vertices) == len(coordinates)
        np.testing.assert_array_equal(_numpy(mesh.vertices_tensor), vertices)
        original_faces = {tuple(face): index for index, face in enumerate(faces)}
        retained_indices = [original_faces[tuple(face)] for face in mesh.edges]
        assert np.all(np.diff(retained_indices) > 0)
        edges, counts = np.unique(np.sort(np.concatenate([
            mesh.edges[:, [0, 1]], mesh.edges[:, [1, 2]], mesh.edges[:, [2, 0]]]), axis=1),
            axis=0, return_counts=True)
        assert np.all(counts <= 2), (name, surface, "nonmanifold edge")
        neighbors = {}
        for a, b in edges:
            neighbors.setdefault(int(a), set()).add(int(b))
            neighbors.setdefault(int(b), set()).add(int(a))
        if neighbors:
            pending, visited = [next(iter(neighbors))], set()
            while pending:
                vertex = pending.pop()
                if vertex not in visited:
                    visited.add(vertex)
                    pending.extend(neighbors[vertex] - visited)
            assert visited == set(neighbors), (name, surface, "disconnected retained patch")
        before = vertices[mesh.edges]
        after = mesh.vertices[mesh.edges]
        original_normals = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
        final_normals = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
        assert np.all(np.linalg.norm(final_normals, axis=1) > 0), (name, surface, "zero area")
        assert np.all(np.einsum("ij,ij->i", original_normals, final_normals) > 0), (name, surface, "winding")
        ordinary = mesh.stack_index != 0 or name != "fault_mixed"
        if ordinary:
            assert np.all(np.isin(np.flatnonzero(ids >= 0), mesh.edges)), (name, surface, "unsupported IDs")
        for row in np.flatnonzero(ids >= 0):
            members.setdefault(int(ids[row]), []).append((surface, int(row)))
    assert members
    for contact_id, rows in members.items():
        assert len(rows) >= 2, (name, contact_id, "singleton ID")
        assert len({meshes[s].stack_index for s, _ in rows}) == len(rows)
        surface, row = rows[0]
        for partner, partner_row in rows[1:]:
            np.testing.assert_array_equal(cells[surface][row], cells[partner][partner_row])
            np.testing.assert_array_equal(meshes[surface].vertices[row], meshes[partner].vertices[partner_row])
    if name in ("three_stack_erosion", "three_stack_onlap"):
        assert any(len(rows) == 3 for rows in members.values())
    if name == "same_group_competition":
        assert [(m.stack_index, m.surface_index) for m in meshes] == [(0, 0), (0, 1), (1, 0)]
        assert set(map(tuple, cells[0])) & set(map(tuple, cells[1])) & set(map(tuple, cells[2]))
        competed = 0
        for rows in members.values():
            target_rows = [row for surface, row in rows if surface == 2]
            if not target_rows:
                continue
            row = target_rows[0]
            candidates = [np.flatnonzero(np.all(coordinates == cells[2][row], axis=1))
                          for coordinates in cells[:2]]
            if not all(len(candidate) == 1 for candidate in candidates):
                continue
            competed += 1
            distances = [np.sum((snapshots[s][0][candidate[0]] - snapshots[2][0][row]) ** 2)
                         for s, candidate in enumerate(candidates)]
            chosen = int(np.argmin(distances))
            assert (chosen, int(candidates[chosen][0])) in rows
            assert meshes[1 - chosen].contact_report["contact_ids"][candidates[1 - chosen][0]] == -1
        assert competed > 0


def _internal_boundary_gaps(extracted):
    case, descriptor, meshes, cells, _, metadata = extracted
    _, fault_pairs, truncation_pairs = contact_api._contact_relations(metadata, descriptor.stack_structure)
    all_edges, boundary_edges = [], []
    for mesh in meshes:
        edges = np.sort(np.concatenate([mesh.edges[:, [0, 1]], mesh.edges[:, [1, 2]],
                                        mesh.edges[:, [2, 0]]]), axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        all_edges.append({tuple(sorted(mesh.contact_report["contact_ids"][edge]))
                          for edge in unique if np.all(mesh.contact_report["contact_ids"][edge] >= 0)})
        boundary_edges.append(unique[counts == 1])
    domain_cells = np.array(list(product(*(range(n) for n in case.resolution))))
    lower, upper = domain_cells.min(axis=0), domain_cells.max(axis=0)
    seam_count = 0
    gaps = []
    for target, (mesh, edges) in enumerate(zip(meshes, boundary_edges)):
        controllers = {controller for controller, partner in truncation_pairs | fault_pairs if partner == target}
        if not controllers:
            continue
        for edge in edges:
            coordinates = cells[target][edge]
            domain_boundary = np.any(np.all(coordinates == lower, axis=0) |
                                     np.all(coordinates == upper, axis=0))
            ids = mesh.contact_report["contact_ids"][edge]
            if domain_boundary and not np.all(ids >= 0):
                continue
            seam_count += int(not domain_boundary)
            if not np.all(ids >= 0):
                gaps.append((target, coordinates.tolist(), ids.tolist(), "unmarked internal boundary"))
                continue
            key = tuple(sorted(ids))
            if not any(key in all_edges[c] for c in controllers):
                gaps.append((target, coordinates.tolist(), ids.tolist(), "missing actual controller edge"))
    return seam_count, gaps


@pytest.mark.parametrize("backend", [config.AvailableBackends.numpy, config.AvailableBackends.PYTORCH])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", REAL_CONTACT_CASES)
def test_real_extraction_internal_boundary_has_actual_controller_edge(real_contact_meshes, name, dtype, backend):
    extracted = real_contact_meshes(name, dtype, backend)
    seam_count, gaps = _internal_boundary_gaps(extracted)
    assert not gaps, (name, seam_count, gaps)
    _, descriptor, meshes, _, _, metadata = extracted
    _, _, pairs = contact_api._contact_relations(metadata, descriptor.stack_structure)
    patches = []
    for mesh in meshes:
        keys = np.sort(mesh.contact_report['contact_ids'][mesh.edges], axis=1)
        patches.append({tuple(row) for row in keys if row[0] >= 0 and row[0] < row[1] < row[2]})
    for controller, target in pairs:
        assert not patches[controller] & patches[target], (name, controller, target, "duplicate shared patch")
    # The original junction intentionally has full ownership: it has contacts,
    # but no truncated internal boundary. The chain cases exercise real seams.
    assert seam_count == 0 if name == "three_way_junction" else seam_count > 0


def test_fault_mixed_closes_junction_without_moving_fault_anchor(real_contact_meshes):
    extracted = real_contact_meshes("fault_mixed")
    seam_count, gaps = _internal_boundary_gaps(extracted)
    assert seam_count == 5
    assert not gaps
    meshes, cells = extracted[2:4]
    assert any(conflict["cell"] == (2, 2, 2) and conflict["reason"] == "fault_anchored"
               for conflict in meshes[0].contact_report["conflicts"])
    rows = [int(np.flatnonzero(np.all(coordinates == [2, 2, 2], axis=1))[0]) for coordinates in cells]
    assert meshes[1].contact_report["contact_ids"][rows[1]] >= 0
    assert len({mesh.contact_report['contact_ids'][row] for mesh, row in zip(meshes, rows)}) == 1
    for mesh, row in zip(meshes, rows):
        np.testing.assert_array_equal(mesh.vertices[row], _numpy(meshes[0].vertices_tensor)[rows[0]])
    assert meshes[0].contact_report['fault_junction_attachment_count'] == 1
    assert meshes[0].contact_report['fault_junction_rejected_count'] == 0
    assert not extracted[1].stack_structure.faults_relations[0, 2]
    assert len(meshes[0].contact_report['fault_overlap_vertices'][2]) == 0


@pytest.mark.parametrize("hidden", [0, 1])
def test_wholly_hidden_ownership_excludes_contact_grouping(extraction, hidden):
    descriptor = _descriptor([1, 1], [StackRelationType.ERODE, StackRelationType.BASEMENT])
    positions = [[[.2, .3, .4]], [[.6, .7, .8]]]
    owned = [np.ones((1, 8), dtype=bool), np.ones((1, 8), dtype=bool)]
    owned[hidden][:] = False
    result = _synthetic_reconcile(positions, [[[1, 2, 3]]] * 2,
                                  [(0, 0, .4), (1, 0, .5)], descriptor, ownership=owned)
    for i in range(2):
        mesh = result[(i, 0)]
        np.testing.assert_array_equal(mesh.vertices, positions[i])
        assert mesh.contact_report["contact_ids"][0] == -1
        assert mesh.contact_report["contact_count"] == 0


@pytest.mark.parametrize("masking", [MeshExtractionMaskingOptions.RAW, MeshExtractionMaskingOptions.DISJOINT])
def test_non_intersect_masking_is_not_a_contact_support_gate(extraction, monkeypatch, masking):
    _, descriptor, inputs, options, levels = extraction()
    options.evaluation_options.mesh_extraction_masking_options = masking
    extract = dc.compute_dual_contouring_v2
    call = Mock(side_effect=lambda **kw: extract(**dict(kw, max_workers=1)))
    monkeypatch.setattr(dc, "compute_dual_contouring_v2", call)
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    call.assert_called_once()
    assert len(meshes) == 2
    for mesh in meshes:
        _assert_report(mesh)


def test_extent_capping_rejected_before_extraction_to_preserve_contact_ids(extraction, monkeypatch):
    _, descriptor, inputs, options, levels = extraction()
    options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.SCALAR_LESS_EQUAL
    calls = []
    for helper in ("prepare_contact_constraints", "get_triangulation_codes", "_interp_on_edges",
                   "compute_dual_contouring_v2", "reconcile_contact_meshes"):
        call = Mock(side_effect=AssertionError("Extent capping reached contact extraction"))
        monkeypatch.setattr(dc, helper, call, raising=False)
        calls.append(call)
    with pytest.raises(ValueError, match="contact identities.*boundary vertices"):
        dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    for call in calls:
        call.assert_not_called()


@pytest.mark.parametrize("target,chosen", [(.7, 1), (.5, 0)])
def test_competing_same_group_surfaces_choose_nearest_original_with_stable_tie_break(extraction, target, chosen):
    descriptor = _descriptor([2, 1], [StackRelationType.ERODE, StackRelationType.BASEMENT])
    positions = [[[.25, .5, .5]], [[.75, .5, .5]], [[target, .5, .5]]]
    metadata = [(0, 0, .25), (0, 1, .75), (1, 0, target)]
    cells = [[[2, 3, 4]]] * 3
    stable_id = None
    for order in permutations(range(3)):
        result = _synthetic_reconcile(positions, cells, metadata, descriptor, order)
        a, other, b = result[(0, chosen)], result[(0, 1 - chosen)], result[(1, 0)]
        expected = (np.asarray(positions[chosen]) + positions[2]) / 2
        np.testing.assert_allclose(a.vertices, expected)
        np.testing.assert_array_equal(a.vertices, b.vertices)
        np.testing.assert_array_equal(other.vertices, positions[1 - chosen])
        a_id, b_id = a.contact_report["contact_ids"][0], b.contact_report["contact_ids"][0]
        assert a_id == b_id and a_id >= 0
        assert other.contact_report["contact_ids"][0] == -1
        if stable_id is None:
            stable_id = a_id
        assert a_id == stable_id


def test_geological_eligibility_precedes_nearest_position(extraction):
    descriptor = _descriptor([1, 1, 1], [StackRelationType.FAULT, StackRelationType.ERODE, StackRelationType.BASEMENT])
    positions = [[[.49, .5, .5]], [[.2, .5, .5]], [[.5, .5, .5]]]
    result = _synthetic_reconcile(positions, [[[1, 2, 3]]] * 3,
                                  [(0, 0, .49), (1, 0, .2), (2, 0, .5)], descriptor)
    np.testing.assert_array_equal(result[(0, 0)].vertices, positions[0])
    assert result[(0, 0)].contact_report["contact_ids"][0] == -1
    np.testing.assert_allclose(result[(1, 0)].vertices, [[.35, .5, .5]])
    np.testing.assert_array_equal(result[(1, 0)].vertices, result[(2, 0)].vertices)


def test_sparse_cells_keep_vertex_order_and_unshared_positions(extraction):
    descriptor = _descriptor([1, 1], [StackRelationType.ERODE, StackRelationType.BASEMENT])
    cells = [[[9000000, 2, 3], [4, 5, 6], [8, 9, 10]],
             [[8, 9, 10], [10000000, 2, 3], [9000000, 2, 3]]]
    positions = [[[.1, .2, .3], [.2, .4, .6], [.3, .2, .1]],
                 [[.7, .6, .5], [.6, .4, .2], [.9, .8, .7]]]
    result = _synthetic_reconcile(positions, cells, [(0, 0, .4), (1, 0, .5)], descriptor)
    a, b = result[(0, 0)], result[(1, 0)]
    np.testing.assert_array_equal(a.vertices[1], positions[0][1])
    np.testing.assert_array_equal(b.vertices[1], positions[1][1])
    for i, j in ((0, 2), (2, 0)):
        np.testing.assert_allclose(a.vertices[i], (np.asarray(positions[0][i]) + positions[1][j]) / 2)
        np.testing.assert_array_equal(a.vertices[i], b.vertices[j])
        assert a.contact_report["contact_ids"][i] == b.contact_report["contact_ids"][j] >= 0
    assert a.contact_report["contact_ids"][1] == b.contact_report["contact_ids"][1] == -1
    assert a.contact_report["contact_ids"][0] != a.contact_report["contact_ids"][2]


def test_fault_anchors_are_original_and_triangle_removal_is_directional(extraction):
    faults = np.zeros((4, 4), dtype=bool)
    faults[0, 1] = True
    descriptor = _descriptor([1] * 4, [StackRelationType.FAULT, StackRelationType.ERODE,
                                       StackRelationType.BASEMENT, StackRelationType.FAULT], faults)
    shared = [[i, 0, 0] for i in range(3)]
    cells = [shared, shared + [[3, 0, 0]], shared, shared]
    positions = [[[.1, .2, .3], [.2, .4, .3], [.3, .2, .5]],
                 [[.8, .2, .3], [.8, .4, .3], [.8, .2, .5], [.9, .9, .9]],
                 [[.6, .2, .3], [.6, .4, .3], [.6, .2, .5]],
                 [[.9, .2, .3], [.9, .4, .3], [.9, .2, .5]]]
    faces = [[[0, 1, 2]], [[0, 1, 2], [0, 1, 3]], [[0, 1, 2]], [[0, 1, 2]]]
    metadata = [(i, 0, .4) for i in range(4)]
    for order in (range(4), reversed(range(4))):
        result = _synthetic_reconcile(positions, cells, metadata, descriptor, order, faces)
        fault, layer, unrelated_fault = result[(0, 0)], result[(1, 0)], result[(3, 0)]
        np.testing.assert_array_equal(fault.vertices, positions[0])
        np.testing.assert_array_equal(layer.vertices[:3], positions[0])
        np.testing.assert_array_equal(layer.vertices[3], positions[1][3])
        np.testing.assert_array_equal(fault.edges, faces[0])
        np.testing.assert_array_equal(layer.edges, [[0, 1, 3]])
        np.testing.assert_array_equal(unrelated_fault.vertices, positions[3])
        np.testing.assert_array_equal(unrelated_fault.edges, faces[3])
        assert np.all(unrelated_fault.contact_report["contact_ids"] == -1)
        np.testing.assert_array_equal(layer.vertices_tensor, positions[1])


@pytest.mark.parametrize("name,partners", [("fault", [{1}, {0}]), ("fault_mixed", [{1}, {0}, set()])])
def test_fault_extraction_qef_uses_only_explicit_directional_partners(extraction, monkeypatch, name, partners):
    _, descriptor, inputs, options, levels = extraction(name)
    inject = Mock()
    monkeypatch.setattr(contact_api, "find_and_inject_multi_surface_constraints_multicore", inject, raising=False)
    extract = dc.compute_dual_contouring_v2
    call = Mock(side_effect=lambda **kw: extract(**dict(kw, max_workers=1)))
    monkeypatch.setattr(dc, "compute_dual_contouring_v2", call)
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    inject.assert_called_once()
    assert inject.call_args.kwargs["allowed_partners_per_surface"] == partners
    call.assert_called_once()
    assert len(meshes) == len(partners)
    for mesh in meshes:
        _assert_report(mesh)


@pytest.mark.parametrize("backend,dtype", [
    (config.AvailableBackends.numpy, "float32"),
    (config.AvailableBackends.PYTORCH, "float32"),
    (config.AvailableBackends.PYTORCH, "float64"),
])
def test_supported_backend_and_dtype_return_detached_numpy_arrays(extraction, monkeypatch, backend, dtype):
    if backend is config.AvailableBackends.PYTORCH:
        pytest.importorskip("torch")
    _, descriptor, inputs, options, levels = extraction(dtype=dtype, backend=backend)
    assert _numpy(levels[0].outputs[0].exported_fields.scalar_field).dtype == np.dtype(dtype)
    extract = dc.compute_dual_contouring_v2
    monkeypatch.setattr(dc, "compute_dual_contouring_v2", lambda **kw: extract(**dict(kw, max_workers=1)))
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    assert BackendTensor.engine_backend is backend and BackendTensor.dtype == dtype
    assert any(mesh.contact_report["contact_count"] > 0 for mesh in meshes)
    for mesh in meshes:
        _assert_report(mesh)
        assert np.issubdtype(_numpy(mesh.vertices_tensor).dtype, np.floating)
        assert not np.shares_memory(mesh.vertices, _numpy(mesh.vertices_tensor))
        if backend is config.AvailableBackends.PYTORCH:
            import torch
            assert isinstance(mesh.vertices_tensor, torch.Tensor)


@pytest.mark.parametrize("backend,dtype", [
    (config.AvailableBackends.numpy, "float64"),
    (config.AvailableBackends.PYTORCH, "float32"),
    (config.AvailableBackends.PYTORCH, "float64"),
])
@pytest.mark.parametrize("name", ["fault", "fault_parallel"])
def test_fault_extraction_runs_qef_and_preserves_directed_geometry(extraction, monkeypatch, backend, dtype, name):
    _, descriptor, inputs, options, levels = extraction(name, dtype=dtype, backend=backend)
    levels[0].outputs[0].squeezed_mask_array[:] = False
    extract = dc.compute_dual_contouring_v2
    before = []

    def capture(**kwargs):
        meshes = extract(**dict(kwargs, max_workers=1))
        before.extend((_numpy(mesh.vertices).copy(), _numpy(mesh.edges).copy()) for mesh in meshes)
        return meshes

    monkeypatch.setattr(dc, "compute_dual_contouring_v2", capture)
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    fault, target = meshes
    fault_ids, target_ids = _assert_report(fault), _assert_report(target)
    common = set(fault_ids[fault_ids >= 0]) & set(target_ids[target_ids >= 0])
    assert common
    np.testing.assert_array_equal(fault.vertices, before[0][0])
    np.testing.assert_array_equal(fault.edges, before[0][1])
    for contact_id in common:
        i, j = np.flatnonzero(fault_ids == contact_id)[0], np.flatnonzero(target_ids == contact_id)[0]
        np.testing.assert_array_equal(target.vertices[j], before[0][0][i])
    overlap = target.contact_report['fault_overlap_vertices'][1]
    remove = np.all(np.isin(before[1][1], overlap), axis=1)
    assert bool(remove.any()) == (name == "fault_parallel")
    np.testing.assert_array_equal(target.edges, before[1][1][~remove])
    if backend is config.AvailableBackends.numpy:
        monkeypatch.setattr(dc, "compute_dual_contouring_v2", extract)
        _, descriptor, inputs, options, levels = extraction(name)
        levels[0].outputs[0].squeezed_mask_array[:] = False
        monkeypatch.setattr(dc, "DUAL_CONTOURING_VERTEX_OVERLAP", config.DualContouringOverlap.pretty)
        legacy = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
        for actual, reference in zip(meshes, legacy):
            np.testing.assert_array_equal(actual.vertices, reference.vertices)
            np.testing.assert_array_equal(actual.edges, reference.edges)


def test_descriptor_copy_preserves_external_callback_and_array_identity(extraction, monkeypatch):
    _, descriptor, inputs, options, levels = extraction()

    class Callback:
        def __call__(self, xyz):
            return xyz[:, 2]

        def __deepcopy__(self, memo):
            raise AssertionError("External callbacks must not be deep-copied")

    callback = Callback()
    functions = CustomInterpolationFunctions(np.array([.39]), callback)
    stacks = descriptor.stack_structure
    stacks.interp_functions_per_stack = [functions, None]
    cursor = stacks.stack_number
    gradients = dc.interpolate_all_fields_no_octree

    def capture(interpolation_input, options, data_descriptor):
        copied = data_descriptor.stack_structure
        assert data_descriptor is not descriptor and copied is not stacks
        assert copied.interp_functions_per_stack[0] is functions
        assert copied.interp_functions_per_stack[0].implicit_function is callback
        assert copied.number_of_surfaces_per_stack is stacks.number_of_surfaces_per_stack
        assert copied.faults_relations is stacks.faults_relations
        return gradients(interpolation_input, options, data_descriptor)

    call = Mock(side_effect=capture)
    monkeypatch.setattr(dc, "interpolate_all_fields_no_octree", call)
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    call.assert_called_once()
    assert len(meshes) == 2
    assert stacks.stack_number == cursor
    assert stacks.interp_functions_per_stack[0] is functions


@pytest.mark.parametrize("name", ["planar_erosion", "onlap"])
def test_compute_model_external_fields_have_shared_cell_contact_ids(extraction, name):
    # Use only backend isolation, not the synthetic factory or its gradient patch.
    case = build_contact_case(name, resolution=6)
    functions = [CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([level]),
        implicit_function=lambda xyz, normal=normal: xyz @ normal,
        gx_function=lambda xyz, normal=normal: np.full(len(xyz), normal[0]),
        gy_function=lambda xyz, normal=normal: np.full(len(xyz), normal[1]),
        gz_function=lambda xyz, normal=normal: np.full(len(xyz), normal[2]),
    ) for normal, level in zip(case.normals, case.levels)]
    descriptor = _descriptor([1, 1], case.relations)
    descriptor.stack_structure.interp_functions_per_stack = functions
    grid = EngineGrid(octree_grid=RegularGrid(np.array([0., 1., 0., 1., 0., 1.]), [6, 6, 6]))
    inputs = InterpolationInput(SurfacePoints(np.empty((0, 3))),
                                Orientations(np.empty((0, 3)), np.empty((0, 3))), grid)
    original_values = grid.values.copy()
    options = InterpolationOptions.from_args(10., 1.)
    options.evaluation_options.number_octree_levels = 2
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    solution = compute_model(inputs, options, descriptor)
    assert len(solution.octrees_output) == len(solution.dc_meshes) == 2
    assert inputs.grid is grid
    np.testing.assert_array_equal(grid.values, original_values)
    meshes = solution.dc_meshes
    ids = [_assert_report(mesh) for mesh in meshes]
    common_ids = np.intersect1d(ids[0][ids[0] >= 0], ids[1][ids[1] >= 0])
    assert len(common_ids) > 0
    for contact_id in common_ids:
        indices = [int(np.flatnonzero(v == contact_id)[0]) for v in ids]
        a, b = indices
        cells = [m.dc_data.left_right_codes[m.dc_data.valid_voxels] for m in meshes]
        np.testing.assert_array_equal(cells[0][a], cells[1][b])
        expected = (_numpy(meshes[0].vertices_tensor)[a] + _numpy(meshes[1].vertices_tensor)[b]) / 2
        np.testing.assert_allclose(meshes[0].vertices[a], expected)
        np.testing.assert_array_equal(meshes[0].vertices[a], meshes[1].vertices[b])


@pytest.mark.parametrize("members", [
    ("none", "pretty"), ("none", "watertight"), ("pretty", "watertight"),
    ("none", "pretty", "watertight"),
])
def test_legacy_combined_flags_match_pretty_dispatch_and_output(extraction, monkeypatch, members):
    combined = config.DualContouringOverlap(0)
    for member in members:
        combined |= getattr(config.DualContouringOverlap, member)
    inject = dc.find_and_inject_multi_surface_constraints_multicore
    extract = dc.compute_dual_contouring_v2
    calls = [Mock(side_effect=lambda **kw: inject(**dict(kw, max_workers=1))),
             Mock(wraps=dc.average_overlapping_vertices),
             Mock(wraps=dc.remove_fault_overlap_triangles)]
    for helper, call in zip(("find_and_inject_multi_surface_constraints_multicore",
                             "average_overlapping_vertices", "remove_fault_overlap_triangles"), calls):
        monkeypatch.setattr(dc, helper, call)
    monkeypatch.setattr(dc, "compute_dual_contouring_v2", lambda **kw: extract(**dict(kw, max_workers=1)))
    monkeypatch.setattr(dc, "reconcile_contact_meshes", Mock(side_effect=AssertionError("Legacy flags entered contact stage")))
    monkeypatch.setattr(dc, "prepare_contact_constraints",
                        Mock(side_effect=AssertionError("Legacy flags prepared contacts")), raising=False)
    outputs, dispatch = [], []
    for mode in (config.DualContouringOverlap.pretty, combined):
        _, descriptor, inputs, options, levels = extraction()
        monkeypatch.setattr(dc, "DUAL_CONTOURING_VERTEX_OVERLAP", mode)
        meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
        outputs.append([(m.vertices.copy(), m.edges.copy()) for m in meshes])
        dispatch.append([call.call_count for call in calls])
        for call in calls:
            call.reset_mock()
    assert dispatch == [[2, 2, 2], [2, 2, 2]]
    for (vertices, faces), (combined_vertices, combined_faces) in zip(*outputs):
        np.testing.assert_array_equal(vertices, combined_vertices)
        np.testing.assert_array_equal(faces, combined_faces)


@pytest.mark.parametrize("legacy", ["none", "pretty", "watertight"])
def test_combined_contact_aware_flags_fail_before_extraction(extraction, monkeypatch, legacy):
    _, descriptor, inputs, options, levels = extraction()
    combined = config.DualContouringOverlap.contact_aware | getattr(config.DualContouringOverlap, legacy)
    monkeypatch.setattr(dc, "DUAL_CONTOURING_VERTEX_OVERLAP", combined)
    calls = []
    for helper in ("prepare_contact_constraints", "get_triangulation_codes", "_interp_on_edges",
                   "compute_dual_contouring_v2", "reconcile_contact_meshes"):
        call = Mock(side_effect=AssertionError("Combined contact flags reached extraction"))
        calls.append(call)
        monkeypatch.setattr(dc, helper, call, raising=False)
    with pytest.raises(ValueError, match="contact_aware|overlap mode"):
        dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    for call in calls:
        call.assert_not_called()


@pytest.mark.parametrize("mode", ["none", "pretty", "watertight"])
def test_existing_modes_do_not_reconcile_contacts(extraction, monkeypatch, mode):
    _, descriptor, inputs, options, levels = extraction()
    monkeypatch.setattr(dc, "DUAL_CONTOURING_VERTEX_OVERLAP", getattr(config.DualContouringOverlap, mode))
    monkeypatch.setattr(dc, "reconcile_contact_meshes", Mock(side_effect=AssertionError("Legacy mode entered contact stage")))
    monkeypatch.setattr(dc, "prepare_contact_constraints",
                        Mock(side_effect=AssertionError("Legacy mode prepared contacts")), raising=False)
    extract = dc.compute_dual_contouring_v2
    monkeypatch.setattr(dc, "compute_dual_contouring_v2", lambda **kw: extract(**dict(kw, max_workers=1)))
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    assert len(meshes) == 2
    assert all(mesh.contact_report is None for mesh in meshes)


def test_contact_aware_environment_is_selected_once_in_fresh_process():
    env = dict(os.environ, DUAL_CONTOURING_VERTEX_OVERLAP="contact_aware", DEFAULT_BACKEND="numpy",
               DEFAULT_PYKEOPS="False", PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
    code = """
import os
from unittest.mock import patch
with patch('dotenv.load_dotenv', return_value=False):
    from gempy_engine import config
    import importlib
    dc = importlib.import_module('gempy_engine.API.dual_contouring.multi_scalar_dual_contouring')
assert config.DUAL_CONTOURING_VERTEX_OVERLAP is config.DualContouringOverlap.contact_aware
assert dc.DUAL_CONTOURING_VERTEX_OVERLAP is config.DualContouringOverlap.contact_aware
os.environ['DUAL_CONTOURING_VERTEX_OVERLAP'] = 'none'
assert dc.DUAL_CONTOURING_VERTEX_OVERLAP is config.DualContouringOverlap.contact_aware
assert config.DUAL_CONTOURING_VERTEX_OVERLAP is config.DualContouringOverlap.contact_aware
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
