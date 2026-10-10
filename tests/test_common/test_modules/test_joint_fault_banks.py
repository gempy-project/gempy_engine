"""Fault bridge export contracts and genuine production hierarchy integration."""

import copy
import importlib
import sys
from collections import Counter
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.stack_relation_type import StackRelationType as R
from tests.test_common.test_modules.test_joint_integration import _case, cpu_float64

bridge = importlib.import_module("gempy_engine.API.dual_contouring.joint_fault_banks")
adapter = importlib.import_module("gempy_engine.API.dual_contouring.joint_extraction")


def _fault_case():
    descriptor, inputs, options, levels = _case()
    stacks = descriptor.stack_structure
    stacks.number_of_points_per_stack = np.zeros(3, dtype=int)
    stacks.number_of_orientations_per_stack = np.zeros(3, dtype=int)
    stacks.number_of_surfaces_per_stack = np.array([1, 1, 1])
    stacks.masking_descriptor = [R.FAULT, R.BASEMENT, R.BASEMENT]
    stacks.interp_functions_per_stack = [object()] * 3
    stacks.faults_relations = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)
    for level in levels:
        output = level.outputs[0]
        level.outputs = [copy.copy(output) for _ in range(3)]
        for i, item in enumerate(level.outputs):
            item.scalar_field_at_sp = np.array([.13 + i * .1])
    return descriptor, inputs, options, levels


def _mock_sampler(monkeypatch, descriptor, calls):
    def query(points, bank=None, stacks=None):
        stacks = [0, 1, 2] if stacks is None else list(stacks)
        assert bank in (0, 1) or 1 not in stacks
        calls.append((bank, np.asarray(points).copy()))
        raw = np.array([points[:, 0], points[:, 2] + (bank or 0) * .1, points[:, 1]])
        gradients = np.tile(np.eye(3)[[0, 2, 1]][:, None, :], (1, len(points), 1))
        return raw[stacks], gradients[stacks]

    prepare = Mock(return_value=dict(
        fault_stack=0, affected_stacks=[1], unaffected_stacks=[2], query=query,
        diagnostics={"fixed_weights": True}))
    monkeypatch.setattr(bridge, "prepare_fault_bank_sampler", prepare)
    return prepare


def test_bank_assembly_preserves_original_export_order_and_disjoint_namespaces(monkeypatch):
    descriptor, inputs, options, levels = _fault_case()
    calls, extractions = [], []
    _mock_sampler(monkeypatch, descriptor, calls)
    core = importlib.import_module("gempy_engine.API.dual_contouring.joint_topology")

    def extract(origins, spans, shape, extent, samples, groups, indices, relations, isovalues, **kwargs):
        extractions.append(kwargs)
        assert kwargs["defer_emission"] is True and kwargs["include_reference"] is False
        assert set(spans) == {1, 2}
        raw, gradients = kwargs["sample_fields"](np.array([[.1, .2, .3]]))
        if "separator_contacts" in kwargs:
            bank = kwargs["separator_contacts"]["bank"]
            assert kwargs["separator_contacts"]["fault_pairs"] == {(0, 1)}
            assert kwargs["separator_contacts"]["controllers"] == {0: [], 1: [(0, -1)]}
            np.testing.assert_allclose(raw[0], (-1 if bank == 0 else 1) * (.1 - .13))
            np.testing.assert_array_equal(gradients[0, 0], [-1 if bank == 0 else 1, 0, 0])
            assert isovalues[0][0] == .13
        else:
            assert list(groups) == [2]
            assert R.FAULT not in relations
            assert isovalues[0][0] == .13
        keys = [("regular", i) for i in range(3)]
        return dict(vertices=np.eye(3), _triangle_plan=[[dict(keys=tuple(keys))] for _ in groups],
                    _key_to_id=dict(zip(keys, range(3))), vertex_keys=keys, seam_edges=[[0, 1]],
                    diagnostics={"coarsefine_seam_edge_count": 1})

    monkeypatch.setattr(bridge, "extract_adaptive_topology", extract)
    emitter = bridge.emit_adaptive_triangles
    emitted = []

    def emit(plan, ids):
        assert len(extractions) == 3
        emitted.append(plan)
        return emitter(plan, ids)

    monkeypatch.setattr(bridge, "emit_adaptive_triangles", emit)
    meshes = adapter.extract_joint_octree(descriptor, inputs, options, levels)
    assert len(meshes) == 3 and len(extractions) == 3
    assert len(emitted) == 3
    assert [m.exported_surface_index for m in meshes] == [0, 1, 2]
    assert [m.stack_index for m in meshes] == [0, 1, 2]
    assert [m.isovalue for m in meshes] == [.13, .23, .33]
    for mesh in meshes[:2]:
        np.testing.assert_array_equal(mesh.joint_face_bank_ids, [0, 1])
        assert set(mesh.joint_bank_ids) == {0, 1}
        assert not set(mesh.joint_vertex_ids[:3]) & set(mesh.joint_vertex_ids[3:])
        assert not set(mesh.joint_vertex_keys[:3]) & set(mesh.joint_vertex_keys[3:])
        for face, bank in zip(mesh.edges, mesh.joint_face_bank_ids):
            assert np.all(mesh.joint_bank_ids[face] == bank)
        assert len(mesh.joint_seam_edges) == 3
    np.testing.assert_array_equal(meshes[2].joint_bank_ids, [-1, -1, -1])
    assert len(meshes[2].edges) == 1
    assert all(bank in (0, 1, None) for bank, _ in calls)
    assert meshes[0].contact_report["interface"] == "bank_side_fault_skin_not_single_conforming_fault_mesh"


@pytest.mark.parametrize("rejected_partition", [1, -1], ids=["bank1", "unaffected"])
def test_all_partition_validation_barrier_before_emission(monkeypatch, rejected_partition):
    descriptor, inputs, options, levels = _fault_case()
    _mock_sampler(monkeypatch, descriptor, [])
    core = importlib.import_module("gempy_engine.API.dual_contouring.joint_topology")
    prepared = []

    def prepare(*args, **kwargs):
        assert kwargs["defer_emission"] is True and kwargs["include_reference"] is False
        partition = kwargs.get("separator_contacts", {}).get("bank", -1)
        if partition == rejected_partition:
            raise ValueError("unsupported_hanging_branch: rejected later partition")
        keys = [("regular", i) for i in range(3)]
        result = dict(vertices=np.eye(3), vertex_keys=keys, seam_edges=[[0, 1]], diagnostics={},
                      _triangle_plan=[[dict(keys=tuple(keys))] for _ in args[5]],
                      _key_to_id=dict(zip(keys, range(3))))
        prepared.append(result)
        return result

    emit = Mock(side_effect=AssertionError("partition validation barrier was bypassed"))
    monkeypatch.setattr(bridge, "extract_adaptive_topology", prepare)
    monkeypatch.setattr(core, "emit_adaptive_triangles", emit)
    monkeypatch.setattr(bridge, "emit_adaptive_triangles", emit)
    with pytest.raises(ValueError, match="unsupported_hanging_branch"):
        adapter.extract_joint_octree(descriptor, inputs, options, levels)
    assert len(prepared) == (1 if rejected_partition == 1 else 2)
    assert all("faces" not in result and "affected_faces" not in result for result in prepared)
    emit.assert_not_called()


@pytest.mark.parametrize("bad", ["multiple", "unknown", "unknown_orientation", "finite", "cross_partition"])
def test_fault_guards_before_sampler_or_emission(monkeypatch, bad):
    descriptor, inputs, options, levels = _fault_case()
    prepare = _mock_sampler(monkeypatch, descriptor, [])
    if bad == "multiple":
        descriptor.stack_structure.masking_descriptor[2] = R.FAULT
    elif bad == "unknown":
        descriptor.stack_structure.faults_input_data = [object(), None, None]
    elif bad == "unknown_orientation":
        descriptor.stack_structure.faults_input_data = [SimpleNamespace(finite_fault_defined=False), None, None]
    elif bad == "finite":
        from gempy_engine.core.data.kernel_classes.faults import FaultsData
        descriptor.stack_structure.faults_input_data = [FaultsData(finite_fault=object()), None, None]
    else:
        descriptor.stack_structure.masking_descriptor[1] = R.ERODE
    core = importlib.import_module("gempy_engine.API.dual_contouring.joint_topology")
    extract = Mock(side_effect=AssertionError("emission attempted"))
    monkeypatch.setattr(bridge, "extract_adaptive_topology", extract)
    with pytest.raises(ValueError, match="unsupported_"):
        adapter.extract_joint_octree(descriptor, inputs, options, levels)
    prepare.assert_not_called()
    extract.assert_not_called()


def test_ordinary_mesh_bank_fields_default_to_none():
    mesh = DualContouringMesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    assert mesh.joint_bank_ids is None and mesh.joint_face_bank_ids is None


@pytest.mark.parametrize("reason", ["unsupported_nonplanar_fault", "unverified_fault_banks"])
def test_sampler_rejection_precedes_core_emission(monkeypatch, reason):
    descriptor, inputs, options, levels = _fault_case()
    prepare = _mock_sampler(monkeypatch, descriptor, [])
    prepare.side_effect = ValueError(reason)
    core = importlib.import_module("gempy_engine.API.dual_contouring.joint_topology")
    extract = Mock(side_effect=AssertionError("emission attempted"))
    monkeypatch.setattr(bridge, "extract_adaptive_topology", extract)
    with pytest.raises(ValueError, match=reason):
        adapter.extract_joint_octree(descriptor, inputs, options, levels)
    extract.assert_not_called()


def _audit_fixed_queries(monkeypatch):
    feature = importlib.import_module("gempy_engine.API.interp_single._interp_single_feature")
    prepare = bridge.prepare_fault_bank_sampler
    inspected = []

    def audited_prepare(*args):
        no_solve = Mock(side_effect=AssertionError("query attempted a production solve"))
        monkeypatch.setattr(feature, "compute_weights", no_solve)
        scalar = importlib.import_module("gempy_engine.API.interp_single._interp_scalar_field")
        stack_ops = importlib.import_module("gempy_engine.API.interp_single._stack_ops")
        monkeypatch.setattr(scalar, "compute_weights", no_solve)
        monkeypatch.setattr(stack_ops, "compute_weights", no_solve)
        sampler = prepare(*args)
        query = sampler["query"]
        snapshots = copy.deepcopy(sampler["snapshots"])
        production = args[-1][-1]
        points = production.outputs[0].grid.values
        from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
        for stack, (_, weights, stack_options) in enumerate(sampler["snapshots"]):
            cached = WeightCache.load_weights(f"{stack_options.cache_model_name}.{stack}", False)
            assert cached is not None
            np.testing.assert_array_equal(weights, cached["weights"])
        for bank in (0, 1):
            raw, _ = query(points, bank=bank)
            for stack, (snapshot, weights, _) in enumerate(sampler["snapshots"]):
                output = production.outputs[stack]
                if stack in sampler["affected_stacks"]:
                    saturated = snapshot.fault_internal.fault_values_everywhere[0, :len(points)] == bank
                    assert saturated.any()
                else:
                    saturated = np.ones(len(points), dtype=bool)
                np.testing.assert_allclose(raw[stack, saturated],
                    output.exported_fields._scalar_field[:len(points)][saturated], rtol=1e-12, atol=1e-12)

        def audited_query(points, bank=None, stacks=None):
            result = query(points, bank=bank, stacks=stacks)
            for (actual, weights, _), (before, previous_weights, _) in zip(sampler["snapshots"], snapshots):
                np.testing.assert_array_equal(weights, previous_weights)
                assert not weights.flags.writeable
                np.testing.assert_array_equal(actual.xyz_to_interpolate, before.xyz_to_interpolate)
                for name in ("fault_values_everywhere", "fault_values_on_sp", "fault_values_ref", "fault_values_rest"):
                    np.testing.assert_array_equal(getattr(actual.fault_internal, name),
                                                  getattr(before.fault_internal, name))
            no_solve.assert_not_called()
            return result

        sampler["query"] = audited_query
        inspected.append(sampler)
        return sampler

    monkeypatch.setattr(bridge, "prepare_fault_bank_sampler", audited_prepare)
    return inspected


def _assert_actual_incidence(meshes):
    seams = set(map(tuple, meshes[0].joint_seam_edges))
    counts = Counter()
    surface_counts = []
    bank_ids = {}
    for mesh in meshes:
        local_counts = Counter()
        for vertex, bank in zip(mesh.joint_vertex_ids, mesh.joint_bank_ids):
            assert bank_ids.setdefault(int(vertex), int(bank)) == bank
        for triangle, bank in zip(mesh.edges, mesh.joint_face_bank_ids):
            assert np.all(mesh.joint_bank_ids[triangle] == bank)
            ids = mesh.joint_vertex_ids[triangle]
            edges = [tuple(sorted((int(a), int(b)))) for a, b in zip(ids, np.roll(ids, -1))]
            counts.update(edges)
            local_counts.update(edges)
        surface_counts.append(local_counts)
    assert seams
    shared = {edge for edge in counts if sum(bool(row[edge]) for row in surface_counts) > 1}
    assert shared == seams
    assert all(counts[edge] == 3 for edge in seams)
    for a, b in seams:
        assert bank_ids[a] == bank_ids[b]
        assert sorted(row[(a, b)] for row in surface_counts if row[(a, b)]) == [1, 2]


@pytest.mark.parametrize("mixed", [False, True], ids=["full_refine", "mixed_leaves"])
def test_actual_cokriging_fault_and_independent_stack(monkeypatch, mixed):
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.core.data import SurfacePoints, Orientations, TensorsStructure
    from gempy_engine.core.data.engine_grid import EngineGrid
    from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
    from gempy_engine.core.data.interpolation_input import InterpolationInput
    from gempy_engine.core.data.options import InterpolationOptions
    from gempy_engine.core.data.regular_grid import RegularGrid
    from gempy_engine.core.data.stacks_structure import StacksStructure
    points = np.array([
        [.43, .1, .1], [.43, .8, .1], [.43, .1, .8],
        [.1, .1, .57], [.1, .8, .57], [.2, .4, .57],
        [.7, .1, .37], [.7, .8, .37], [.9, .4, .37],
        [.1, .61, .1], [.8, .61, .1], [.1, .61, .8],
    ])
    root = RegularGrid([0., 1., 0., 1., 0., 1.], [4, 4, 4])
    inputs = InterpolationInput(SurfacePoints(points, nugget_effect_scalar=1e-12), Orientations(
        np.array([[.43, .4, .4], [.7, .4, .37], [.4, .61, .4]]),
        np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]), nugget_effect_grad=1e-12),
        EngineGrid(octree_grid=root), unit_values=np.arange(1, 5))
    descriptor = InputDataDescriptor(TensorsStructure(np.array([3, 6, 3])),
        StacksStructure(np.array([3, 6, 3]), np.array([1, 1, 1]), np.array([1, 1, 1]),
                        [R.FAULT, R.BASEMENT, R.BASEMENT],
                        faults_relations=np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)))
    options = InterpolationOptions.init_octree_options(refinement=2)
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.octree_min_level = 0
    options.evaluation_options.mesh_extraction_overlap = "joint"
    _controlled_refinement(monkeypatch, root, mixed, seam=True)
    inspected = _audit_fixed_queries(monkeypatch)
    original = copy.deepcopy(inputs)
    solution = compute_model(inputs, options, descriptor)
    assert len(solution.dc_meshes) == 3
    assert len(inspected) == 1
    separator, displaced, independent = solution.dc_meshes
    assert set(separator.joint_bank_ids) == {0, 1}
    assert set(displaced.joint_bank_ids) == {0, 1}
    assert set(independent.joint_bank_ids) == {-1}
    assert len(independent.edges) > 0
    for bank in (0, 1):
        ids = displaced.joint_bank_ids == bank
        assert ids.any()
        # Actual cokriging (not an analytical callback) has small residual
        # curvature while retaining two distinctly displaced near-horizontal banks.
        assert np.ptp(displaced.vertices[ids, 2]) < 1e-3
    assert abs(displaced.vertices[displaced.joint_bank_ids == 0, 2].mean() -
               displaced.vertices[displaced.joint_bank_ids == 1, 2].mean()) > .1
    assert set(adapter._collect_joint_leaves(solution.octrees_output, 3)[1]) == ({1, 2} if mixed else {1})
    _assert_actual_incidence(solution.dc_meshes)
    if mixed:
        assert separator.contact_report["coarsefine_seam_edge_count"] > 0
    np.testing.assert_array_equal(inputs.surface_points.sp_coords, original.surface_points.sp_coords)


def _controlled_refinement(monkeypatch, root, mixed, seam=False, coarse_parents=None):
    internals = importlib.import_module("gempy_engine.modules.octrees_topology._octree_internals")
    mark_voxel = internals._mark_voxel

    def select(corner_ids):
        shifts, _ = mark_voxel(corner_ids)
        mask = np.ones(len(root.values), dtype=bool)
        if mixed:
            if coarse_parents is not None:
                extent = np.asarray(root.orthogonal_extent)
                indices = np.floor((root.values - extent[::2]) /
                    ((extent[1::2] - extent[::2]) / root.regular_grid_shape)).astype(int)
                for parent in coarse_parents:
                    mask[np.all(indices == parent, axis=1)] = False
            elif seam:
                mask = root.values[:, 1] >= np.mean(np.asarray(root.orthogonal_extent)[2:4])
            else:
                mask[0] = False
        return shifts, mask

    monkeypatch.setattr(internals, "_mark_voxel", select)


@pytest.mark.parametrize("mixed", [False, True], ids=["full_refine", "mixed_leaves"])
def test_actual_model7_root7_two_levels(monkeypatch, mixed):
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.core.data.regular_grid import RegularGrid
    from tests.fixtures.model7_combination import model7_combination_factory
    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=True)
    inputs.grid.octree_grid = RegularGrid(inputs.grid.octree_grid.orthogonal_extent, [7, 7, 7])
    options.evaluation_options.octree_min_level = 0
    _controlled_refinement(monkeypatch, inputs.grid.octree_grid, mixed,
                           coarse_parents=((5, 3, 1), (5, 3, 3)))
    inspected = _audit_fixed_queries(monkeypatch)
    options.evaluation_options.mesh_extraction_overlap = "joint"
    original = copy.deepcopy(inputs)
    assert len(inputs.surface_points.sp_coords) == 102
    assert len(inputs.orientations.dip_positions) == 8
    core = importlib.import_module("gempy_engine.API.dual_contouring.joint_topology")
    extract = core.extract_adaptive_topology
    results = []

    def record(*args, **kwargs):
        assert kwargs["include_reference"] is False
        assert kwargs["defer_emission"] is True
        result = extract(*args, **kwargs)
        assert result.get("faces") is None and result.get("affected_faces") is None
        results.append(result)
        return result

    monkeypatch.setattr(bridge, "extract_adaptive_topology", record)
    solution = compute_model(inputs, options, descriptor)
    assert len(inspected) == 1
    assert len(results) == 2 and [result["bank"] for result in results] == [0, 1]
    origins, spans, shape, _, _, metadata = adapter._collect_joint_leaves(solution.octrees_output, 3)
    np.testing.assert_array_equal(shape, [14, 14, 14])
    assert len(origins) == (2730 if mixed else 2744)
    assert np.count_nonzero(spans == 2) == (2 if mixed else 0)
    assert np.count_nonzero(spans == 1) == (2728 if mixed else 2744)
    assert np.sum(spans ** 3) == 2744
    assert len(solution.dc_meshes) == 4
    expected_triangles = ([[426, 78, 154, 181], [426, 234, 670, 875]] if mixed else
                         [[442, 78, 156, 182], [442, 234, 676, 878]])
    assert [[len(f) for f in result["faces"]] for result in results] == expected_triangles
    assert [len(mesh.edges) for mesh in solution.dc_meshes] == np.sum(expected_triangles, axis=0).tolist()
    assert [(m.stack_index, m.surface_index, m.exported_surface_index, m.isovalue)
            for m in solution.dc_meshes] == [(*row[:2], i, row[2]) for i, row in enumerate(metadata)]
    _assert_actual_incidence(solution.dc_meshes)
    global_keys, global_banks = {}, {}
    for mesh in solution.dc_meshes:
        assert set(mesh.joint_face_bank_ids) == {0, 1}
        assert mesh.contact_report["interface"] == "bank_side_fault_skin_not_single_conforming_fault_mesh"
        assert mesh.contact_report["coarsefine_seam_edge_count"] == (4 if mixed else 0)
        for vertex, bank, key in zip(mesh.joint_vertex_ids, mesh.joint_bank_ids, mesh.joint_vertex_keys):
            assert key[0] == bank
            assert global_keys.setdefault(int(vertex), key) == key
            assert global_banks.setdefault(int(vertex), int(bank)) == bank
        for bank, result in enumerate(results):
            assert np.count_nonzero(mesh.joint_face_bank_ids == bank) == len(result["faces"][mesh.exported_surface_index])
    assert not ({k for i, k in global_keys.items() if global_banks[i] == 0} &
                {k for i, k in global_keys.items() if global_banks[i] == 1})
    assert max(i for i, bank in global_banks.items() if bank == 0) < min(
        i for i, bank in global_banks.items() if bank == 1)
    separator = solution.dc_meshes[0]
    counts = [Counter(tuple(sorted((int(a), int(b)))) for tri in mesh.joint_vertex_ids[mesh.edges]
                      for a, b in zip(tri, np.roll(tri, -1))) for mesh in solution.dc_meshes]
    for bank, result in enumerate(results):
        assert len(result["seam_edges"]) == ([51, 90] if mixed else [52, 91])[bank]
        assert result["diagnostics"]["strict_crossings"]
        assert result["diagnostics"]["no_uniform_fallback"]
        assert result["diagnostics"]["no_post_reconciliation"]
        assert result["diagnostics"]["coarsefine_seam_edge_count"] == (2 if mixed else 0)
        coarsefine = []
        for a, b in separator.joint_seam_edges:
            a, b = int(a), int(b)
            if global_banks[a] == bank and global_keys[a][1][-2] != global_keys[b][1][-2]:
                coarsefine.append((a, b))
                assert sorted(row[(a, b)] for row in counts if row[(a, b)]) == [1, 2]
                identities = global_keys[a][1][1]
                controller = 0 if any(identity[0] == 0 for identity in identities) else 1
                assert counts[controller][(a, b)] == 2
                assert sum(row[(a, b)] for row in counts[controller + 1:]) == 1
        assert len(coarsefine) == (2 if mixed else 0)
        triangles = separator.vertices[separator.edges[separator.joint_face_bank_ids == bank]]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        fault_gradient = np.asarray(inspected[0]["diagnostics"]["plane"][:3])
        assert np.all((normals @ fault_gradient) * (-1 if bank == 0 else 1) > 0)
        for mesh in solution.dc_meshes[1:]:
            values, _ = inspected[0]["query"](mesh.vertices[mesh.joint_bank_ids == bank], bank=bank)
            side = (-1 if bank == 0 else 1) * (values[0] - metadata[0][2])
            assert side.max() < 1e-10
    assert inspected[0]["diagnostics"]["fixed_weights"]
    np.testing.assert_array_equal(inputs.surface_points.sp_coords, original.surface_points.sp_coords)
    np.testing.assert_array_equal(inputs.orientations.dip_positions, original.orientations.dip_positions)
    np.testing.assert_array_equal(inputs.orientations.dip_gradients, original.orientations.dip_gradients)
    np.testing.assert_array_equal(inputs.unit_values, np.arange(1, 6))


def test_actual_model7_lower_half_coarsening_rejects_before_emission(monkeypatch):
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.core.data.regular_grid import RegularGrid
    from tests.fixtures.model7_combination import model7_combination_factory
    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=True)
    root = RegularGrid(inputs.grid.octree_grid.orthogonal_extent, [7, 7, 7])
    inputs.grid.octree_grid = root
    options.evaluation_options.octree_min_level = 0
    options.evaluation_options.mesh_extraction_overlap = "joint"
    _controlled_refinement(monkeypatch, root, True, seam=True)
    core = importlib.import_module("gempy_engine.API.dual_contouring.joint_topology")
    emit = Mock(side_effect=AssertionError("invalid hierarchy reached triangle emission"))
    monkeypatch.setattr(core, "emit_adaptive_triangles", emit)
    with pytest.raises(ValueError, match="unsupported_dual_junction_incidence"):
        compute_model(inputs, options, descriptor)
    emit.assert_not_called()


def test_actual_model7_natural_root9_two_levels(monkeypatch):
    """Natural production refinement, never a controlled mask or uniform fallback."""
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.core.data.regular_grid import RegularGrid
    from tests.fixtures.model7_combination import model7_combination_factory
    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=True)
    extent = inputs.grid.octree_grid.orthogonal_extent.copy()
    root = RegularGrid(extent, [9, 9, 9])
    inputs.grid.octree_grid = root
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.octree_min_level = 0
    options.evaluation_options.mesh_extraction_overlap = "joint"
    original = copy.deepcopy(inputs)
    inspected = _audit_fixed_queries(monkeypatch)
    core = importlib.import_module("gempy_engine.API.dual_contouring.joint_topology")
    extract = core.extract_adaptive_topology
    results = []

    def record(*args, **kwargs):
        assert kwargs["defer_emission"] is True and kwargs["include_reference"] is False
        result = extract(*args, **kwargs)
        assert result.get("faces") is None and result.get("affected_faces") is None
        results.append(result)
        return result

    monkeypatch.setattr(bridge, "extract_adaptive_topology", record)
    solution = compute_model(inputs, options, descriptor)
    origins, spans, shape, actual_extent, _, metadata = adapter._collect_joint_leaves(solution.octrees_output, 3)
    np.testing.assert_array_equal(actual_extent, extent)
    np.testing.assert_array_equal(shape, [18, 18, 18])
    assert len(origins) == 3781
    assert np.count_nonzero(spans == 1) == 3488
    assert np.count_nonzero(spans == 2) == 293
    assert np.sum(spans ** 3) == 5832
    assert len(solution.octrees_output) == 2 and len(results) == 2 and len(inspected) == 1
    assert [result["bank"] for result in results] == [0, 1]
    assert len(solution.dc_meshes) == 4
    assert [[len(f) for f in result["faces"]] for result in results] == [
        [748, 136, 238, 306], [748, 408, 1146, 1572]]
    assert [len(mesh.edges) for mesh in solution.dc_meshes] == [1496, 544, 1384, 1878]
    assert [len(result["seam_edges"]) for result in results] == [68, 119]
    assert [(mesh.stack_index, mesh.surface_index, mesh.exported_surface_index, mesh.isovalue)
            for mesh in solution.dc_meshes] == [(*row[:2], i, row[2]) for i, row in enumerate(metadata)]
    _assert_actual_incidence(solution.dc_meshes)
    keys, bank_labels = {}, {}
    counts = [Counter(tuple(sorted((int(a), int(b)))) for tri in mesh.joint_vertex_ids[mesh.edges]
                      for a, b in zip(tri, np.roll(tri, -1))) for mesh in solution.dc_meshes]
    for mesh in solution.dc_meshes:
        assert len(mesh.joint_bank_ids) == len(mesh.vertices) == len(mesh.joint_vertex_keys)
        assert len(mesh.joint_face_bank_ids) == len(mesh.edges)
        assert set(mesh.joint_bank_ids) == set(mesh.joint_face_bank_ids) == {0, 1}
        assert mesh.contact_report["coarsefine_seam_edge_count"] == 0
        assert len(mesh.joint_seam_edges) == 187
        for vertex, bank, key in zip(mesh.joint_vertex_ids, mesh.joint_bank_ids, mesh.joint_vertex_keys):
            assert key[0] == bank
            assert keys.setdefault(int(vertex), key) == key
            assert bank_labels.setdefault(int(vertex), int(bank)) == bank
        for bank, result in enumerate(results):
            assert np.count_nonzero(mesh.joint_face_bank_ids == bank) == len(result["faces"][mesh.exported_surface_index])
    ids = [{i for i, label in bank_labels.items() if label == bank} for bank in (0, 1)]
    assert not ids[0] & ids[1]
    assert not {keys[i] for i in ids[0]} & {keys[i] for i in ids[1]}
    assert max(ids[0]) < min(ids[1])
    separator = solution.dc_meshes[0]
    for a, b in separator.joint_seam_edges:
        a, b = int(a), int(b)
        # Mixed leaves do not imply mixed-resolution contact interfaces.
        assert keys[a][1][-2] == keys[b][1][-2]
        identities = keys[a][1][1]
        assert len(identities) == 2
        controller = 0 if any(identity[0] == 0 for identity in identities) else 1
        assert counts[controller][(a, b)] == 2
        assert sum(row[(a, b)] for row in counts[controller + 1:]) == 1
    for bank, result in enumerate(results):
        assert result["diagnostics"]["strict_crossings"]
        assert result["diagnostics"]["no_uniform_fallback"]
        assert result["diagnostics"]["no_post_reconciliation"]
        assert result["diagnostics"]["coarsefine_seam_edge_count"] == 0
        triangles = separator.vertices[separator.edges[separator.joint_face_bank_ids == bank]]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        gradient = np.asarray(inspected[0]["diagnostics"]["plane"][:3])
        assert np.all((normals @ gradient) * (-1 if bank == 0 else 1) > 0)
        for mesh in solution.dc_meshes[1:]:
            values, _ = inspected[0]["query"](mesh.vertices[mesh.joint_bank_ids == bank], bank=bank)
            assert ((-1 if bank == 0 else 1) * (values[0] - metadata[0][2])).max() < 1e-10
    assert len(inputs.surface_points.sp_coords) == 102 and len(inputs.orientations.dip_positions) == 8
    np.testing.assert_array_equal(inputs.surface_points.sp_coords, original.surface_points.sp_coords)
    np.testing.assert_array_equal(inputs.orientations.dip_positions, original.orientations.dip_positions)
    np.testing.assert_array_equal(inputs.orientations.dip_gradients, original.orientations.dip_gradients)
    np.testing.assert_array_equal(inputs.unit_values, original.unit_values)
    assert inputs.grid.octree_grid is root
    np.testing.assert_array_equal(root.orthogonal_extent, extent)


def test_actual_model7_natural_root7_rejects_before_any_bank_emission(monkeypatch):
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.core.data.regular_grid import RegularGrid
    from tests.fixtures.model7_combination import model7_combination_factory
    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=True)
    inputs.grid.octree_grid = RegularGrid(inputs.grid.octree_grid.orthogonal_extent, [7, 7, 7])
    options.evaluation_options.octree_min_level = 0
    options.evaluation_options.mesh_extraction_overlap = "joint"
    inspected = _audit_fixed_queries(monkeypatch)
    core = importlib.import_module("gempy_engine.API.dual_contouring.joint_topology")
    extract = core.extract_adaptive_topology
    prepared = []

    def prepare(*args, **kwargs):
        assert kwargs["defer_emission"] is True
        result = extract(*args, **kwargs)
        assert result.get("faces") is None and result.get("affected_faces") is None
        assert "_triangle_plan" in result and "_key_to_id" in result
        prepared.append(result)
        return result

    emit = Mock(side_effect=AssertionError("unsupported natural hierarchy reached any bank emission"))
    monkeypatch.setattr(bridge, "extract_adaptive_topology", prepare)
    monkeypatch.setattr(core, "emit_adaptive_triangles", emit)
    monkeypatch.setattr(bridge, "emit_adaptive_triangles", emit)
    with pytest.raises(ValueError, match="unsupported_hanging_branch"):
        compute_model(inputs, options, descriptor)
    assert len(inspected) == 1 and len(prepared) == 1 and prepared[0]["bank"] == 0
    assert prepared[0].get("faces") is None and prepared[0].get("affected_faces") is None
    emit.assert_not_called()


def test_model7_four_original_exports_with_core_contract_stub(monkeypatch):
    """Audit exact assembly/export metadata independently of geometric extraction."""
    from gempy_engine.API.model.model_api import compute_model
    from gempy_engine.core.data.regular_grid import RegularGrid
    from tests.fixtures.model7_combination import model7_combination_factory
    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=True)
    inputs.grid.octree_grid = RegularGrid(inputs.grid.octree_grid.orthogonal_extent, [7, 7, 7])
    options.evaluation_options.mesh_extraction_overlap = "joint"
    inspected = _audit_fixed_queries(monkeypatch)
    core = importlib.import_module("gempy_engine.API.dual_contouring.joint_topology")
    calls = []

    def extract(origins, spans, shape, extent, samples, groups, indices, relations, isovalues, **kwargs):
        calls.append(kwargs)
        assert kwargs["defer_emission"] is True and kwargs["include_reference"] is False
        np.testing.assert_array_equal(groups, [0, 1, 2, 2])
        np.testing.assert_array_equal(indices, [0, 0, 0, 1])
        assert relations == [R.FAULT, R.ERODE, R.BASEMENT]
        assert [len(v) for v in isovalues] == [1, 1, 2]
        assert kwargs["separator_contacts"]["fault_pairs"] == {(0, 1), (0, 2), (0, 3)}
        assert kwargs["separator_contacts"]["controllers"] == {
            0: [], 1: [(0, -1)], 2: [(1, -1), (0, -1)], 3: [(1, -1), (0, -1)]}
        keys = [("regular", i) for i in range(3)]
        return dict(vertices=np.eye(3), _triangle_plan=[[dict(keys=tuple(keys))] for _ in groups],
                    _key_to_id=dict(zip(keys, range(3))), vertex_keys=keys, seam_edges=[[0, 1]],
                    diagnostics={"coarsefine_seam_edge_count": 0})

    monkeypatch.setattr(bridge, "extract_adaptive_topology", extract)
    solution = compute_model(inputs, options, descriptor)
    assert len(inspected) == 1 and len(calls) == 2
    assert len(solution.dc_meshes) == 4
    assert [(m.stack_index, m.surface_index, m.exported_surface_index) for m in solution.dc_meshes] == [
        (0, 0, 0), (1, 0, 1), (2, 0, 2), (2, 1, 3)]
    original = adapter._collect_joint_leaves(solution.octrees_output, 3)[-1]
    assert [m.isovalue for m in solution.dc_meshes] == [row[2] for row in original]
    for mesh in solution.dc_meshes:
        assert len(mesh.edges) == 2
        np.testing.assert_array_equal(mesh.joint_face_bank_ids, [0, 1])
        assert set(mesh.joint_bank_ids) == {0, 1}
    np.testing.assert_array_equal(inputs.unit_values, np.arange(1, 6))


@pytest.mark.parametrize("existing", [False, True], ids=["empty", "existing_buffers"])
@pytest.mark.parametrize("nonplanar", [False, True], ids=["successful_model7", "rejected_nonplanar"])
def test_joint_model7_preserves_caller_fault_metadata(monkeypatch, existing, nonplanar):
    from gempy_engine.core.data.kernel_classes.faults import FaultsData
    from gempy_engine.core.data.regular_grid import RegularGrid
    from tests.fixtures.model7_combination import model7_combination_factory
    api = importlib.import_module("gempy_engine.API.model.model_api")
    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=True)
    root = RegularGrid(inputs.grid.octree_grid.orthogonal_extent, [7, 7, 7])
    inputs.grid.octree_grid = root
    options.evaluation_options.octree_min_level = 0
    options.evaluation_options.mesh_extraction_overlap = "joint"
    _controlled_refinement(monkeypatch, root, False)
    # Exercise the explicit no-copy configuration as well as the NumPy default.
    monkeypatch.setattr(api, "NOT_MAKE_INPUT_DEEP_COPY", existing)
    fields = ("fault_values_everywhere", "fault_values_on_sp", "fault_values_ref", "fault_values_rest")
    data = FaultsData(**({name: np.array([[i + .125, i + .625]]) for i, name in enumerate(fields)}
                        if existing else {}))
    faults = [None, data, None]
    descriptor.stack_structure.faults_input_data = faults
    cursor = descriptor.stack_structure.stack_number
    references = {name: getattr(data, name) for name in fields}
    before = copy.deepcopy(data)
    if nonplanar:
        inputs.surface_points.sp_coords[0, 0] += .01
    points = inputs.surface_points.sp_coords.copy()
    orientations = inputs.orientations.dip_positions.copy()
    gradients = inputs.orientations.dip_gradients.copy()
    if nonplanar:
        with pytest.raises(ValueError, match="unsupported_nonplanar_fault"):
            api.compute_model(inputs, options, descriptor)
    else:
        solution = api.compute_model(inputs, options, descriptor)
        assert len(solution.dc_meshes) == 4
        assert [len(mesh.edges) for mesh in solution.dc_meshes] == [884, 312, 832, 1060]
    assert descriptor.stack_structure.faults_input_data is faults
    assert faults[1] is data
    assert descriptor.stack_structure.stack_number == cursor
    for name in fields:
        assert getattr(data, name) is references[name]
        np.testing.assert_array_equal(getattr(data, name), getattr(before, name))
    np.testing.assert_array_equal(inputs.surface_points.sp_coords, points)
    np.testing.assert_array_equal(inputs.orientations.dip_positions, orientations)
    np.testing.assert_array_equal(inputs.orientations.dip_gradients, gradients)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_joint_failed_compute_isolates_root_and_descriptor_fault_buffers(monkeypatch, backend):
    from gempy_engine.config import AvailableBackends
    from gempy_engine.core.backend_tensor import BackendTensor as BT
    from gempy_engine.core.data.kernel_classes.faults import FaultsData
    from tests.fixtures.model7_combination import model7_combination_factory
    api = importlib.import_module("gempy_engine.API.model.model_api")
    if backend == "torch":
        torch = pytest.importorskip("torch")
        BT._change_backend(AvailableBackends.PYTORCH, use_gpu=False, use_pykeops=False, dtype="float64", grads=False)
    inputs, options, descriptor = model7_combination_factory(number_octree_levels=2, mesh_extraction=True)
    options.evaluation_options.mesh_extraction_overlap = "joint"
    monkeypatch.setattr(api, "NOT_MAKE_INPUT_DEEP_COPY", True)
    fields = ("fault_values_everywhere", "fault_values_on_sp", "fault_values_ref", "fault_values_rest")
    data = FaultsData(**{name: BT.t.array([[i + .125, i + .625]], dtype=BT.dtype_obj)
                         for i, name in enumerate(fields)})
    inputs._fault_values = data
    descriptor.stack_structure.faults_input_data = [None, data, None]
    before = {name: adapter._numpy(getattr(data, name)).copy() for name in fields}
    references = {name: getattr(data, name) for name in fields}
    caller_points = inputs.surface_points.sp_coords
    if backend == "torch":
        caller_points.requires_grad_(True)

    def fail(interpolation_input, options, data_descriptor):
        assert interpolation_input is not inputs
        assert interpolation_input.surface_points.sp_coords is caller_points
        if backend == "torch":
            assert torch.is_grad_enabled() and caller_points.requires_grad
        assert data_descriptor.stack_structure is not descriptor.stack_structure
        for private in (interpolation_input._fault_values, data_descriptor.stack_structure.faults_input_data[1]):
            assert private is not data
            for name in fields:
                assert getattr(private, name) is not references[name]
                getattr(private, name)[...] = -999.
        data_descriptor.stack_structure.stack_number = 2
        raise ValueError("unsupported_nonplanar_fault: failed interpolation isolation probe")

    monkeypatch.setattr(api, "interpolate_n_octree_levels", fail)
    grad_context = torch.enable_grad() if backend == "torch" else nullcontext()
    previous_grad_enabled = torch.is_grad_enabled() if backend == "torch" else None
    with grad_context:
        with pytest.raises(ValueError, match="unsupported_nonplanar_fault"):
            api.compute_model(inputs, options, descriptor)
        if backend == "torch":
            assert torch.is_grad_enabled() and caller_points.requires_grad
    assert inputs._fault_values is data
    assert descriptor.stack_structure.faults_input_data[1] is data
    assert descriptor.stack_structure.stack_number == -1
    for name in fields:
        assert getattr(data, name) is references[name]
        np.testing.assert_array_equal(adapter._numpy(getattr(data, name)), before[name])
    if backend == "torch":
        assert torch.is_grad_enabled() == previous_grad_enabled and caller_points.requires_grad
