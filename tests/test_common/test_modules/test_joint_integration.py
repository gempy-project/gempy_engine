"""Production joint adapter tests; no edits to the independent topology core."""

import importlib
import os
import subprocess
import sys
from collections import Counter
from itertools import product
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gempy_engine.config import AvailableBackends, DualContouringOverlap
from gempy_engine.core.backend_tensor import BackendTensor as BT
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.options.evaluation_options import MeshExtentCapping, MeshExtractionMaskingOptions, TriangulationMethod
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from gempy_engine.modules.octrees_topology._octree_common import _generate_next_level_centers

adapter = importlib.import_module("gempy_engine.API.dual_contouring.joint_extraction")
dispatch = importlib.import_module("gempy_engine.API.dual_contouring.multi_scalar_dual_contouring")


@pytest.fixture(autouse=True)
def cpu_float64():
    saved = dict(engine_backend=BT.engine_backend, use_gpu=BT.use_gpu,
                 use_pykeops=BT.use_pykeops, dtype=BT.dtype, grads=BT.COMPUTE_GRADS)
    BT._change_backend(AvailableBackends.numpy, use_gpu=False, use_pykeops=False, dtype="float64", grads=False)
    yield
    BT._change_backend(**saved)


def _case():
    root = RegularGrid([0, 1, 0, 1, 0, 1], [2, 2, 2])
    refined = np.zeros(8, dtype=bool)
    refined[0] = True
    xyz, offsets = _generate_next_level_centers(root.values[refined], root.dxdydz)
    child = RegularGrid.from_octree_level(xyz, root, refined, offsets)
    levels = []
    for grid in (root, child):
        corners = grid.corners_values
        engine = EngineGrid(octree_grid=grid)
        output_grid = SimpleNamespace(corners_grid_slice=slice(None))
        output = SimpleNamespace(
            grid=output_grid, scalar_field_at_sp=np.array([.3, .7]), weights=None,
            exported_fields=SimpleNamespace(scalar_field=corners[:, 2]),
        )
        levels.append(SimpleNamespace(grid=engine, outputs=[output]))
    stacks = StacksStructure(np.array([0]), np.array([0]), np.array([2]),
                            [StackRelationType.BASEMENT], interp_functions_per_stack=[object()])
    descriptor = InputDataDescriptor(TensorsStructure(np.array([], dtype=int)), stacks)
    inputs = InterpolationInput(None, None, EngineGrid(octree_grid=root))
    options = InterpolationOptions.from_args(range=1, c_o=1)
    return descriptor, inputs, options, levels


@pytest.mark.parametrize("mode", ["none", "pretty", "joint"])
def test_joint_environment_and_flag_values(mode):
    env = dict(os.environ, DUAL_CONTOURING_VERTEX_OVERLAP=mode)
    script = (
        "from gempy_engine.config import DualContouringOverlap as D, DUAL_CONTOURING_VERTEX_OVERLAP as mode; "
        "assert [D.none.value,D.pretty.value,D.watertight.value,D.joint.value,D.joint_contacts.value] == [1,2,4,8,16]; "
        f"assert mode is D.{mode}"
    )
    subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True)


@pytest.mark.parametrize("value", ["joint", DualContouringOverlap.joint])
def test_per_model_options_accept_enum_or_string(value):
    from gempy_engine.core.data.options.evaluation_options import EvaluationOptions
    options = InterpolationOptions.from_args(range=1, c_o=1)
    options.evaluation_options = EvaluationOptions(mesh_extraction_overlap=value)
    rebuilt = InterpolationOptions.model_validate(options.model_dump())
    assert rebuilt.evaluation_options.mesh_extraction_overlap == value


@pytest.mark.parametrize("mode", ["joint", DualContouringOverlap.joint])
def test_per_model_joint_dispatch_precedes_legacy(monkeypatch, mode):
    descriptor, inputs, options, levels = _case()
    options.evaluation_options.mesh_extraction_overlap = mode
    monkeypatch.setattr(dispatch, "DUAL_CONTOURING_VERTEX_OVERLAP", DualContouringOverlap.pretty)
    legacy = Mock(side_effect=AssertionError("legacy triangulation called"))
    monkeypatch.setattr(dispatch, "get_triangulation_codes", legacy)
    extract = Mock(return_value=["joint"])
    monkeypatch.setattr(adapter, "extract_joint_octree", extract)
    assert dispatch.dual_contouring_multi_scalar(descriptor, inputs, options, levels) == ["joint"]
    extract.assert_called_once_with(descriptor, inputs, options, levels)
    legacy.assert_not_called()


@pytest.mark.parametrize("value", ["unknown", "joint|pretty", 16,
    DualContouringOverlap.joint | DualContouringOverlap.none,
    DualContouringOverlap.joint | DualContouringOverlap.joint_contacts,
    DualContouringOverlap.joint_contacts | DualContouringOverlap.pretty])
def test_invalid_modes_before_mutations(value):
    descriptor, inputs, options, levels = _case()
    options.evaluation_options.mesh_extraction_overlap = value
    original_grid, cursor = inputs.grid, descriptor.stack_structure.stack_number
    with pytest.raises(ValueError):
        dispatch.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    assert inputs.grid is original_grid
    assert descriptor.stack_structure.stack_number == cursor


@pytest.mark.parametrize("value", [DualContouringOverlap(i) for i in range(8)])
def test_all_legacy_bitfields_and_explicit_override(monkeypatch, value):
    descriptor, inputs, options, levels = _case()
    options.evaluation_options.mesh_extraction_overlap = value
    monkeypatch.setattr(dispatch, "DUAL_CONTOURING_VERTEX_OVERLAP", DualContouringOverlap.joint)
    legacy = Mock(side_effect=RuntimeError("reached legacy"))
    monkeypatch.setattr(dispatch, "get_triangulation_codes", legacy)
    with pytest.raises(RuntimeError, match="reached legacy"):
        dispatch.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    legacy.assert_called_once()


def test_mixed_depth_leaves_and_offset_coordinates():
    descriptor, _, _, levels = _case()
    levels[1].grid.octree_grid._integer_coordinates -= 1
    origins, spans, shape, extent, samples, metadata = adapter._collect_joint_leaves(levels, 1)
    assert origins.shape == (15, 3)
    np.testing.assert_array_equal(spans, [2] * 7 + [1] * 8)
    np.testing.assert_array_equal(shape, [4, 4, 4])
    assert np.sum(spans ** 3) == np.prod(shape)
    assert samples.shape == (2, 15, 8)
    np.testing.assert_array_equal(samples[0], samples[1])
    assert metadata == [(0, 0, .3), (0, 1, .7)]
    assert np.all(origins >= 0)


def test_inconsistent_metadata_rejected():
    _, _, _, levels = _case()
    levels[0].outputs[0].scalar_field_at_sp[0] += .1
    with pytest.raises(ValueError, match="metadata/isovalues"):
        adapter._collect_joint_leaves(levels, 1)


def test_large_offset_isovalue_difference_rejected():
    _, _, _, levels = _case()
    for level in levels:
        level.outputs[0].scalar_field_at_sp[:] = [1e12, 1e12 + 1]
    levels[1].outputs[0].scalar_field_at_sp[0] += .1
    with pytest.raises(ValueError, match="metadata/isovalues"):
        adapter._collect_joint_leaves(levels, 1)


@pytest.mark.parametrize("guard", ["cap", "mask", "null", "float32"])  # CUDA is supported (test_joint_contacts)
def test_joint_guards(monkeypatch, guard):
    descriptor, inputs, options, levels = _case()
    if guard == "cap":
        options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.SCALAR_LESS_EQUAL
    elif guard == "mask":
        options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW
    elif guard == "null":
        descriptor.stack_structure.masking_descriptor[0] = StackRelationType.NULL_SPACE
    else:
        monkeypatch.setattr(BT, "dtype", "float32")
    original_grid = inputs.grid
    with pytest.raises((ValueError, NotImplementedError)):
        adapter.extract_joint_octree(descriptor, inputs, options, levels)
    assert inputs.grid is original_grid
    assert descriptor.stack_structure.stack_number == -1


def test_joint_fault_dispatch_uses_supported_bridge(monkeypatch):
    descriptor, inputs, options, levels = _case()
    descriptor.stack_structure.masking_descriptor[0] = StackRelationType.FAULT
    bridge = importlib.import_module("gempy_engine.API.dual_contouring.joint_fault_banks")
    extract = Mock(return_value=["supported_fault_bridge"])
    monkeypatch.setattr(bridge, "extract_fault_joint_octree", extract)
    assert adapter.extract_joint_octree(descriptor, inputs, options, levels) == ["supported_fault_bridge"]
    extract.assert_called_once_with(descriptor, inputs, options, levels)


@pytest.mark.parametrize("fail_query", [False, True])
def test_adapter_callback_isolation_and_shared_ids(monkeypatch, fail_query):
    descriptor, inputs, options, levels = _case()
    queried = []

    def interpolate(query_input, query_options, query_descriptor):
        queried.append(query_input)
        assert query_input is not inputs
        assert query_input.weights is not inputs.weights
        assert query_options is not options
        assert query_options.evaluation_options.compute_scalar_gradient
        query_descriptor.stack_structure.stack_number = 0
        if fail_query:
            raise RuntimeError("query failed")
        points = query_input.grid.custom_grid.values
        return [SimpleNamespace(grid=SimpleNamespace(custom_grid_slice=slice(None)),
            exported_fields=SimpleNamespace(scalar_field=points[:, 2],
                gx_field=np.zeros(len(points)), gy_field=np.zeros(len(points)), gz_field=np.ones(len(points))))]

    def extract(origins, spans, domain_shape, extent, samples, surface_to_stack,
                surface_indices, stack_relations, isovalues, *, sample_fields, ownership, include_reference):
        assert len(origins) == 15 and samples.shape == (2, 15, 8)
        values, gradients = sample_fields(np.array([[.1, .2, .3], [.4, .5, .6]]))
        np.testing.assert_array_equal(values, [[.3, .6], [.3, .6]])
        np.testing.assert_array_equal(gradients, np.tile([0., 0., 1.], (2, 2, 1)))
        return dict(vertices=np.array(list(product((0., 1.), repeat=3))),
                    faces=[[[0, 1, 2]], [[0, 2, 3]]], vertex_keys=list(range(8)),
                    diagnostics={"joint": True}, seam_edges=np.array([[0, 2]]))

    monkeypatch.setattr(adapter, "extract_adaptive_topology", extract)
    monkeypatch.setattr(adapter, "interpolate_all_fields_no_octree", interpolate)
    original_grid = inputs.grid
    if fail_query:
        with pytest.raises(RuntimeError, match="query failed"):
            adapter.extract_joint_octree(descriptor, inputs, options, levels)
        assert inputs.grid is original_grid and queried[0].grid is original_grid
        assert descriptor.stack_structure.stack_number == -1
        assert options.evaluation_options.compute_scalar_gradient is False
        return
    meshes = adapter.extract_joint_octree(descriptor, inputs, options, levels)
    assert inputs.grid is original_grid and queried[0].grid is original_grid
    assert descriptor.stack_structure.stack_number == -1
    assert options.evaluation_options.compute_scalar_gradient is False
    np.testing.assert_array_equal(meshes[0].joint_vertex_ids, [0, 1, 2])
    np.testing.assert_array_equal(meshes[1].joint_vertex_ids, [0, 2, 3])
    np.testing.assert_array_equal(meshes[0].vertices[[0, 2]], meshes[1].vertices[[0, 1]])
    assert all(mesh.contact_report == {"joint": True} for mesh in meshes)


@pytest.mark.parametrize("unconformity", [False, True])
@pytest.mark.parametrize("real_core", [False, True])
def test_production_compute_plane_and_unconformity(monkeypatch, unconformity, real_core):
    from gempy_engine.API.model.model_api import compute_model

    if real_core:
        pytest.importorskip("gempy_engine.API.dual_contouring.joint_topology")
    normals = [np.array([0., 0., 1.])]
    isovalues = [.37]
    if unconformity:
        normals.append(np.array([1., 0., .2]))
        isovalues.append(.43)
    functions = [CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([iso]),
        implicit_function=lambda xyz, normal=normal: xyz @ normal,
        gx_function=lambda xyz, normal=normal: np.full(len(xyz), normal[0]),
        gy_function=lambda xyz, normal=normal: np.full(len(xyz), normal[1]),
        gz_function=lambda xyz, normal=normal: np.full(len(xyz), normal[2]),
    ) for normal, iso in zip(normals, isovalues)]
    n = len(normals)
    stacks = StacksStructure(np.zeros(n, dtype=int), np.zeros(n, dtype=int),
        np.ones(n, dtype=int), [StackRelationType.ERODE] * (n - 1) + [StackRelationType.BASEMENT],
        interp_functions_per_stack=functions)
    descriptor = InputDataDescriptor(TensorsStructure(np.array([], dtype=int)), stacks)
    grid = EngineGrid(octree_grid=RegularGrid([0., 1., 0., 1., 0., 1.], [4, 4, 4]))
    inputs = InterpolationInput(SurfacePoints(np.empty((0, 3))),
        Orientations(np.empty((0, 3)), np.empty((0, 3))), grid)
    options = InterpolationOptions.from_args(range=1., c_o=1.)
    options.evaluation_options.number_octree_levels = 2
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.mesh_extraction_overlap = "joint"
    calls = []
    if not real_core:
        def extract(origins, spans, shape, extent, samples, surface_to_stack, surface_indices,
                    relations, levels, *, sample_fields, ownership, include_reference):
            points = np.array([[.12, .23, .34], [.45, .56, .67]])
            values, gradients = sample_fields(points)
            calls.append(points)
            np.testing.assert_allclose(values, np.stack([points @ normal for normal in normals]))
            np.testing.assert_allclose(gradients, np.stack([np.tile(normal, (2, 1)) for normal in normals]))
            assert np.sum(spans ** 3) == np.prod(shape)
            return dict(vertices=np.array([[0., 0., .37], [1., 0., .37], [0., 1., .37]]),
                        faces=[np.array([[0, 1, 2]]) for _ in normals], vertex_keys=[0, 1, 2],
                        diagnostics={}, seam_edges=np.empty((0, 2), dtype=int))
        monkeypatch.setattr(adapter, "extract_adaptive_topology", extract)
    solution = compute_model(inputs, options, descriptor)
    assert len(solution.dc_meshes) == n
    assert inputs.grid is grid
    for mesh in solution.dc_meshes:
        assert np.isfinite(mesh.vertices).all()
        assert len(mesh.edges) > 0
        assert len(mesh.joint_vertex_ids) == len(mesh.vertices)
        assert np.all(mesh.edges < len(mesh.vertices))
    if not real_core:
        assert calls


@pytest.mark.parametrize("invalid_fingerprint", [False, True])
@pytest.mark.parametrize("backend", [AvailableBackends.numpy, AvailableBackends.PYTORCH])
def test_production_cokriging_plane_uses_fixed_weights(monkeypatch, invalid_fingerprint, backend):
    model_api = importlib.import_module("gempy_engine.API.model.model_api")
    scalar_api = importlib.import_module("gempy_engine.API.interp_single._interp_scalar_field")
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache

    pytest.importorskip("gempy_engine.API.dual_contouring.joint_topology")
    if backend is AvailableBackends.PYTORCH:
        pytest.importorskip("torch")
    BT._change_backend(backend, use_gpu=False, use_pykeops=False, dtype="float64", grads=False)
    stacks = StacksStructure(np.array([3]), np.array([1]), np.array([1]), [StackRelationType.BASEMENT])
    descriptor = InputDataDescriptor(TensorsStructure(np.array([3])), stacks)
    grid = EngineGrid(octree_grid=RegularGrid([0., 1., 0., 1., 0., 1.], [4, 4, 4]))
    inputs = InterpolationInput(SurfacePoints(np.array([[.1, .1, .37], [.9, .1, .37], [.1, .9, .37]])),
        Orientations(np.array([[.5, .5, .37]]), np.array([[0., 0., 1.]])), grid)
    options = InterpolationOptions.from_args(range=2., c_o=1.)
    options.evaluation_options.number_octree_levels = 2
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.mesh_extraction_overlap = "joint"
    production_evaluate = adapter._evaluate_sys_eq
    calls = []
    solver = Mock(side_effect=AssertionError("joint query called solver"))
    production_dispatch = model_api.dual_contouring_multi_scalar

    def extract(**kwargs):
        monkeypatch.setattr(scalar_api, "_solve_interpolation_result", solver)
        if invalid_fingerprint:
            WeightCache.memory_cache[f"{kwargs['options'].cache_model_name}.0"]["hash"] = "overwritten"
        return production_dispatch(**kwargs)

    def evaluate(solver_input, weights, query_options, grid):
        cached = WeightCache.load_weights(f"{query_options.cache_model_name}.0", False)
        assert weights is not cached["weights"]
        assert not np.shares_memory(adapter._numpy(weights), adapter._numpy(cached["weights"]))
        if isinstance(weights, np.ndarray):
            assert not weights.flags.writeable
        else:
            assert not weights.requires_grad and weights.grad_fn is None
        if calls:
            np.testing.assert_array_equal(weights, calls[0])
        calls.append(adapter._numpy(weights).copy())
        fields = production_evaluate(solver_input, weights, query_options, grid=grid)
        # A later callback must never consult this now-corrupted shared cache.
        cached["weights"][:] = 99
        cached["hash"] = "changed between callbacks"
        return fields

    monkeypatch.setattr(model_api, "dual_contouring_multi_scalar", extract)
    monkeypatch.setattr(adapter, "_evaluate_sys_eq", evaluate)
    monkeypatch.setattr(adapter, "interpolate_all_fields_no_octree",
                        Mock(side_effect=AssertionError("cokriging used cache-resolving orchestration")))
    if invalid_fingerprint:
        with pytest.raises(ValueError, match="fingerprint mismatch.*no solve allowed"):
            model_api.compute_model(inputs, options, descriptor)
        solver.assert_not_called()
        assert not calls
        return
    solution = model_api.compute_model(inputs, options, descriptor)
    solver.assert_not_called()
    assert len(calls) > 1
    assert calls and len(solution.dc_meshes) == 1
    mesh = solution.dc_meshes[0]
    assert len(mesh.edges) > 0
    # Production Torch QEF regularization introduces a small plane displacement.
    np.testing.assert_allclose(mesh.vertices[:, 2], .37, atol=1e-5)
    assert inputs.grid is grid


@pytest.mark.parametrize("selection", ["explicit", "environment", "invalid_flags", "invalid_environment", "unknown"])
def test_public_compute_torch_joint_cursor_isolation(monkeypatch, selection):
    from gempy_engine.API.model.model_api import compute_model

    pytest.importorskip("torch")
    BT._change_backend(AvailableBackends.PYTORCH, use_gpu=False, use_pykeops=False, dtype="float64", grads=False)
    descriptor, inputs, options, _ = _case()
    functions = CustomInterpolationFunctions(
        np.array([.37]), lambda xyz: xyz[:, 2],
        lambda xyz: BT.t.zeros(len(xyz), dtype=BT.dtype_obj),
        lambda xyz: BT.t.zeros(len(xyz), dtype=BT.dtype_obj),
        lambda xyz: BT.t.ones(len(xyz), dtype=BT.dtype_obj))
    descriptor.stack_structure.number_of_surfaces_per_stack = np.array([1])
    descriptor.stack_structure.interp_functions_per_stack = [functions]
    inputs.surface_points = SurfacePoints(np.empty((0, 3)))
    inputs.orientations = Orientations(np.empty((0, 3)), np.empty((0, 3)))
    options.evaluation_options.number_octree_levels = 2
    options.evaluation_options.number_octree_levels_surface = 2
    options.evaluation_options.mesh_extraction_overlap = {
        "explicit": "joint", "environment": None,
        "invalid_environment": None,
        "invalid_flags": DualContouringOverlap.joint | DualContouringOverlap.pretty,
        "unknown": "unknown",
    }[selection]
    monkeypatch.setattr(dispatch, "DUAL_CONTOURING_VERTEX_OVERLAP",
                        DualContouringOverlap.joint | DualContouringOverlap.pretty if selection == "invalid_environment" else
                        DualContouringOverlap.joint if selection == "environment" else DualContouringOverlap.none)
    original_cursor = descriptor.stack_structure.stack_number
    original_grid = inputs.grid
    if selection in ("invalid_flags", "invalid_environment", "unknown"):
        with pytest.raises(ValueError):
            compute_model(inputs, options, descriptor)
    else:
        solution = compute_model(inputs, options, descriptor)
        assert len(solution.dc_meshes) == 1 and len(solution.dc_meshes[0].edges)
    assert descriptor.stack_structure.stack_number == original_cursor
    assert descriptor.stack_structure.faults_input_data is None
    assert inputs.grid is original_grid


@pytest.mark.parametrize("backend", [AvailableBackends.numpy, AvailableBackends.PYTORCH])
def test_public_compute_joint_mixed_leaf_seam(monkeypatch, backend):
    from gempy_engine.API.model.model_api import compute_model

    if backend is AvailableBackends.PYTORCH:
        pytest.importorskip("torch")
    BT._change_backend(backend, use_gpu=False, use_pykeops=False, dtype="float64", grads=False)
    normals = [np.array([0., 0., 1.]), np.array([1., 0., .2])]
    functions = [CustomInterpolationFunctions(
        np.array([iso]),
        lambda xyz, normal=normal: xyz @ BT.t.array(normal, dtype=BT.dtype_obj),
        lambda xyz, normal=normal: BT.t.ones(len(xyz), dtype=BT.dtype_obj) * normal[0],
        lambda xyz, normal=normal: BT.t.ones(len(xyz), dtype=BT.dtype_obj) * normal[1],
        lambda xyz, normal=normal: BT.t.ones(len(xyz), dtype=BT.dtype_obj) * normal[2],
    ) for normal, iso in zip(normals, [.37, .43])]
    stacks = StacksStructure(np.zeros(2, dtype=int), np.zeros(2, dtype=int), np.ones(2, dtype=int),
        [StackRelationType.ERODE, StackRelationType.BASEMENT], interp_functions_per_stack=functions)
    descriptor = InputDataDescriptor(TensorsStructure(np.array([], dtype=int)), stacks)
    root = RegularGrid([0., 1., 0., 1., 0., 1.], [2, 2, 2])
    grid = EngineGrid(octree_grid=root)
    inputs = InterpolationInput(SurfacePoints(np.empty((0, 3))),
        Orientations(np.empty((0, 3)), np.empty((0, 3))), grid)
    options = InterpolationOptions.from_args(range=1., c_o=1.)
    options.evaluation_options.number_octree_levels = 3
    options.evaluation_options.number_octree_levels_surface = 3
    options.evaluation_options.octree_min_level = 0
    options.evaluation_options.mesh_extraction_overlap = "joint"

    # Control selection only: production still generates children and samples every
    # level. Refine all root cells, then y >= .5, making a genuine span-2/span-1 seam.
    internals = importlib.import_module("gempy_engine.modules.octrees_topology._octree_internals")
    mark_voxel = internals._mark_voxel
    child_centers, _ = _generate_next_level_centers(root.values, root.dxdydz)
    selections = []

    def controlled_selection(corner_ids):
        shifts, _ = mark_voxel(corner_ids)
        if not selections:
            selected = BT.t.array(np.ones(len(root.values), dtype=bool), dtype="bool")
        else:
            assert len(selections) == 1 and len(corner_ids) == len(child_centers)
            selected = child_centers[:, 1] >= .5
        selections.append(adapter._numpy(selected).copy())
        return shifts, selected

    monkeypatch.setattr(internals, "_mark_voxel", controlled_selection)
    solution = compute_model(inputs, options, descriptor)
    assert len(selections) == 2 and selections[0].all()
    assert selections[1].sum() == len(selections[1]) // 2
    levels = solution.octrees_output
    np.testing.assert_array_equal(adapter._numpy(levels[1].grid.octree_grid.active_cells), selections[0])
    np.testing.assert_array_equal(adapter._numpy(levels[2].grid.octree_grid.active_cells), selections[1])
    origins, spans, shape, _, samples, _ = adapter._collect_joint_leaves(levels, 2)
    assert set(spans) == {1, 2} and len(origins) == 32 + 256
    assert np.sum(spans ** 3) == np.prod(shape)
    assert samples.shape == (2, len(origins), 8)

    controller, target = solution.dc_meshes
    for mesh in (controller, target):
        assert mesh.contact_report["coarsefine_seam_edge_count"] > 0
        assert mesh.contact_report["no_post_reconciliation"]
        assert mesh.contact_report["no_uniform_fallback"]
    np.testing.assert_array_equal(controller.joint_seam_edges, target.joint_seam_edges)
    keys = dict(zip(controller.joint_vertex_ids, controller.joint_vertex_keys))
    shared = np.intersect1d(controller.joint_vertex_ids, target.joint_vertex_ids)
    coarsefine_edges = [tuple(sorted(map(int, edge))) for edge in controller.joint_seam_edges
                       if keys[edge[0]][-2] != keys[edge[1]][-2]]
    assert coarsefine_edges
    assert len(coarsefine_edges) == controller.contact_report["coarsefine_seam_edge_count"]
    counts = []
    for mesh in (controller, target):
        global_faces = mesh.joint_vertex_ids[mesh.edges]
        counts.append(Counter(tuple(sorted((int(a), int(b)))) for face in global_faces
                              for a, b in zip(face, np.roll(face, -1))))
    positions = [dict(zip(mesh.joint_vertex_ids, mesh.vertices)) for mesh in (controller, target)]
    for edge in coarsefine_edges:
        assert {keys[vertex][-2] for vertex in edge} == {1, 2}
        assert all(keys[vertex][0] == "joint" and vertex in shared for vertex in edge)
        assert counts[0][edge] == 2 and counts[1][edge] == 1
        for vertex in edge:
            np.testing.assert_array_equal(positions[0][vertex], positions[1][vertex])
            np.testing.assert_allclose(positions[0][vertex][[0, 2]], [.43 - .2 * .37, .37], atol=1e-12)
    assert descriptor.stack_structure.stack_number == -1 and inputs.grid is grid


@pytest.mark.parametrize("backend", [AvailableBackends.numpy, AvailableBackends.PYTORCH])
@pytest.mark.parametrize("triangulation", [TriangulationMethod.LEGACY, TriangulationMethod.QUADS])
@pytest.mark.parametrize("normal,isovalue", [
    ([1., 0., 0.], .37), ([-1., 0., 0.], -.37),
    ([0., 1., 0.], .37), ([0., -1., 0.], -.37),
    ([0., 0., 1.], .37), ([0., 0., -1.], -.37),
    ([3., 5., 7.], 6.13),
], ids=["x-positive", "x-negative", "y-positive", "y-negative", "z-positive", "z-negative", "oblique"])
def test_public_compute_none_joint_ordinary_parity(backend, triangulation, normal, isovalue):
    from gempy_engine.API.model.model_api import compute_model

    if backend is AvailableBackends.PYTORCH:
        pytest.importorskip("torch")
    BT._change_backend(backend, use_gpu=False, use_pykeops=False, dtype="float64", grads=False)

    def oriented_triangle(points):
        vertices = tuple(map(tuple, points))
        # Cyclic rotation preserves winding; reversal must remain distinguishable.
        return min(vertices[i:] + vertices[:i] for i in range(3))

    geometries = []
    for mode in ("none", "joint"):
        functions = CustomInterpolationFunctions(
            np.array([isovalue]),
            lambda xyz: xyz @ BT.t.array(normal, dtype=BT.dtype_obj),
            lambda xyz: BT.t.ones(len(xyz), dtype=BT.dtype_obj) * normal[0],
            lambda xyz: BT.t.ones(len(xyz), dtype=BT.dtype_obj) * normal[1],
            lambda xyz: BT.t.ones(len(xyz), dtype=BT.dtype_obj) * normal[2])
        stacks = StacksStructure(np.array([0]), np.array([0]), np.array([1]),
            [StackRelationType.BASEMENT], interp_functions_per_stack=[functions])
        descriptor = InputDataDescriptor(TensorsStructure(np.array([], dtype=int)), stacks)
        grid = EngineGrid(octree_grid=RegularGrid([0., 1., 0., 1., 0., 1.], [2, 2, 2]))
        inputs = InterpolationInput(SurfacePoints(np.empty((0, 3))),
            Orientations(np.empty((0, 3)), np.empty((0, 3))), grid)
        options = InterpolationOptions.from_args(range=1., c_o=1.)
        options.evaluation_options.number_octree_levels = 2
        options.evaluation_options.number_octree_levels_surface = 2
        options.evaluation_options.octree_min_level = 2
        options.evaluation_options.triangulation_method = triangulation
        options.evaluation_options.mesh_extraction_overlap = mode
        options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
        options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.NONE
        solution = compute_model(inputs, options, descriptor)
        assert len(solution.octrees_output) == 2 and len(solution.dc_meshes) == 1
        leaves = solution.octrees_output[-1].grid.octree_grid
        assert len(leaves.values) == 64
        assert adapter._numpy(leaves.active_cells).all()
        np.testing.assert_array_equal(adapter._numpy(leaves.regular_grid_shape), [4, 4, 4])
        mesh = solution.dc_meshes[0]
        assert len(mesh.vertices) and len(mesh.edges) and np.isfinite(mesh.vertices).all()
        geometries.append((Counter(map(tuple, mesh.vertices)),
                           Counter(oriented_triangle(points) for points in mesh.vertices[mesh.edges])))
    assert geometries[0] == geometries[1]
