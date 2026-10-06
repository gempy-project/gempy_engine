"""Contact-aware production extraction with independent affine field oracles."""

import importlib
from itertools import product
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
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.generic_grid import GenericGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.octree_level import OctreeLevel
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.options.evaluation_options import MeshExtractionMaskingOptions
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure
from tests.fixtures.contact_cases import build_contact_case


dc = importlib.import_module("gempy_engine.API.dual_contouring.multi_scalar_dual_contouring")
ROOT = Path(__file__).resolve().parents[3]


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

    def make(name="planar_erosion"):
        case = build_contact_case("planar_erosion" if name == "single" else name, resolution=6)
        if name == "single":
            case.normals, case.levels = case.normals[:1], case.levels[:1]
            case.relations, case.contacts = (StackRelationType.BASEMENT,), ()
        elif name == "parallel_false_overlap":
            # Both planes share cells; the target is wholly on the retained side.
            case.levels[1] = .40
        if name == "planar_erosion":
            # INTERSECT keeps corner-owned cells, but the seam must also lie
            # inside the finite center-to-center target triangle patch.
            case.levels[0] = .39
        root = RegularGrid(np.array([0., 1., 0., 1., 0., 1.]), list(case.resolution))
        offsets = np.array(list(product((0, 1), repeat=3)))
        corners = ((root.integer_coordinates[:, None, :] + offsets) / case.resolution).reshape(-1, 3)
        grid = EngineGrid(octree_grid=root, corners_grid=GenericGrid(corners))
        inputs = InterpolationInput(SurfacePoints(np.empty((0, 3))),
                                    Orientations(np.empty((0, 3)), np.empty((0, 3))), grid)
        n = len(case.normals)
        stacks = StacksStructure(np.zeros(n, dtype=int), np.zeros(n, dtype=int),
                                 np.ones(n, dtype=int), list(case.relations),
                                 faults_relations=np.zeros((n, n), dtype=bool))
        descriptor = InputDataDescriptor(TensorsStructure(np.array([], dtype=int)), stacks)
        options = InterpolationOptions.from_args(10., 1.)
        options.evaluation_options.number_octree_levels = 1
        options.evaluation_options.number_octree_levels_surface = 1
        options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
        xyz = grid.values
        ownership = np.ones((n, len(xyz)), dtype=bool)
        if n == 2:
            controller, truncated, sign = ((1, 0, 1) if case.relations[0] is StackRelationType.ONLAP
                                            else (0, 1, -1))
            ownership[truncated] &= sign * (xyz @ case.normals[controller] - case.levels[controller]) >= 0
        outputs = [SimpleNamespace(
            grid=grid, scalar_field_at_sp=np.array([level]),
            exported_fields=SimpleNamespace(scalar_field=xyz @ normal),
            scalar_fields=SimpleNamespace(stack_relation=relation),
            squeezed_mask_array=mask, mask_components=mask.copy(),
        ) for normal, level, relation, mask in zip(case.normals, case.levels, case.relations, ownership)]

        def gradients(interpolation_input, options, data_descriptor):
            assert options.evaluation_options.compute_scalar_gradient
            assert not options.evaluation_options.compute_scalar
            count = len(interpolation_input.grid.custom_grid.values)
            data_descriptor.stack_structure.stack_number = n - 1
            return [SimpleNamespace(exported_fields=SimpleNamespace(
                gx_field=np.full(count, normal[0]), gy_field=np.full(count, normal[1]),
                gz_field=np.full(count, normal[2]),
            )) for normal in case.normals]

        monkeypatch.setattr(dc, "interpolate_all_fields_no_octree", gradients)
        return case, descriptor, inputs, options, [OctreeLevel(grid, outputs)]

    try:
        yield make
    finally:
        BackendTensor._change_backend(**saved)
        BackendTensor.COMPUTE_GRADS = saved["grads"]
        BackendTensor.pykeops_enabled = saved_keops


@pytest.mark.parametrize("name", ["planar_erosion", "onlap"])
def test_contact_dispatch_clipping_seam_edges_and_tensor_contract(extraction, monkeypatch, name):
    case, descriptor, inputs, options, levels = extraction(name)
    original_grid = inputs.grid
    original_values = original_grid.values.copy()
    original_weights = inputs.weights
    original_cursor = descriptor.stack_structure.stack_number
    original_options = options.model_dump_json()
    original_fields = [o.exported_fields.scalar_field.copy() for o in levels[0].outputs]
    original_masks = [o.squeezed_mask_array.copy() for o in levels[0].outputs]
    snapshots = []
    extract = dc.compute_dual_contouring_v2

    def capture(**kwargs):
        meshes = extract(**dict(kwargs, max_workers=1))
        snapshots.extend((m.vertices, m.vertices.copy(), m.edges.copy()) for m in meshes)
        return meshes

    extraction_call = Mock(side_effect=capture)
    stage = Mock(wraps=dc.reconcile_contact_meshes)
    monkeypatch.setattr(dc, "compute_dual_contouring_v2", extraction_call)
    monkeypatch.setattr(dc, "reconcile_contact_meshes", stage)
    for helper in ("find_and_inject_multi_surface_constraints_multicore",
                   "average_overlapping_vertices", "remove_fault_overlap_triangles"):
        monkeypatch.setattr(dc, helper, Mock(side_effect=AssertionError("Legacy overlap must be bypassed")))
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    extraction_call.assert_called_once()
    stage.assert_called_once()
    assert len(extraction_call.call_args.kwargs["dc_data_list"]) == len(meshes) == 2
    controller, truncated, sign = case.contacts[0]
    seam_segments = []
    for index, (mesh, (tensor, before_vertices, before_faces)) in enumerate(zip(meshes, snapshots)):
        assert mesh.vertices_tensor is tensor
        np.testing.assert_array_equal(mesh.vertices_tensor, before_vertices)
        assert not np.shares_memory(mesh.vertices, mesh.vertices_tensor)
        assert isinstance(mesh.vertices, np.ndarray) and isinstance(mesh.edges, np.ndarray)
        assert len(mesh.edges) > 0 and np.isfinite(mesh.vertices).all()
        assert np.all((mesh.edges >= 0) & (mesh.edges < len(mesh.vertices)))
        np.testing.assert_allclose(mesh.vertices @ case.normals[index], case.levels[index], atol=1e-8, rtol=0)
        triangles = mesh.vertices[mesh.edges]
        assert np.all(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                              triangles[:, 2] - triangles[:, 0]), axis=1) > 1e-12)
        assert (mesh.stack_index, mesh.surface_index, mesh.exported_surface_index) == (index, 0, index)
        assert mesh.isovalue == case.levels[index]
        assert mesh.contact_report["cell_vertex_correspondence"] is False
        assert mesh.contact_report["differentiable_topology"] is False
        edges = np.concatenate([mesh.edges[:, [0, 1]], mesh.edges[:, [1, 2]], mesh.edges[:, [2, 0]]])
        unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        points = mesh.vertices[unique]
        on_seam = np.all(np.abs(points @ case.normals.T - case.levels) < 1e-8, axis=(1, 2))
        assert np.any(on_seam), "Coincident vertices without seam edges are insufficient"
        assert np.all(counts[on_seam] == (1 if index == truncated else 2))
        segments = [tuple(sorted(tuple(p) for p in np.round(segment, 8))) for segment in points[on_seam]]
        seam_segments.append(sorted(segments))
        if index == truncated:
            residuals = sign * (triangles @ case.normals[controller] - case.levels[controller])
            assert np.all(residuals >= -1e-8)
            assert np.any(residuals > 1e-3)
            assert not np.array_equal(mesh.edges, before_faces)
        else:
            residuals = triangles @ case.normals[truncated] - case.levels[truncated]
            assert np.any(residuals < -1e-3) and np.any(residuals > 1e-3)
    assert seam_segments[0] == seam_segments[1]
    assert inputs.grid is original_grid and inputs.weights is original_weights
    np.testing.assert_array_equal(inputs.grid.values, original_values)
    assert descriptor.stack_structure.stack_number == original_cursor
    assert options.model_dump_json() == original_options
    for output, field, mask in zip(levels[0].outputs, original_fields, original_masks):
        np.testing.assert_array_equal(output.exported_fields.scalar_field, field)
        np.testing.assert_array_equal(output.squeezed_mask_array, mask)


@pytest.mark.parametrize("name", ["single", "parallel_false_overlap"])
def test_no_contact_preserves_independent_extraction(extraction, monkeypatch, name):
    case, descriptor, inputs, options, levels = extraction(name)
    snapshots = []
    extract = dc.compute_dual_contouring_v2

    def capture(**kwargs):
        meshes = extract(**dict(kwargs, max_workers=1))
        snapshots.extend((m.vertices.copy(), m.edges.copy()) for m in meshes)
        return meshes

    call = Mock(side_effect=capture)
    monkeypatch.setattr(dc, "compute_dual_contouring_v2", call)
    meshes = dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    call.assert_called_once()
    assert len(meshes) == len(case.normals)
    for index, (mesh, (vertices, faces)) in enumerate(zip(meshes, snapshots)):
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.edges, faces)
        np.testing.assert_allclose(mesh.vertices @ case.normals[index], case.levels[index], atol=1e-8, rtol=0)
        if name == "single":
            assert mesh.contact_report is None
        else:
            assert mesh.contact_report["status"] == "no_contact"


@pytest.mark.parametrize("unsupported,match", [
    ("fault", "fault"), ("fault_matrix", "fault"), ("fault_data", "fault"),
    ("null_space", "null-space"), ("raw", "INTERSECT"), ("disjoint", "INTERSECT"),
    ("three_stacks", "one or two stacks"), ("two_surfaces", "one surface"),
    ("relations", "ERODE.*BASEMENT"), ("non_affine", "affine"),
])
def test_unsupported_inputs_fail_before_extraction(extraction, monkeypatch, unsupported, match):
    _, descriptor, inputs, options, levels = extraction()
    stacks = descriptor.stack_structure
    if unsupported == "fault":
        stacks.masking_descriptor[0] = StackRelationType.FAULT
    elif unsupported == "fault_matrix":
        stacks.faults_relations[0, 1] = True
    elif unsupported == "fault_data":
        stacks.faults_input_data = [object(), None]
    elif unsupported == "null_space":
        stacks.masking_descriptor[0] = StackRelationType.NULL_SPACE
    elif unsupported in ("raw", "disjoint"):
        options.evaluation_options.mesh_extraction_masking_options = getattr(
            MeshExtractionMaskingOptions, unsupported.upper())
    elif unsupported == "three_stacks":
        stacks.number_of_points_per_stack = np.zeros(3, dtype=int)
    elif unsupported == "two_surfaces":
        stacks.number_of_surfaces_per_stack[0] = 2
    elif unsupported == "relations":
        stacks.masking_descriptor[:] = [StackRelationType.BASEMENT] * 2
    elif unsupported == "non_affine":
        levels[0].outputs[0].exported_fields.scalar_field += inputs.grid.values[:, 0] ** 2
    calls = []
    for helper in ("get_triangulation_codes", "_interp_on_edges", "compute_dual_contouring_v2",
                   "reconcile_contact_meshes"):
        call = Mock(side_effect=AssertionError("Unsupported input reached extraction"))
        calls.append(call)
        monkeypatch.setattr(dc, helper, call)
    with pytest.raises(ValueError, match=match):
        dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    for call in calls:
        call.assert_not_called()


@pytest.mark.parametrize("backend,dtype,match", [
    (config.AvailableBackends.numpy, "float32", "float64|precision"),
    (config.AvailableBackends.PYTORCH, "float64", "NumPy|numpy|backend"),
])
def test_unsupported_backend_and_precision_fail_before_extraction(extraction, monkeypatch, backend, dtype, match):
    _, descriptor, inputs, options, levels = extraction()
    # Only validation should run; do not install a different tensor backend.
    monkeypatch.setattr(BackendTensor, "engine_backend", backend)
    monkeypatch.setattr(BackendTensor, "dtype", dtype)
    calls = []
    for helper in ("get_triangulation_codes", "_interp_on_edges", "compute_dual_contouring_v2",
                   "reconcile_contact_meshes"):
        call = Mock(side_effect=AssertionError("Unsupported backend reached extraction"))
        calls.append(call)
        monkeypatch.setattr(dc, helper, call)
    with pytest.raises(ValueError, match=match):
        dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    for call in calls:
        call.assert_not_called()


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
def test_compute_model_external_affine_fields_have_matching_seam_edges(extraction, name):
    # Use the backend fixture only: no synthetic factory or gradient patch here.
    case = build_contact_case(name, resolution=6)
    if name == "planar_erosion":
        # Two levels produce 12 cells per axis. The retained target patch
        # reaches z=.375, so .39 has no finite triangle support for a seam.
        case.levels[0] = .36
    functions = [CustomInterpolationFunctions(
        scalar_field_at_surface_points=np.array([level]),
        implicit_function=lambda xyz, normal=normal: xyz @ normal,
        gx_function=lambda xyz, normal=normal: np.full(len(xyz), normal[0]),
        gy_function=lambda xyz, normal=normal: np.full(len(xyz), normal[1]),
        gz_function=lambda xyz, normal=normal: np.full(len(xyz), normal[2]),
    ) for normal, level in zip(case.normals, case.levels)]
    descriptor = InputDataDescriptor(
        TensorsStructure(np.array([], dtype=int)),
        StacksStructure(np.zeros(2, dtype=int), np.zeros(2, dtype=int),
                        np.ones(2, dtype=int), list(case.relations),
                        interp_functions_per_stack=functions),
    )
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
    controller, truncated, sign = case.contacts[0]
    seam_segments = []
    for index, mesh in enumerate(solution.dc_meshes):
        assert mesh.vertices.dtype == np.float64
        assert mesh.contact_report["status"] == "reconciled"
        assert len(mesh.edges) > 0 and np.isfinite(mesh.vertices).all()
        assert np.all((mesh.edges >= 0) & (mesh.edges < len(mesh.vertices)))
        np.testing.assert_allclose(mesh.vertices @ case.normals[index], case.levels[index], atol=1e-8, rtol=0)
        triangles = mesh.vertices[mesh.edges]
        assert np.all(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                              triangles[:, 2] - triangles[:, 0]), axis=1) > 1e-12)
        if index == truncated:
            assert np.all(sign * (triangles @ case.normals[controller] - case.levels[controller]) >= -1e-8)
        edges = np.concatenate([mesh.edges[:, [0, 1]], mesh.edges[:, [1, 2]], mesh.edges[:, [2, 0]]])
        unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        points = mesh.vertices[unique]
        on_seam = np.all(np.abs(points @ case.normals.T - case.levels) < 1e-8, axis=(1, 2))
        assert np.any(on_seam)
        assert np.all(counts[on_seam] == (1 if index == truncated else 2))
        seam_segments.append(sorted(tuple(sorted(tuple(p) for p in np.round(segment, 8)))
                                    for segment in points[on_seam]))
    assert seam_segments[0] == seam_segments[1]


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
    for helper in ("prepare_planar_contact", "reconcile_contact_meshes"):
        monkeypatch.setattr(dc, helper, Mock(side_effect=AssertionError("Legacy flags entered contact stage")))
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
    for helper in ("prepare_planar_contact", "get_triangulation_codes", "_interp_on_edges",
                   "compute_dual_contouring_v2", "reconcile_contact_meshes"):
        call = Mock(side_effect=AssertionError("Combined contact flags reached extraction"))
        calls.append(call)
        monkeypatch.setattr(dc, helper, call)
    with pytest.raises(ValueError, match="contact_aware|overlap mode"):
        dc.dual_contouring_multi_scalar(descriptor, inputs, options, levels)
    for call in calls:
        call.assert_not_called()


@pytest.mark.parametrize("mode", ["none", "pretty", "watertight"])
def test_existing_modes_do_not_prepare_or_reconcile_contacts(extraction, monkeypatch, mode):
    _, descriptor, inputs, options, levels = extraction()
    monkeypatch.setattr(dc, "DUAL_CONTOURING_VERTEX_OVERLAP", getattr(config.DualContouringOverlap, mode))
    for helper in ("prepare_planar_contact", "reconcile_contact_meshes"):
        monkeypatch.setattr(dc, helper, Mock(side_effect=AssertionError("Legacy mode entered contact stage")))
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
