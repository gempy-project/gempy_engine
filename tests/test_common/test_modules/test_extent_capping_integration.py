"""Extent capping through the production model and boundary evaluation APIs."""

import importlib
import json
import warnings
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine.config import AvailableBackends
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data import TensorsStructure
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.input_data_descriptor import InputDataDescriptor
from gempy_engine.core.data.interpolation_functions import CustomInterpolationFunctions
from gempy_engine.core.data.interpolation_input import InterpolationInput
from gempy_engine.core.data.kernel_classes.orientations import Orientations
from gempy_engine.core.data.kernel_classes.surface_points import SurfacePoints
from gempy_engine.core.data.options import InterpolationOptions
from gempy_engine.core.data.options.evaluation_options import MeshExtentCapping
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.core.data.stack_relation_type import StackRelationType
from gempy_engine.core.data.stacks_structure import StacksStructure


dc = importlib.import_module("gempy_engine.API.dual_contouring.extent_capping")
EXTENT = np.array([-2., 4., 10., 14., -8., -3.])


@pytest.fixture(params=[AvailableBackends.numpy, AvailableBackends.PYTORCH], ids=["numpy", "pytorch-cpu"])
def backend(request, monkeypatch):
    torch = pytest.importorskip("torch") if request.param is AvailableBackends.PYTORCH else None
    grad_enabled = torch.is_grad_enabled() if torch is not None else None
    saved = dict(engine_backend=BackendTensor.engine_backend, use_gpu=BackendTensor.use_gpu,
                 use_pykeops=BackendTensor.use_pykeops, dtype=BackendTensor.dtype,
                 grads=BackendTensor.COMPUTE_GRADS)
    saved_pykeops = BackendTensor.pykeops_enabled
    monkeypatch.setenv("GEMPY_SKIP_TRIANGULATION", "0")
    try:
        BackendTensor._change_backend(request.param, use_gpu=False, use_pykeops=False, dtype="float64")
        BackendTensor.pykeops_enabled = False
        yield request.param
    finally:
        BackendTensor._change_backend(**saved)
        BackendTensor.COMPUTE_GRADS = saved["grads"]
        BackendTensor.pykeops_enabled = saved_pykeops
        if torch is not None:
            torch.set_grad_enabled(grad_enabled)


@pytest.fixture
def plane_model(backend):
    def make(levels=(-5.7,), normal=(0., 0., 1.), extent=EXTENT, resolution=(3, 3, 3)):
        functions = CustomInterpolationFunctions(
            scalar_field_at_surface_points=np.asarray(levels),
            implicit_function=lambda xyz: xyz[:, 0] * normal[0] + xyz[:, 1] * normal[1] + xyz[:, 2] * normal[2],
            gx_function=lambda xyz: xyz[:, 0] * 0 + normal[0],
            gy_function=lambda xyz: xyz[:, 0] * 0 + normal[1],
            gz_function=lambda xyz: xyz[:, 0] * 0 + normal[2],
        )
        structure = InputDataDescriptor(
            TensorsStructure(number_of_points_per_surface=np.array([], dtype=int)),
            StacksStructure(
                number_of_points_per_stack=np.array([0]),
                number_of_orientations_per_stack=np.array([0]),
                number_of_surfaces_per_stack=np.array([len(levels)]),
                masking_descriptor=[StackRelationType.BASEMENT],
                interp_functions_per_stack=[functions],
            ),
        )
        inputs = InterpolationInput(
            SurfacePoints(np.empty((0, 3))),
            Orientations(np.empty((0, 3)), np.empty((0, 3))),
            EngineGrid(octree_grid=RegularGrid(np.array(extent, dtype=float), list(resolution))),
            np.arange(len(levels) + 1),
        )
        options = InterpolationOptions.from_args(10., 1.)
        options.evaluation_options.number_octree_levels = 2
        options.evaluation_options.number_octree_levels_surface = 2
        return inputs, options, structure

    return make


@pytest.mark.parametrize("mode", list(MeshExtentCapping))
def test_capping_option_json_roundtrip(mode):
    options = InterpolationOptions.from_args(10., 1.)
    assert options.evaluation_options.mesh_extraction_extent_capping is MeshExtentCapping.NONE
    options.evaluation_options.mesh_extraction_extent_capping = mode
    serialized = options.model_dump_json()
    assert json.loads(serialized)["evaluation_options"]["mesh_extraction_extent_capping"] == mode.value
    restored = InterpolationOptions.model_validate_json(serialized)
    assert restored.evaluation_options.mesh_extraction_extent_capping is mode


def test_default_and_explicit_none_have_identical_arrays(plane_model, monkeypatch):
    boundary = Mock(side_effect=AssertionError("Disabled capping must not evaluate the boundary"))
    monkeypatch.setattr(dc, "_interp_on_boundary", boundary)
    inputs, options, descriptor = plane_model()
    default = compute_model(inputs, options, descriptor)
    inputs, options, descriptor = plane_model()
    options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.NONE
    explicit = compute_model(inputs, options, descriptor)
    assert len(default.dc_meshes) == len(explicit.dc_meshes) == 1
    np.testing.assert_array_equal(default.dc_meshes[0].vertices, explicit.dc_meshes[0].vertices)
    np.testing.assert_array_equal(default.dc_meshes[0].edges, explicit.dc_meshes[0].edges)
    assert default.dc_meshes[0].capping_report is None
    for first, second in zip(default.octrees_output, explicit.octrees_output):
        np.testing.assert_array_equal(BackendTensor.t.to_numpy(first.grid.octree_grid.values),
                                      BackendTensor.t.to_numpy(second.grid.octree_grid.values))
        np.testing.assert_array_equal(BackendTensor.t.to_numpy(first.outputs[0].exported_fields.scalar_field),
                                      BackendTensor.t.to_numpy(second.outputs[0].exported_fields.scalar_field))
    boundary.assert_not_called()


@pytest.mark.parametrize("mode", list(MeshExtentCapping))
def test_all_modes_preserve_exact_extent_and_caller_grid(plane_model, backend, mode):
    inputs, options, descriptor = plane_model()
    grid = inputs.grid
    root = grid.octree_grid
    original_extent = BackendTensor.t.to_numpy(root.orthogonal_extent).copy()
    original_values = BackendTensor.t.to_numpy(root.values).copy()
    options.evaluation_options.mesh_extraction_extent_capping = mode
    solution = compute_model(inputs, options, descriptor)
    assert len(solution.octrees_output) == 2
    root_values = BackendTensor.t.to_numpy(solution.octrees_output[0].grid.octree_grid.values)
    assert root_values.dtype == np.float64
    coordinate_tolerance = 2 * np.finfo(root_values.dtype).eps * np.max(np.abs(EXTENT))
    for level in solution.octrees_output:
        octree = level.grid.octree_grid
        np.testing.assert_array_equal(BackendTensor.t.to_numpy(octree.orthogonal_extent), EXTENT)
        if backend is AvailableBackends.PYTORCH:
            assert octree.orthogonal_extent.device.type == "cpu"
        coordinates = BackendTensor.t.to_numpy(octree.integer_coordinates)
        shape = BackendTensor.t.to_numpy(octree.regular_grid_shape)
        expected = EXTENT[::2] + (coordinates + .5) * (EXTENT[1::2] - EXTENT[::2]) / shape
        np.testing.assert_allclose(BackendTensor.t.to_numpy(octree.values), expected,
                                   rtol=0, atol=coordinate_tolerance)
    np.testing.assert_array_equal(
        BackendTensor.t.to_numpy(solution.octrees_output[0].grid.octree_grid.orthogonal_extent), EXTENT)
    assert inputs.grid is grid
    assert grid.octree_grid is root
    np.testing.assert_array_equal(BackendTensor.t.to_numpy(root.orthogonal_extent), EXTENT)
    np.testing.assert_array_equal(BackendTensor.t.to_numpy(root.orthogonal_extent), original_extent)
    np.testing.assert_array_equal(BackendTensor.t.to_numpy(root.values), original_values)


def test_enabled_planes_are_closed_with_analytic_volume_and_reuse_boundary(plane_model, monkeypatch):
    levels = (-4.7, -6.3)
    inputs, options, descriptor = plane_model(levels)
    options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.SCALAR_LESS_EQUAL
    boundary = Mock(wraps=dc._interp_on_boundary)
    monkeypatch.setattr(dc, "_interp_on_boundary", boundary)
    solution = compute_model(inputs, options, descriptor)
    boundary.assert_called_once()
    assert len(solution.dc_meshes) == len(levels)
    volumes = []
    for index, (mesh, height) in enumerate(zip(solution.dc_meshes, levels)):
        report = mesh.capping_report
        assert report["watertight"], report
        assert report["closure_success"], report
        assert not report["warnings"], report
        assert report["open_edges_before"] > 0
        assert report["open_edges_after"] == 0
        assert report["cap_max_plane_error"] == 0
        assert report["added_cap_triangles"] > 0
        assert report["added_transition_triangles"] > 0
        assert report["boundary_scalar_points"] == len(boundary.call_args.args[0])
        assert (mesh.stack_index, mesh.surface_index, mesh.exported_surface_index) == (0, index, index)
        assert mesh.isovalue == height
        assert mesh.inside_convention == "scalar <= isovalue"
        np.testing.assert_array_equal(mesh.vertices.min(axis=0), EXTENT[::2])
        np.testing.assert_array_equal(mesh.vertices.max(axis=0)[:2], EXTENT[1:4:2])
        assert np.all(mesh.vertices <= EXTENT[1::2])
        triangles = mesh.vertices[mesh.edges]
        edges = np.concatenate([mesh.edges[:, [0, 1]], mesh.edges[:, [1, 2]], mesh.edges[:, [2, 0]]])
        _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)
        # Shift the origin to avoid cancellation for a translated model box.
        triangles = triangles - EXTENT[::2]
        volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
        volumes.append(volume)
    np.testing.assert_allclose(volumes, 6 * 4 * (np.asarray(levels) + 8), rtol=1e-10, atol=0)


@pytest.mark.parametrize("fail", [False, True], ids=["success", "failure"])
def test_boundary_chunks_restore_grid_stack_and_options(plane_model, monkeypatch, fail):
    inputs, options, _ = plane_model()
    descriptor = SimpleNamespace(stack_structure=SimpleNamespace(n_stacks=2, stack_number=1))
    saved_grid = inputs.grid
    options.evaluation_options.evaluation_chunk_size = 3
    options.evaluation_options.compute_scalar = False
    options.evaluation_options.compute_scalar_gradient = True
    saved_options = options.model_dump_json()
    points = np.arange(21, dtype=float).reshape(7, 3)
    seen = []

    def interpolate(actual_input, boundary_options, actual_descriptor):
        assert actual_input is inputs
        assert actual_descriptor is descriptor
        assert boundary_options is not options
        assert boundary_options.evaluation_options.compute_scalar is True
        assert boundary_options.evaluation_options.compute_scalar_gradient is False
        xyz = actual_input.grid.custom_grid.values
        seen.append(BackendTensor.t.to_numpy(xyz).copy())
        descriptor.stack_structure.stack_number = 0
        if fail and len(seen) == 2:
            raise RuntimeError("boundary evaluation failed")
        # Include sentinel values outside the custom-grid slice.
        return [SimpleNamespace(
            grid=SimpleNamespace(custom_grid_slice=slice(1, len(xyz) + 1)),
            exported_fields=SimpleNamespace(scalar_field=BackendTensor.t.concatenate([
                BackendTensor.t.array([-999.]), xyz[:, 2] + offset, BackendTensor.t.array([-999.])
            ])),
        ) for offset in (0, 100)]

    interpolation = Mock(side_effect=interpolate)
    monkeypatch.setattr(dc, "interpolate_all_fields_no_octree", interpolation)
    if fail:
        with pytest.raises(RuntimeError, match="boundary evaluation failed"):
            dc._interp_on_boundary(points, inputs, options, descriptor)
    else:
        scalars = dc._interp_on_boundary(points, inputs, options, descriptor)
        assert len(scalars) == 2
        for actual, offset in zip(scalars, (0, 100)):
            np.testing.assert_array_equal(actual, points[:, 2] + offset)
    assert interpolation.call_count == (2 if fail else 3)
    np.testing.assert_array_equal(np.concatenate(seen), points[:6] if fail else points)
    assert inputs.grid is saved_grid
    assert descriptor.stack_structure.stack_number == 1
    assert options.model_dump_json() == saved_options


def assert_report_matches_mesh(mesh, emitted):
    """Best-effort closure must not conceal topology or geometry defects."""
    report = mesh.capping_report
    assert report is not None
    assert np.isfinite(mesh.vertices).all()
    assert mesh.edges.ndim == 2 and mesh.edges.shape[1] == 3
    assert np.all((mesh.edges >= 0) & (mesh.edges < len(mesh.vertices)))
    edges = np.concatenate([mesh.edges[:, [0, 1]], mesh.edges[:, [1, 2]], mesh.edges[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    assert report["open_edges_after"] == np.count_nonzero(counts == 1), report
    assert report["nonmanifold_edges"] == np.count_nonzero(counts > 2), report
    directions = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    assert report["orientation_conflicts"] == np.count_nonzero((counts == 2) & (np.abs(directions) == 2)), report
    triangles = mesh.vertices[mesh.edges]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert report["degenerate_triangles"] == np.count_nonzero(np.all(cross == 0, axis=1)), report
    defects = ("missing_qef", "escaped_qef", "unsupported_topology", "boundary_edges",
               "nonmanifold_edges", "orientation_conflicts", "nonmanifold_vertices",
               "invalid_triangles", "nonfinite_vertices", "nonfinite_triangles", "degenerate_triangles")
    for key in defects:
        if report[key]:
            assert not report["closure_success"], report
            assert f"{key}: {report[key]}" in report["warnings"], report
    if report["warnings"]:
        assert any(issubclass(w.category, RuntimeWarning)
                   and all(message in str(w.message) for message in report["warnings"])
                   for w in emitted), report
    if report["closure_success"]:
        assert report["watertight"] and not report["warnings"], report
    elif len(mesh.edges):
        assert report["warnings"], report
    if report["watertight"]:
        assert len(counts) > 0
        np.testing.assert_array_equal(counts, 2)
    assert report["added_triangles"] == report["added_cap_triangles"] + report["added_transition_triangles"]
    assert report["cap_max_plane_error"] == 0


@pytest.mark.parametrize("normal,level,volume", [
    ((-.2, -.1, 1.), .313, .463),
    ((.2, .1, -1.), -.313, .537),
])
def test_oblique_plane_production_volume(plane_model, normal, level, volume):
    inputs, options, descriptor = plane_model((level,), normal, (0, 1, 0, 1, 0, 1))
    options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.SCALAR_LESS_EQUAL
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        solution = compute_model(inputs, options, descriptor)
    assert len(solution.dc_meshes) == 1
    mesh = solution.dc_meshes[0]
    assert_report_matches_mesh(mesh, emitted)
    assert mesh.capping_report["closure_success"], mesh.capping_report
    original = mesh.vertices[:mesh.capping_report["original_vertex_count"]]
    np.testing.assert_allclose(original @ normal, level, rtol=0, atol=2e-13)
    assert np.all(mesh.vertices >= -2e-13) and np.all(mesh.vertices <= 1 + 2e-13)
    triangles = mesh.vertices[mesh.edges]
    actual_volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    assert actual_volume == pytest.approx(volume, rel=1e-11)


@pytest.mark.parametrize("normal,level", [
    ((0., 0., 1.), .5),
    ((1., 1., 1.), 1.),
    ((1., 1., 1.), 1.5),
], ids=["lattice-plane", "box-corners", "lattice-points"])
def test_exact_plane_ties_have_honest_diagnostics(plane_model, normal, level):
    inputs, options, descriptor = plane_model((level,), normal, (0, 1, 0, 1, 0, 1), (2, 2, 2))
    options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.SCALAR_LESS_EQUAL
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        solution = compute_model(inputs, options, descriptor)
    assert len(solution.dc_meshes) == 1
    mesh = solution.dc_meshes[0]
    assert len(mesh.vertices) > 0 and len(mesh.edges) > 0
    assert_report_matches_mesh(mesh, emitted)
    report = mesh.capping_report
    assert report["closure_success"] or report["warnings"], report
    added = mesh.vertices[report["original_vertex_count"]:]
    assert len(np.unique(added, axis=0)) == len(added)


@pytest.mark.parametrize("height", [-9., -2.], ids=["fully-outside", "fully-inside"])
def test_nonintersecting_plane_produces_no_artificial_box(plane_model, height):
    inputs, options, descriptor = plane_model((height,))
    options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.SCALAR_LESS_EQUAL
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        solution = compute_model(inputs, options, descriptor)
    assert len(solution.dc_meshes) == 1
    mesh = solution.dc_meshes[0]
    assert mesh.vertices.shape == (0, 3)
    assert mesh.edges.shape == (0, 3)
    assert_report_matches_mesh(mesh, emitted)
    assert mesh.capping_report["added_vertices"] == 0
    assert mesh.capping_report["added_triangles"] == 0
    assert not mesh.capping_report["closure_success"]


@pytest.mark.parametrize("skip", ["1", "true"])
def test_skip_triangulation_skips_capping(plane_model, monkeypatch, skip):
    inputs, options, descriptor = plane_model()
    options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.SCALAR_LESS_EQUAL
    monkeypatch.setenv("GEMPY_SKIP_TRIANGULATION", skip)
    boundary = Mock(side_effect=AssertionError("Skipped triangulation must not evaluate caps"))
    cap = Mock(side_effect=AssertionError("Skipped triangulation must not add triangles"))
    monkeypatch.setattr(dc, "_interp_on_boundary", boundary)
    monkeypatch.setattr(dc, "cap_mesh", cap)
    solution = compute_model(inputs, options, descriptor)
    assert len(solution.dc_meshes) == 1
    mesh = solution.dc_meshes[0]
    assert len(mesh.vertices) > 0
    assert mesh.edges.size == 0
    assert mesh.capping_report is None
    boundary.assert_not_called()
    cap.assert_not_called()


@pytest.mark.parametrize("backend", [AvailableBackends.numpy], indirect=True)
@pytest.mark.parametrize("normal,level", [((0., 0., 1.), .43), ((-.2, -.1, 1.), .313)])
def test_numpy_pytorch_mesh_equivalence(plane_model, normal, level):
    torch = pytest.importorskip("torch")
    saved_grad = torch.is_grad_enabled()
    meshes = []
    try:
        for engine in (AvailableBackends.numpy, AvailableBackends.PYTORCH):
            BackendTensor._change_backend(engine, use_gpu=False, use_pykeops=False,
                                          dtype="float64", grads=saved_grad)
            inputs, options, descriptor = plane_model((level,), normal, (0, 1, 0, 1, 0, 1))
            options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.SCALAR_LESS_EQUAL
            meshes.append(compute_model(inputs, options, descriptor).dc_meshes[0])
        first, second = meshes
        assert first.capping_report["closure_success"], first.capping_report
        assert second.capping_report["closure_success"], second.capping_report
        np.testing.assert_array_equal(first.edges, second.edges)
        np.testing.assert_allclose(first.vertices, second.vertices, rtol=0, atol=2e-13)
        assert first.capping_report == second.capping_report
    finally:
        torch.set_grad_enabled(saved_grad)


def test_fault_model_capping_reports_all_limitations(graben_fault_model, monkeypatch):
    inputs, descriptor, options = deepcopy(graben_fault_model)
    options.evaluation_options.number_octree_levels = 3
    options.evaluation_options.number_octree_levels_surface = 3
    options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.SCALAR_LESS_EQUAL
    monkeypatch.setenv("GEMPY_SKIP_TRIANGULATION", "0")
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        solution = compute_model(inputs, options, descriptor)
    assert len(solution.dc_meshes) == 6
    assert {mesh.stack_index for mesh in solution.dc_meshes} == {0, 1, 2}
    for mesh in solution.dc_meshes:
        assert_report_matches_mesh(mesh, emitted)
        assert mesh.capping_report["boundary_scalar_points"] == 0
        assert mesh.capping_report["added_triangles"] == 0
        assert "boundary ownership" in mesh.capping_report["skipped_reason"]
        assert mesh.inside_convention == "scalar <= isovalue"


def test_capped_export_offsets_do_not_mutate_meshes(plane_model, monkeypatch):
    raw_module = importlib.import_module("gempy_engine.core.data.raw_arrays_solution")
    monkeypatch.setattr(raw_module, "require_subsurface", lambda: SimpleNamespace(
        UnstructuredData=SimpleNamespace(from_array=lambda **kwargs: kwargs)))
    monkeypatch.setattr(raw_module, "require_pandas", lambda: SimpleNamespace(DataFrame=lambda data: data))
    inputs, options, descriptor = plane_model((-4.7, -6.3))
    options.evaluation_options.mesh_extraction_extent_capping = MeshExtentCapping.SCALAR_LESS_EQUAL
    meshes = compute_model(inputs, options, descriptor).dc_meshes
    raw = raw_module.RawArraysSolution(vertices=[mesh.vertices for mesh in meshes],
                                       edges=[mesh.edges for mesh in meshes])
    original = [mesh.edges.copy() for mesh in meshes]
    expected = np.concatenate((original[0], original[1] + len(meshes[0].vertices)))
    for _ in range(2):
        exported = raw.meshes_to_subsurface()
        np.testing.assert_array_equal(exported["cells"], expected)
        for mesh, triangles in zip(meshes, original):
            np.testing.assert_array_equal(mesh.edges, triangles)
