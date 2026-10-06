"""Phase 1 legacy baselines, not correctness oracles for contact-aware meshing.

Four isolated imports cover configuration and production extraction together.
The direct parallel-plane probe deliberately has no geological contact: sharing
a voxel is not an intersection. No shared analytic-case fixtures are required.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = Path(__file__).with_name("contact_legacy_reference.json")


def _mesh_reference(mesh):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.edges)
    assert vertices.dtype == np.float64
    assert vertices.ndim == faces.ndim == 2
    assert vertices.shape[1] == faces.shape[1] == 3
    assert len(vertices) > 0 and len(faces) > 0
    assert np.isfinite(vertices).all()
    assert np.issubdtype(faces.dtype, np.integer)
    assert np.all((faces >= 0) & (faces < len(vertices)))
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    assert np.all(areas > 1e-10)
    assert len(np.unique(np.sort(faces, axis=1), axis=0)) == len(faces)
    vi = np.linspace(0, len(vertices) - 1, 3, dtype=int)
    fi = np.linspace(0, len(faces) - 1, 3, dtype=int)
    return dict(vertices=len(vertices), triangles=len(faces),
                bounds=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                sum=vertices.sum(axis=0).tolist(), positions=vertices[vi].tolist(),
                faces=faces[fi].tolist(),
                connectivity_sha256=hashlib.sha256(faces.astype("<i8").tobytes()).hexdigest())


def _parallel_qef_probe():
    from copy import deepcopy
    from types import SimpleNamespace
    from gempy_engine.core.data.dual_contouring_data import DualContouringData
    from gempy_engine.modules.dual_contouring._gen_vertices import generate_dual_contouring_vertices
    from gempy_engine.modules.dual_contouring.overlapping import average_overlapping_vertices
    from gempy_engine.modules.dual_contouring.weighted_qef_setup_multicore import (
        _build_allowed_partners, find_and_inject_multi_surface_constraints_multicore,
    )

    # Four vertical edge crossings in one unit cell, at two disjoint heights.
    heights = np.array([.4, .6])
    codes = np.zeros((1, 3), dtype=int)
    valid = np.zeros((1, 12), dtype=bool)
    valid[0, :4] = True
    data = [DualContouringData(
        xyz_on_edge=np.array([[0., 0., z], [1., 0., z], [0., 1., z], [1., 1., z]]),
        valid_edges=valid.copy(), xyz_on_centers=np.array([[.5, .5, .5]]),
        dxdydz=np.ones(3), n_surfaces_to_export=1, left_right_codes=codes.copy(),
        gradients=np.tile([0., 0., 1.], (4, 1)), base_number=(2, 2, 2),
    ) for z in heights]
    clean = np.array([generate_dual_contouring_vertices(d)[0] for d in data])
    np.testing.assert_allclose(clean[:, 2], heights, rtol=0, atol=1e-15)
    results = {}
    for name, stacks, relations in [
        ("different_stacks", [0, 1], np.zeros((2, 2), dtype=bool)),
        ("same_stack_empty_partners", [0, 0], np.zeros((1, 1), dtype=bool)),
        ("same_stack_absent_fault_matrix", [0, 0], None),
    ]:
        probe = deepcopy(data)
        find_and_inject_multi_surface_constraints_multicore(
            probe, [codes, codes], (2, 2, 2), max_workers=1,
            surface_to_stack=stacks, faults_relations=relations,
        )
        raw = np.array([generate_dual_contouring_vertices(d)[0] for d in probe])
        # Independent weighted least-squares oracle: 4 own rows + 1 mass
        # point row, versus 4 partner rows with weight 10. Not mesh-derived.
        expected = (5 * heights + 40 * heights[::-1]) / 45
        np.testing.assert_allclose(raw[:, 2], expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(raw[:, :2], .5, rtol=0, atol=1e-15)
        assert all(np.count_nonzero(d.extra_weights == 10) == 4 for d in probe)
        meshes = [SimpleNamespace(vertices=raw[i:i + 1].copy()) for i in range(2)]
        average_overlapping_vertices(
            meshes, [codes, codes], (2, 2, 2), surface_to_stack=stacks,
            stacks_structure=SimpleNamespace(faults_relations=relations, n_stacks=max(stacks) + 1),
        )
        post = np.array([m.vertices[0, 2] for m in meshes])
        np.testing.assert_allclose(post, .5 if stacks == [0, 1] else expected,
                                   rtol=0, atol=1e-15)
        partners = _build_allowed_partners(stacks, relations, 2)
        results[name] = dict(clean_z=clean[:, 2].tolist(), raw_z=raw[:, 2].tolist(),
                             post_z=post.tolist(),
                             partners=None if partners is None else
                             [None if p is None else sorted(p) for p in partners])
    return results


def _worker():
    """Invoked only in a fresh interpreter, never in pytest's backend process."""
    import importlib
    from copy import deepcopy
    from unittest.mock import patch

    # Ignore machine-local .env files, including a possible overlap override
    # when the default test has explicitly removed the environment variable.
    with patch("dotenv.load_dotenv", return_value=False):
        from gempy_engine import config
        from gempy_engine.API.model.model_api import compute_model
        from gempy_engine.core.backend_tensor import BackendTensor
        from gempy_engine.core.data.options.evaluation_options import (
            MeshExtractionMaskingOptions, MeshExtentCapping, OctreeRefinementMode, TriangulationMethod,
        )
        from tests.fixtures.simple_models import unconformity_complex_factory
        dc = importlib.import_module("gempy_engine.API.dual_contouring.multi_scalar_dual_contouring")

    selected = config.DUAL_CONTOURING_VERTEX_OVERLAP
    assert dc.DUAL_CONTOURING_VERTEX_OVERLAP is selected
    os.environ["DUAL_CONTOURING_VERTEX_OVERLAP"] = "pretty" if selected.name == "none" else "none"
    assert config.DUAL_CONTOURING_VERTEX_OVERLAP is selected
    assert dc.DUAL_CONTOURING_VERTEX_OVERLAP is selected
    BackendTensor._change_backend(config.AvailableBackends.numpy, use_gpu=False,
                                  use_pykeops=False, dtype="float64", grads=False)
    BackendTensor.pykeops_enabled = False
    inputs, options, descriptor = unconformity_complex_factory()
    evaluation = options.evaluation_options
    evaluation.number_octree_levels = 2
    evaluation.number_octree_levels_surface = 2
    evaluation.mesh_extraction = True
    evaluation.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    evaluation.mesh_extraction_extent_capping = MeshExtentCapping.NONE
    evaluation.octree_refinement_mode = OctreeRefinementMode.FAST
    evaluation.triangulation_method = TriangulationMethod.LEGACY
    evaluation.deduplicate_octree_corners = False
    evaluation.compute_scalar = True
    evaluation.compute_scalar_gradient = False
    options.debug = False
    original_grid = inputs.grid
    original_values = inputs.grid.octree_grid.values.copy()
    original_points = inputs.surface_points.sp_coords.copy()
    raw_snapshots, clean_snapshots, deltas, calls = [], [], [], []
    extract = dc.compute_dual_contouring_v2
    inject = dc.find_and_inject_multi_surface_constraints_multicore

    def capture(dc_data_list, max_workers=None):
        meshes = extract(dc_data_list, max_workers=1)
        raw_snapshots[:] = [_mesh_reference(m) for m in meshes]
        clean_data = deepcopy(dc_data_list)
        for data in clean_data:
            data.extra_edge_xyz = data.extra_edge_normals = data.extra_weights = None
        clean_meshes = extract(clean_data, max_workers=1)
        clean_snapshots[:] = [_mesh_reference(m) for m in clean_meshes]
        deltas[:] = [float(np.max(np.linalg.norm(m.vertices - c.vertices, axis=1)))
                     for m, c in zip(meshes, clean_meshes)]
        calls.append(len(meshes))
        return meshes

    def serial_inject(**kwargs):
        kwargs["max_workers"] = 1
        return inject(**kwargs)

    with patch.object(dc, "compute_dual_contouring_v2", side_effect=capture), \
            patch.object(dc, "find_and_inject_multi_surface_constraints_multicore", side_effect=serial_inject) as qef, \
            patch.object(dc, "average_overlapping_vertices", wraps=dc.average_overlapping_vertices) as average, \
            patch.object(dc, "remove_fault_overlap_triangles", wraps=dc.remove_fault_overlap_triangles) as remove:
        solution = compute_model(inputs, options, descriptor)
    assert inputs.grid is original_grid
    np.testing.assert_array_equal(inputs.grid.octree_grid.values, original_values)
    np.testing.assert_array_equal(inputs.surface_points.sp_coords, original_points)
    result = dict(mode=selected.name, calls=calls,
                  dispatch=[qef.call_count, average.call_count, remove.call_count],
                  clean=clean_snapshots, raw=raw_snapshots,
                  final=[_mesh_reference(m) for m in solution.dc_meshes], qef_displacement=deltas)
    # Independently run the uninstrumented production path too: capturing stages
    # and forcing one QEF worker must not change the stored legacy baseline.
    fresh_inputs, _, fresh_descriptor = unconformity_complex_factory()
    untouched = compute_model(fresh_inputs, deepcopy(options), fresh_descriptor)
    assert [_mesh_reference(m) for m in untouched.dc_meshes] == result["final"]
    if selected.name == "none":
        result["parallel_qef"] = _parallel_qef_probe()
    print("CONTACT_LEGACY_RESULT=" + json.dumps(result, allow_nan=False))


@pytest.fixture(scope="module")
def legacy_runs():
    results = {}
    for mode in ("default", "none", "pretty", "watertight"):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("GEMPY_", "DUAL_CONTOURING_"))}
        env.update(DEFAULT_BACKEND="numpy", DEFAULT_TENSOR_DTYPE="float64", DEFAULT_PYKEOPS="False",
                   USE_GPU="False", DEBUG_MODE="False", OPTIMIZE_MEMORY="True",
                   NOT_MAKE_INPUT_DEEP_COPY="False", SET_RAW_ARRAYS_IN_SOLUTION="False",
                   ONLY_LITH_SOLUTION="False", LINE_PROFILER_ENABLED="False",
                   PYKEOPS_SOLVER="False",
                   GEMPY_FLAT_STACKS="False", GEMPY_SKIP_TRIANGULATION="0",
                   DUAL_CONTOURING_MULTITHREAD="False", DUAL_CONTOURING_FAULT_OVERLAP_THREADING="False",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", BLIS_NUM_THREADS="1", PYTHONHASHSEED="0", MPLBACKEND="Agg",
                   PYTHONPATH=str(ROOT))
        if mode != "default":
            env["DUAL_CONTOURING_VERTEX_OVERLAP"] = mode
        command = [sys.executable, "-c", "import runpy; runpy.run_path(" + repr(str(Path(__file__).resolve()))
                   + ")[\"_worker\"]()"]
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        payload = [line.removeprefix("CONTACT_LEGACY_RESULT=") for line in completed.stdout.splitlines()
                   if line.startswith("CONTACT_LEGACY_RESULT=")]
        assert len(payload) == 1, completed.stdout
        results[mode] = json.loads(payload[0])
    return results


@pytest.mark.parametrize("mode", ["default", "none", "pretty", "watertight"])
def test_legacy_compute_model_numerical_reference(legacy_runs, mode):
    """Static pre-change samples and full connectivity digest catch joint regressions."""
    references = json.loads(REFERENCE.read_text())
    actual = legacy_runs[mode]
    assert actual["mode"] == ("none" if mode == "default" else mode)
    expected = references[references["modes"][actual["mode"]]]
    assert len(actual["final"]) == len(expected) == 4
    for mesh, reference in zip(actual["final"], expected):
        for key in ("vertices", "triangles", "faces", "connectivity_sha256"):
            assert mesh[key] == reference[key], key
        for key in ("bounds", "sum", "positions"):
            np.testing.assert_allclose(mesh[key], reference[key], rtol=1e-10, atol=1e-8, err_msg=key)


def test_legacy_import_time_dispatch_and_default(legacy_runs):
    assert legacy_runs["default"] == legacy_runs["none"]
    for mode, run in legacy_runs.items():
        assert run["calls"] == [1, 2, 4]  # Earlier meshes are rebuilt inside the stack loop.
        assert run["dispatch"] == ([0, 0, 0] if mode in ("default", "none") else [3, 3, 3])
    assert legacy_runs["pretty"]["final"] == legacy_runs["watertight"]["final"]
    assert legacy_runs["none"]["final"] != legacy_runs["pretty"]["final"]


def test_legacy_qef_extraction_is_distinct_from_post_overlap(legacy_runs):
    none = legacy_runs["none"]
    assert none["clean"] == none["raw"] == none["final"]
    assert none["qef_displacement"] == [0., 0., 0., 0.]
    for mode in ("pretty", "watertight"):
        run = legacy_runs[mode]
        assert run["clean"] == none["clean"]
        assert min(run["qef_displacement"]) > 1e-3
        assert run["raw"] != run["clean"]
        assert run["final"] != run["raw"]
        # Non-fault overlap moves vertices but does not reconcile connectivity.
        assert [m["connectivity_sha256"] for m in run["raw"]] == [
            m["connectivity_sha256"] for m in run["final"]]


def test_parallel_same_cell_surfaces_are_distorted_before_averaging(legacy_runs):
    probe = legacy_runs["none"]["parallel_qef"]
    assert probe["different_stacks"]["partners"] == [[1], [0]]
    assert probe["same_stack_empty_partners"]["partners"] == [None, None]
    assert probe["same_stack_absent_fault_matrix"]["partners"] is None
    for case in probe.values():
        np.testing.assert_allclose(case["raw_z"], [26 / 45, 19 / 45], rtol=0, atol=1e-15)
        assert case["raw_z"][0] > case["raw_z"][1]  # Even reverses stratigraphic ordering.
    np.testing.assert_allclose(probe["different_stacks"]["post_z"], [.5, .5], rtol=0, atol=1e-15)
    for name in ("same_stack_empty_partners", "same_stack_absent_fault_matrix"):
        assert probe[name]["post_z"] == probe[name]["raw_z"]  # Same-stack averaging is skipped.
