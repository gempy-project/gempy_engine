"""Phase 1 legacy baselines, independent of the contact correctness catalogue.

Run with single-thread BLAS (OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1).
GEMPY_CONTACT_BENCHMARK_ROUNDS controls pedantic rounds (default 5); iterations
are always 1 so mutable inputs are fresh for every timed call. Use --benchmark-json
to retain extra_info. No timing thresholds or seam-correctness claims are made.
"""

import copy
import importlib
import os
from itertools import combinations
from unittest.mock import patch

import numpy as np
import pytest

from gempy_engine.API.model.model_api import compute_model
from gempy_engine import config
from gempy_engine.config import AvailableBackends, DualContouringOverlap
from gempy_engine.core.backend_tensor import BackendTensor
from gempy_engine.core.data.dual_contouring_mesh import DualContouringMesh
from gempy_engine.core.data.engine_grid import EngineGrid
from gempy_engine.core.data.options.evaluation_options import (
    MeshExtractionMaskingOptions, MeshExtentCapping, OctreeRefinementMode, TriangulationMethod,
)
from gempy_engine.core.data.regular_grid import RegularGrid
from gempy_engine.modules.dual_contouring._find_vertex_overlap import _generate_voxel_codes
from gempy_engine.modules.dual_contouring.overlapping import (
    average_overlapping_vertices, remove_fault_overlap_triangles,
)
from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
from tests.fixtures.simple_models import (
    _gen_simple_model_3_layers, unconformity_complex_factory,
)


dc = importlib.import_module("gempy_engine.API.dual_contouring.multi_scalar_dual_contouring")


@pytest.fixture(autouse=True)
def numpy_benchmark_backend():
    previous = dict(engine_backend=BackendTensor.engine_backend, use_gpu=BackendTensor.use_gpu,
                    use_pykeops=BackendTensor.use_pykeops, dtype=BackendTensor.dtype,
                    grads=BackendTensor.COMPUTE_GRADS)
    pykeops_enabled = BackendTensor.pykeops_enabled
    BackendTensor._change_backend(AvailableBackends.numpy, use_gpu=False,
                                  use_pykeops=False, dtype="float64")
    BackendTensor.pykeops_enabled = False
    try:
        yield
    finally:
        BackendTensor._change_backend(**previous)
        BackendTensor.pykeops_enabled = pykeops_enabled


def make_model(case, resolution):
    """Use existing factories, never mutate their session-scoped fixture objects."""
    if case == "unconformity":
        model = unconformity_complex_factory()
        extent = [0, 10, 0, 2, 0, 5]
    else:
        extent = [0.25, 0.75, 0.25, 0.75, 0.25, 0.75]
        grid = EngineGrid(octree_grid=RegularGrid(extent, [resolution] * 3))
        descriptor, interpolation_input, options = _gen_simple_model_3_layers(grid)
        model = interpolation_input, options, descriptor
    interpolation_input, options, descriptor = copy.deepcopy(model)
    interpolation_input.original_grid.octree_grid = RegularGrid(extent, [resolution] * 3)
    interpolation_input.set_grid_to_original()
    options.debug = False
    options.cache_mode = options.CacheMode.NO_CACHE
    evaluation = options.evaluation_options
    evaluation.number_octree_levels = 2
    evaluation.number_octree_levels_surface = 2
    evaluation.mesh_extraction = True
    evaluation.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    evaluation.mesh_extraction_extent_capping = MeshExtentCapping.NONE
    evaluation.octree_refinement_mode = OctreeRefinementMode.FAST
    evaluation.triangulation_method = TriangulationMethod.LEGACY
    evaluation.deduplicate_octree_corners = False
    return interpolation_input, options, descriptor


def legacy_overlap(meshes, cell_coordinates, base_number, surface_to_stack, structure, enabled):
    if enabled:
        average_overlapping_vertices(meshes, cell_coordinates, base_number, surface_to_stack, structure)
        remove_fault_overlap_triangles(meshes, cell_coordinates, base_number, surface_to_stack, structure)
    return meshes


def extract(descriptor, interpolation_input, options, octrees):
    # Match compute_model's cold-cache lifecycle; preparation is not timed.
    try:
        return dc.dual_contouring_multi_scalar(descriptor, interpolation_input, options, octrees)
    finally:
        WeightCache.clear_cache()


def prepare(scope, case, resolution, mode):
    """Return target, untimed per-round setup, and JSON-safe workload metadata."""
    model = make_model(case, resolution)
    interpolation_input, options, descriptor = model
    info = dict(scope=scope, case=case, mode=mode, backend="numpy", dtype="float64",
                root_resolution=[resolution] * 3, octree_levels=2,
                masking="INTERSECT", extent_capping="none", cache="NO_CACHE",
                stacks=descriptor.stack_structure.n_stacks,
                horizons_per_stack=descriptor.stack_structure.number_of_surfaces_per_stack.tolist(),
                input_triangles=None,
                actual_contact_cells=None, inserted_seam_vertices=0,
                contact_count_note="Legacy shared cells are candidates, not a geometric contact oracle.",
                environment_flags={key: getattr(config, key) for key in
                                   ("OPTIMIZE_MEMORY", "SET_RAW_ARRAYS_IN_SOLUTION",
                                    "NOT_MAKE_INPUT_DEEP_COPY", "LINE_PROFILER_ENABLED",
                                    "DUAL_CONTOURING_FAULT_OVERLAP_THREADING")},
                threads={key: os.getenv(key) for key in
                         ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                          "DUAL_CONTOURING_MULTITHREAD")})

    if scope == "model":
        return compute_model, lambda: (copy.deepcopy(model), {}), info

    # Keep mesh_extraction enabled to evaluate corners, but omit extraction itself.
    with patch("gempy_engine.API.model.model_api.dual_contouring_multi_scalar", return_value=None):
        octrees = compute_model(*copy.deepcopy(model)).octrees_output
    extraction_args = descriptor, interpolation_input, options, octrees
    info["leaf_cells"] = len(octrees[-1].grid.octree_grid.values)
    if scope == "extraction":
        return extract, lambda: (copy.deepcopy(extraction_args), {}), info

    captured = {}
    original = dc.compute_dual_contouring_v2

    def capture(*args, **kwargs):
        meshes = original(*args, **kwargs)
        data = kwargs["dc_data_list"]
        # Only the final accumulated-surface pass, before destructive averaging.
        captured["args"] = (
            [DualContouringMesh(m.vertices.copy(), m.edges.copy()) for m in meshes],
            [d.left_right_codes[d.valid_voxels].copy() for d in data],
            data[0].base_number,
            [d.n_surfaces_to_export for d in data],
            copy.deepcopy(descriptor.stack_structure),
            mode != "none" and descriptor.stack_structure.n_stacks > 1,
        )
        return meshes

    with patch.object(dc, "compute_dual_contouring_v2", side_effect=capture):
        extract(*copy.deepcopy(extraction_args))
    stage_args = captured["args"]
    meshes, coordinates, base, mapping, _, enabled = stage_args
    add_cell_counts(info, coordinates, base, mapping)
    info.update(overlap_dispatched=enabled,
                stage_note="Final accumulated pass only; excludes QEF and repeated earlier stack passes.")
    add_mesh_counts(info, meshes, "input")
    return legacy_overlap, lambda: (copy.deepcopy(stage_args), {}), info


def add_mesh_counts(info, meshes, prefix):
    info[f"{prefix}_vertices"] = sum(len(m.vertices) for m in meshes)
    info[f"{prefix}_triangles"] = sum(len(m.edges) for m in meshes)
    if meshes and all(m.dc_data is not None for m in meshes):
        data = [m.dc_data for m in meshes]
        add_cell_counts(info, [d.left_right_codes[d.valid_voxels] for d in data],
                        data[0].base_number, [d.n_surfaces_to_export for d in data])


def add_cell_counts(info, coordinates, base, mapping):
    codes = _generate_voxel_codes(coordinates, base)
    pairs = [(i, j) for i, j in combinations(range(len(codes)), 2) if mapping[i] != mapping[j]]
    shared = [np.intersect1d(codes[i], codes[j]) for i, j in pairs]
    info.update(active_cells_per_surface=[len(c) for c in codes],
                candidate_surface_pairs=len(pairs),
                shared_cell_pair_incidences=sum(len(c) for c in shared),
                unique_shared_cells=len(np.unique(np.concatenate(shared))) if shared else 0)


@pytest.mark.parametrize("resolution", [4, 8], ids=lambda value: f"r{value}")
@pytest.mark.parametrize("mode", ["none", "pretty", "watertight"])
@pytest.mark.parametrize("case", ["single_stack", "unconformity"])
@pytest.mark.parametrize("scope", ["overlap", "extraction", "model"])
def test_benchmark_contacts(benchmark, monkeypatch, scope, case, resolution, mode):
    monkeypatch.setenv("DUAL_CONTOURING_MULTITHREAD", "False")
    monkeypatch.setenv("GEMPY_SKIP_TRIANGULATION", "0")
    monkeypatch.setattr(dc, "DUAL_CONTOURING_VERTEX_OVERLAP", DualContouringOverlap[mode])
    target, setup, info = prepare(scope, case, resolution, mode)
    rounds = int(os.getenv("GEMPY_CONTACT_BENCHMARK_ROUNDS", "5"))
    result = benchmark.pedantic(target, setup=setup, rounds=rounds, iterations=1, warmup_rounds=0)
    meshes = result.dc_meshes if scope == "model" else result
    # Validation and accounting are deliberately outside the timed callable.
    assert meshes and any(len(m.edges) for m in meshes)
    for mesh in meshes:
        assert np.isfinite(mesh.vertices).all()
        assert mesh.edges.size == 0 or (mesh.edges.min() >= 0 and mesh.edges.max() < len(mesh.vertices))
    add_mesh_counts(info, meshes, "output")
    if scope == "model":
        info["leaf_cells"] = len(result.octrees_output[-1].grid.octree_grid.values)
    benchmark.extra_info.update(info)
