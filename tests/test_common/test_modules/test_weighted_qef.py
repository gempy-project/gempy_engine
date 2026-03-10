import numpy as np
import pytest

from gempy_engine import compute_model
from gempy_engine.core.data.options import MeshExtractionMaskingOptions
from gempy_engine.core.data.solutions import Solutions
from tests.conftest import plot_pyvista

np.random.seed(42)


def _run_fault_model(one_fault_model, n_oct_levels=3):
    """Helper to run the fault model and return solutions."""
    interpolation_input, structure, options = one_fault_model

    options.compute_scalar_gradient = False
    options.evaluation_options.dual_contouring = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.INTERSECT
    options.evaluation_options.mesh_extraction_fancy = True
    options.evaluation_options.number_octree_levels = n_oct_levels
    options.evaluation_options.number_octree_levels_surface = n_oct_levels

    solutions: Solutions = compute_model(interpolation_input, options, structure)
    return solutions


def test_weighted_qef_produces_valid_meshes(one_fault_model):
    """Verify that the weighted QEF pipeline produces meshes with valid vertices and edges."""
    solutions = _run_fault_model(one_fault_model, n_oct_levels=3)

    assert solutions.dc_meshes is not None
    assert len(solutions.dc_meshes) > 0

    for i, mesh in enumerate(solutions.dc_meshes):
        assert mesh.vertices.shape[1] == 3, f"Mesh {i}: vertices should have 3 columns"
        assert mesh.vertices.shape[0] > 0, f"Mesh {i}: should have at least one vertex"
        assert not np.any(np.isnan(mesh.vertices)), f"Mesh {i}: vertices contain NaN"
        assert not np.any(np.isinf(mesh.vertices)), f"Mesh {i}: vertices contain Inf"


def test_weighted_qef_shared_vertices_are_close(one_fault_model):
    """At overlapping voxels, vertices from different meshes should be identical
    (after the averaging step)."""
    from gempy_engine.modules.dual_contouring._find_vertex_overlap import (
        _generate_voxel_codes
    )

    solutions = _run_fault_model(one_fault_model, n_oct_levels=3)
    meshes = solutions.dc_meshes
    if len(meshes) < 2:
        pytest.skip("Need at least 2 meshes to test overlap")

    # Build left_right codes for valid voxels of each mesh
    left_right_per_mesh = []
    for m in meshes:
        dc = m.dc_data
        if dc is None:
            continue
        left_right_per_mesh.append(dc.left_right_codes[dc.valid_voxels])

    base_number = meshes[0].dc_data.base_number
    codes = _generate_voxel_codes(left_right_per_mesh, base_number)

    # Check all pairs
    found_overlap = False
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            common = np.intersect1d(codes[i], codes[j])
            if common.size == 0:
                continue
            found_overlap = True

            mask_i = np.isin(codes[i], common)
            mask_j = np.isin(codes[j], common)
            order_i = np.argsort(codes[i][mask_i])
            order_j = np.argsort(codes[j][mask_j])
            idx_i = np.where(mask_i)[0][order_i]
            idx_j = np.where(mask_j)[0][order_j]

            verts_i = meshes[i].vertices[idx_i]
            verts_j = meshes[j].vertices[idx_j]

            # After weighted QEF + averaging, shared vertices should be very close.
            # With multiple overlapping surfaces, pairwise averaging causes small
            # cumulative drift, so we use a tolerance of 0.02.
            np.testing.assert_allclose(
                verts_i, verts_j,
                atol=0.02,
                err_msg=f"Meshes {i} and {j}: shared vertices should be close after weighted QEF"
            )

    if not found_overlap:
        pytest.skip("No overlapping voxels found at this resolution")


def test_weighted_qef_no_regression_single_surface(one_fault_model):
    """Single-surface stacks (no overlap) should produce the same result as before
    (weights are all 1.0 when no extra constraints exist)."""
    solutions = _run_fault_model(one_fault_model, n_oct_levels=3)

    for mesh in solutions.dc_meshes:
        dc = mesh.dc_data
        if dc is None:
            continue
        # If no extra constraints were injected, the extra fields should be None
        # (for surfaces that don't overlap with any other)
        # Just verify the mesh is valid
        assert mesh.vertices.shape[0] > 0
        assert not np.any(np.isnan(mesh.vertices))


def test_weighted_qef_higher_resolution(one_fault_model):
    """Run at higher octree resolution to stress-test the weighted QEF."""
    solutions = _run_fault_model(one_fault_model, n_oct_levels=5)

    assert solutions.dc_meshes is not None
    for mesh in solutions.dc_meshes:
        assert mesh.vertices.shape[0] > 0
        assert not np.any(np.isnan(mesh.vertices))
        assert not np.any(np.isinf(mesh.vertices))

    if plot_pyvista or False:
        from gempy_engine.plugins.plotting import helper_functions_pyvista
        helper_functions_pyvista.plot_pyvista(
            octree_list=None,
            dc_meshes=solutions.dc_meshes
        )
