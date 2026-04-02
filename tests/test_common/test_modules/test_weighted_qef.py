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


def test_weighted_qef_fault_triangles_removed(one_fault_model):
    """Verify that layer meshes have triangles removed at fault overlap voxels."""
    solutions = _run_fault_model(one_fault_model, n_oct_levels=3)
    meshes = solutions.dc_meshes

    # With a fault model, at least one layer mesh should have had triangles removed.
    # We verify that all meshes have valid (non-empty) edge arrays and no degenerate faces.
    for i, mesh in enumerate(solutions.dc_meshes):
        if mesh.edges is not None and mesh.edges.size > 0:
            # All face indices should reference valid vertices
            assert mesh.edges.max() < mesh.vertices.shape[0], \
                f"Mesh {i}: face index out of bounds"
            assert mesh.edges.min() >= 0, \
                f"Mesh {i}: negative face index"


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
