from gempy_engine import compute_model
from gempy_engine.core.data.options import MeshExtractionMaskingOptions
from gempy_engine.modules.faults.finite_faults import (
    get_local_frame,
    get_ellipsoid_distance,
    project_points_onto_surface,
    cubic_hermite_taper,
    quadratic_taper,
    spline_taper
)
from tests.conftest import plot_pyvista
import numpy as np


# --- Phase 1: Local Coordinate System & Analytical Ellipsoid UV ---

def test_local_frame_orthogonal():
    normals = [
            np.array([1, 0, 0.0]),
            np.array([0, 1, 0.0]),
            np.array([1, 1, 1.0]),
            np.array([0, 0, 1.0])
    ]

    for n in normals:
        u, v, w = get_local_frame(n)
        # Check normalization
        assert np.isclose(np.linalg.norm(u), 1.0)
        assert np.isclose(np.linalg.norm(v), 1.0)
        assert np.isclose(np.linalg.norm(w), 1.0)

        # Check orthogonality
        assert np.isclose(np.dot(u, v), 0.0)
        assert np.isclose(np.dot(u, w), 0.0)
        assert np.isclose(np.dot(v, w), 0.0)


def test_ellipsoid_distance():
    center = np.array([0, 0, 0.0])
    u = np.array([1, 0, 0.0])
    v = np.array([0, 1, 0.0])
    w = np.array([0, 0, 1.0])
    a, b = 2.0, 1.0

    points = np.array([
            [0, 0, 0],  # Center
            [2, 0, 0],  # On strike boundary
            [0, 1, 0],  # On dip boundary
            [4, 0, 0],  # Outside
            [1, 0, 0]  # Inside
    ], dtype=float)

    distances = get_ellipsoid_distance(points, center, u, v, a, b)

    assert np.isclose(distances[0], 0.0)
    assert np.isclose(distances[1], 1.0)
    assert np.isclose(distances[2], 1.0)
    assert distances[3] > 1.0
    assert distances[4] < 1.0


# --- Phase 2: Point Projection ("Walking the Gradient") ---

def test_projection_on_plane():
    # Surface: F(x,y,z) = z - 5 = 0  => z = 5
    # grad(F) = [0, 0, 1]
    # NOTE: Since we added a 0.5 factor to handle GemPy's quadratic scalar fields, 
    # a single step on a linear field will only go halfway.
    points = np.array([
            [0, 0, 10.0],
            [1, 2, 0.0],
            [5, 5, 5.0]
    ])
    f_values = points[:, 2] - 5.0
    gx = np.zeros(3)
    gy = np.zeros(3)
    gz = np.ones(3)

    projected = project_points_onto_surface(points, f_values, (gx, gy, gz), target_scalar_value=0.0)

    # With 0.5 factor, it goes halfway:
    # [0, 0, 10] -> [0, 0, 7.5]
    # [1, 2, 0]  -> [1, 2, 2.5]
    # [5, 5, 5]  -> [5, 5, 5.0]
    expected = np.array([
            [0, 0, 7.5],
            [1, 2, 2.5],
            [5, 5, 5.0]
    ])

    assert np.allclose(projected, expected)


# --- Phase 3: Slip Tapering Functions ---

def test_taper_bounds():
    d = np.array([0, 0.5, 1.0, 1.5])

    s_cubic = cubic_hermite_taper(d)
    assert np.isclose(s_cubic[0], 1.0)
    assert 0 < s_cubic[1] < 1.0
    assert np.isclose(s_cubic[2], 0.0)
    assert np.isclose(s_cubic[3], 0.0)

    s_quad = quadratic_taper(d)
    assert np.isclose(s_quad[0], 1.0)
    assert 0 < s_quad[1] < 1.0
    assert np.isclose(s_quad[2], 0.0)
    assert np.isclose(s_quad[3], 0.0)


# --- Phase 4: Integration ---

def test_finite_fault_full_pipeline(one_fault_model):
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir()

    interpolation_input, structure, options = one_fault_model
    options.evaluation_options.number_octree_levels = 4
    options.evaluation_options.mesh_extraction = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW
    options.evaluation_options.compute_scalar_gradient = True

    # Run the model to get the fault surface
    solutions = compute_model(interpolation_input, options, structure)

    # Get the fault mesh (first stack is the fault)
    fault_mesh = solutions.dc_meshes[0]

    # Get the scalar field and gradients for the fault (stack 0)
    # octrees_output[0] is the root level (contains dense grid)
    # outputs_centers[0] is the output for stack 0 (the fault)
    fault_output = solutions.octrees_output[0].outputs_centers[0]
    grid = fault_output.grid

    scalar_field = fault_output.exported_fields.scalar_field
    gx = fault_output.exported_fields.gx_field
    gy = fault_output.exported_fields.gy_field
    gz = fault_output.exported_fields.gz_field

    # 1. Project grid points onto the fault surface
    # We'll use a subset of the grid points for speed
    points = grid.values
    projected_points = project_points_onto_surface(
        points, scalar_field, (gx, gy, gz), target_scalar_value=fault_output.scalar_field_at_sp[0]
    )

    # 2. Define local frame at the center of the fault
    # Let's pick a point on the fault as center
    center_idx = len(projected_points) // 2
    center = projected_points[center_idx]

    # Get the gradient at the center to define the normal
    normal = np.array([gx[center_idx], gy[center_idx], gz[center_idx]])
    u, v, w = get_local_frame(normal)

    # 3. Calculate UV ellipsoid distance and slip
    a, b = 1.0, 1.0  # radii in local units
    d = get_ellipsoid_distance(projected_points, center, u, v, a, b)
    slip_multiplier = cubic_hermite_taper(d)

    # 4. (Optional) Visualization
    if plot_pyvista or False:
        import pyvista as pv
        p = pv.Plotter()

        # Plot the fault mesh
        dual_mesh = pv.PolyData(fault_mesh.vertices, np.insert(fault_mesh.edges, 0, 3, axis=1).ravel())
        p.add_mesh(dual_mesh, color="white", opacity=0.5, show_edges=True)

        # Plot the slip multiplier on the grid
        grid_pv = pv.PolyData(points)
        grid_pv["slip"] = slip_multiplier
        # Filter to only show points near the fault for clarity
        mask = np.abs(scalar_field - fault_output.scalar_field_at_sp[0]) < 3
        if np.any(mask):
            extracted_grid = grid_pv.extract_points(mask)
            p.add_mesh(extracted_grid, scalars="slip", point_size=5, render_points_as_spheres=True)

        p.show()

    # Simple assertions to verify the pipeline
    assert len(slip_multiplier) == len(points)
    assert np.max(slip_multiplier) <= 1.0
    assert np.min(slip_multiplier) >= 0.0


def test_visualize_slip_on_fault_mesh(one_fault_model):
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir()

    interpolation_input, structure, options = one_fault_model
    options.evaluation_options.number_octree_levels = 4
    options.evaluation_options.mesh_extraction = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW
    options.evaluation_options.compute_scalar_gradient = True

    # Run the model to get the fault surface
    solutions = compute_model(interpolation_input, options, structure)

    # Get the fault mesh (first stack is the fault)
    fault_mesh = solutions.dc_meshes[0]

    # We need the gradient to find the center and normal
    # fault_output[0] is the root level (contains dense grid)
    fault_output = solutions.octrees_output[0].outputs_centers[0]

    # Calculate slip for the MESH VERTICES
    # To do this correctly, we need the gradient and scalar field at the mesh vertices
    # However, for a simple visualization test, we can approximate the center and normal
    # or interpolate the fields to the vertices. 
    # For now, let's use the center of the fault mesh.
    mesh_vertices = fault_mesh.vertices
    center = np.mean(mesh_vertices, axis=0)

    # Find the closest point in the grid to the center to get the normal
    grid_points = fault_output.grid.values
    dist_to_center = np.linalg.norm(grid_points - center, axis=1)
    center_idx = np.argmin(dist_to_center)

    gx = fault_output.exported_fields.gx_field
    gy = fault_output.exported_fields.gy_field
    gz = fault_output.exported_fields.gz_field

    normal = np.array([gx[center_idx], gy[center_idx], gz[center_idx]])
    u, v, w = get_local_frame(normal)

    # Calculate UV ellipsoid distance and slip for vertices
    a, b = 0.5, 0.5  # smaller radii to see the taper on the mesh
    d = get_ellipsoid_distance(mesh_vertices, center, u, v, a, b)
    slip_multiplier = cubic_hermite_taper(d)

    if plot_pyvista:
        import pyvista as pv
        p = pv.Plotter()
       
        # Plot the fault mesh with slip as texture
        dual_mesh = pv.PolyData(fault_mesh.vertices, np.insert(fault_mesh.edges, 0, 3, axis=1).ravel())
        dual_mesh["slip"] = slip_multiplier
        p.add_mesh(dual_mesh, scalars="slip", cmap="viridis", show_edges=True)

        p.show()

    assert len(slip_multiplier) == len(mesh_vertices)


def test_visualize_projected_points_on_fault(one_fault_model):
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir()

    interpolation_input, structure, options = one_fault_model
    options.evaluation_options.number_octree_levels = 4
    options.evaluation_options.mesh_extraction = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW
    options.evaluation_options.compute_scalar_gradient = True

    # Run the model
    solutions = compute_model(interpolation_input, options, structure)
    fault_mesh = solutions.dc_meshes[0]
    fault_output = solutions.octrees_output[0].outputs_centers[0]

    # Pick some points in a small box around the fault
    # We use the grid points and filter them
    grid_points = fault_output.grid.values
    scalar_field = fault_output.exported_fields.scalar_field
    target_val = fault_output.scalar_field_at_sp[0]

    # Filter points near the fault surface
    mask = np.abs(scalar_field - target_val) < 2.0
    # Subsample for clarity
    indices = np.where(mask)[0][::10]
    points_to_project = grid_points[indices]

    gx = fault_output.exported_fields.gx_field
    gy = fault_output.exported_fields.gy_field
    gz = fault_output.exported_fields.gz_field

    projected_points = project_points_onto_surface(
        points_to_project,
        scalar_field[indices],
        (gx[indices], gy[indices], gz[indices]),
        target_scalar_value=target_val
    )

    if plot_pyvista:
        import pyvista as pv
        p = pv.Plotter()

        # Plot the fault mesh
        dual_mesh = pv.PolyData(fault_mesh.vertices, np.insert(fault_mesh.edges, 0, 3, axis=1).ravel())
        p.add_mesh(dual_mesh, color="white", opacity=0.3, show_edges=True)

        # Plot original points
        p.add_points(points_to_project, color="red", point_size=10, label="Original Points")

        # Plot projected points
        p.add_points(projected_points, color="blue", point_size=10, label="Projected Points")

        # Draw lines between original and projected points
        for start, end in zip(points_to_project, projected_points):
            p.add_lines(np.array([start, end]), color="yellow", width=2)

        p.add_legend()
        p.show()

    assert len(projected_points) == len(points_to_project)


def test_visualize_spline_slip_on_fault_mesh(one_fault_model):
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir()

    interpolation_input, structure, options = one_fault_model
    options.evaluation_options.number_octree_levels = 4
    options.evaluation_options.mesh_extraction = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW
    options.evaluation_options.compute_scalar_gradient = True

    # Run the model to get the fault surface
    solutions = compute_model(interpolation_input, options, structure)

    # Get the fault mesh (first stack is the fault)
    fault_mesh = solutions.dc_meshes[0]

    # We need the gradient to find the center and normal
    # fault_output[0] is the root level (contains dense grid)
    fault_output = solutions.octrees_output[0].outputs_centers[0]

    mesh_vertices = fault_mesh.vertices
    center = np.mean(mesh_vertices, axis=0)

    # Find the closest point in the grid to the center to get the normal
    grid_points = fault_output.grid.values
    dist_to_center = np.linalg.norm(grid_points - center, axis=1)
    center_idx = np.argmin(dist_to_center)

    gx = fault_output.exported_fields.gx_field
    gy = fault_output.exported_fields.gy_field
    gz = fault_output.exported_fields.gz_field

    normal = np.array([gx[center_idx], gy[center_idx], gz[center_idx]])
    u, v, w = get_local_frame(normal)

    # --- Spline Configuration ---
    # Define a bell-shaped spline for slip tapering
    cp_bell = np.array([
        [0.0, 1.0],
        [0.2, 0.95],
        [0.5, 0.5],
        [0.8, 0.05],
        [1.0, 0.0]
    ])

    # Calculate UV ellipsoid distance and slip for vertices
    # Using different radii as requested
    a, b = 0.8, 0.4  
    d = get_ellipsoid_distance(mesh_vertices, center, u, v, a, b)
    slip_multiplier = spline_taper(d, cp_bell)

    if plot_pyvista:
        import pyvista as pv
        p = pv.Plotter()
       
        # Plot the fault mesh with spline-based slip as texture
        dual_mesh = pv.PolyData(fault_mesh.vertices, np.insert(fault_mesh.edges, 0, 3, axis=1).ravel())
        dual_mesh["slip"] = slip_multiplier
        p.add_mesh(dual_mesh, scalars="slip", cmap="magma", show_edges=True)
        p.add_title("Finite Fault Slip - Spline Taper (a=0.8, b=0.4)")

        p.show()

    assert len(slip_multiplier) == len(mesh_vertices)
    assert np.max(slip_multiplier) <= 1.02 # allow for small spline overshoot if any
    assert np.min(slip_multiplier) >= -0.02
