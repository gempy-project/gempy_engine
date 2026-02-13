import numpy as np
import pytest
from gempy_engine import compute_model
from gempy_engine.core.data.options import MeshExtractionMaskingOptions
from gempy_engine.modules.faults.finite_faults import FiniteFault, TaperType, project_points_onto_surface
from tests.conftest import plot_pyvista


@pytest.fixture
def finite_fault_setup(one_fault_model):
    from gempy_engine.modules.weights_cache.weights_cache_interface import WeightCache
    WeightCache.initialize_cache_dir()

    interpolation_input, structure, options = one_fault_model
    num_octree = 5
    options.evaluation_options.number_octree_levels = num_octree
    options.evaluation_options.number_octree_levels_surface = num_octree
    options.evaluation_options.mesh_extraction = True
    options.evaluation_options.mesh_extraction_masking_options = MeshExtractionMaskingOptions.RAW
    options.evaluation_options.compute_scalar_gradient = True

    # Run the model to get the fault surface
    solutions = compute_model(interpolation_input, options, structure)

    return solutions, interpolation_input, structure, options


def test_finite_fault_integration_suite(finite_fault_setup):
    solutions, interpolation_input, structure, options = finite_fault_setup

    # Get the fault mesh (first stack is the fault)
    fault_mesh = solutions.dc_meshes[0]

    # Get the scalar field and gradients for the fault (stack 0)
    fault_output = solutions.octrees_output[0].outputs_centers[0]
    grid_points = fault_output.grid.values
    scalar_field = fault_output.exported_fields.scalar_field
    gx = fault_output.exported_fields.gx_field
    gy = fault_output.exported_fields.gy_field
    gz = fault_output.exported_fields.gz_field

    # Define a center for the finite fault (mean of mesh vertices)
    center = np.mean(fault_mesh.vertices, axis=0)  # * This should be input

    # Find normal at center
    dist_to_center = np.linalg.norm(grid_points - center, axis=1) # * Points to evaluate
    center_idx = np.argmin(dist_to_center)
    normal = np.array([gx[center_idx], gy[center_idx], gz[center_idx]])

    # Define some spline configurations
    cp_bell = np.array([
            [0.0, 1.0],
            [0.2, 0.95],
            [0.5, 0.5],
            [0.8, 0.05],
            [1.0, 0.0]
    ])

    cp_plateau = np.array([  # * This is the nicest
            [0.0, 1.0],
            [0.6, 1.0],
            [0.8, 0.5],
            [1.0, 0.0]
    ])

    # Define various FiniteFault configurations
    configs = [
            FiniteFault(center=center, strike_radius=0.8, dip_radius=0.4, taper=TaperType.CUBIC),
            FiniteFault(center=center, strike_radius=(1.0, 0.5), dip_radius=0.6, taper=TaperType.QUADRATIC),
            FiniteFault(center=center, strike_radius=0.7, dip_radius=0.7, taper=TaperType.SPLINE, spline_control_points=cp_bell),
            FiniteFault(center=center, strike_radius=0.8, dip_radius=0.4, taper=TaperType.CUBIC, rotation=45),
            FiniteFault(center=center, strike_radius=0.8, dip_radius=1, taper=TaperType.SPLINE, spline_control_points=cp_plateau, rotation=30),
            FiniteFault(center=center, strike_radius=(1.2,0.8), dip_radius=(2,1), taper=TaperType.SPLINE, spline_control_points=cp_plateau, rotation=30)
    ]

    if not plot_pyvista:
        pytest.skip("PyVista plotting is disabled.")

    import pyvista as pv

    for i, ff in enumerate(configs):
        p = pv.Plotter(title=f"Finite Fault Config {i}: {ff.taper.name}")

        # 1. Plot the fault mesh with slip multiplier as texture
        mesh_vertices = fault_mesh.vertices
        slip_mesh = ff.calculate_slip(
            points=mesh_vertices, 
            normal=normal # Here is the normal used
        )

        dual_mesh = pv.PolyData(mesh_vertices, np.insert(fault_mesh.edges, 0, 3, axis=1).ravel())
        dual_mesh["slip"] = slip_mesh
        p.add_mesh(dual_mesh, scalars="slip", cmap="viridis", show_edges=True, opacity=0.8)

        # 2. Projection Lines
        # Pick a few points near the fault to show projection
        target_val = fault_output.scalar_field_at_sp[0]
        mask = np.abs(scalar_field - target_val) < 1.5
        indices = np.where(mask)[0][::30]  # Subsample

        points_to_project = grid_points[indices]
        projected_points = project_points_onto_surface(
            points_to_project,
            scalar_field[indices],
            (gx[indices], gy[indices], gz[indices]),
            target_scalar_value=target_val
        )

        p.add_points(points_to_project, color="red", point_size=8, label="Original Points")
        p.add_points(projected_points, color="blue", point_size=8, label="Projected Points")

        for start, end in zip(points_to_project, projected_points):
            p.add_lines(np.array([start, end]), color="white", width=1)

        # 3. Local Frame (u, v, w) at center
        from gempy_engine.modules.faults.finite_faults import get_local_frame
        u, v, w = get_local_frame(normal, angle_deg=ff.rotation)

        origin = center
        p.add_arrows(origin, u, mag=0.5, color="cyan", label="Strike (u)")
        p.add_arrows(origin, v, mag=0.5, color="magenta", label="Dip (v)")
        p.add_arrows(origin, w, mag=0.5, color="yellow", label="Normal (w)")

        # 4. Volumetric slip (on grid)
        # Filter grid points by slip > 0.01 for visualization
        all_slip = ff.calculate_slip(grid_points, normal)
        mask_slip = all_slip > 0.01
        if np.any(mask_slip):
            grid_pv = pv.PolyData(grid_points[mask_slip])
            grid_pv["slip_vol"] = all_slip[mask_slip]
            p.add_mesh(grid_pv, scalars="slip_vol", point_size=4, cmap="viridis",
                       render_points_as_spheres=True, opacity=0.5)

        p.add_legend()
        p.show()


if __name__ == "__main__":
    # This allows running the file directly to see plots
    from tests.conftest import one_fault_model as one_fault_model_fixture

    # Mocking or getting the fixture would be complex here, 
    # better to run via pytest with --plot_pyvista
    pass
