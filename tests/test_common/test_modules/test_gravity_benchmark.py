
def test_gravity_sphere_analytical_benchmark():
    """
    Benchmark test comparing gravity calculation against analytical solution for a sphere.
    
    A homogeneous sphere has an analytical gravity solution:
    g_z = (4/3) * π * G * R³ * Δρ / r² for r > R (outside)
    
    This provides a simple validation benchmark for the gravity implementation.
    """
    import numpy as np
    from gempy_engine.core.data.centered_grid import CenteredGrid
    from gempy_engine.core.data.geophysics_input import GeophysicsInput
    from gempy_engine.modules.geophysics.gravity_gradient import calculate_gravity_gradient

    # Sphere parameters
    sphere_radius = 100  # meters
    sphere_center = np.array([500, 500, 500])  # center of model domain
    density_contrast = 0.5  # g/cm³ (density difference from background)

    # Observation points along a line above the sphere
    observation_heights = np.array([600, 700, 800, 1000, 1200])  # z coordinates
    n_observations = len(observation_heights)

    centers = np.column_stack([
            np.full(n_observations, sphere_center[0]),  # x = sphere center
            np.full(n_observations, sphere_center[1]),  # y = sphere center
            observation_heights
    ])

    # Create fine voxelized grid around sphere to approximate it
    voxel_resolution = np.array([20, 20, 20])  # voxels per dimension
    grid_radius = np.array([150, 150, 150])  # capture the sphere

    geophysics_grid = CenteredGrid(
        centers=centers,
        resolution=voxel_resolution,
        radius=grid_radius
    )

    # Calculate gravity gradient kernel
    gravity_gradient = calculate_gravity_gradient(geophysics_grid, ugal=True)

    # Create synthetic sphere density model
    # For each voxel center, check if it's inside the sphere
    voxel_centers = geophysics_grid.values
    distances_from_center = np.linalg.norm(voxel_centers - sphere_center, axis=1)

    # Binary mask: 1 inside sphere, 0 outside
    inside_sphere = (distances_from_center <= sphere_radius).astype(float)

    # For this test, we need to assign densities per voxel, not per geological ID
    # This requires a modified workflow or direct multiplication
    # Simplified approach: assume density_per_voxel = inside_sphere * density_contrast

    # Calculate numerical gravity
    G = 6.674e-3  # ugal units
    n_voxels_per_device = len(voxel_centers) // n_observations

    gravity_numerical = []
    for i in range(n_observations):
        voxel_slice = slice(i * n_voxels_per_device, (i + 1) * n_voxels_per_device)
        density_distribution = inside_sphere[voxel_slice] * density_contrast
        grav = np.sum(density_distribution * gravity_gradient)
        gravity_numerical.append(grav)


    gravity_numerical = np.array(gravity_numerical)

    # Calculate analytical solution
    gravity_analytical = []
    for height in observation_heights:
        r = height - sphere_center[2]  # distance from sphere center

        if r > sphere_radius:
            # Outside the sphere: point mass approximation
            # g_z = G * M / r² where M = (4/3)πR³ρ
            # In ugal: need to convert properly
            volume = (4/3) * np.pi * sphere_radius**3
            mass = volume * density_contrast  # in g (if volume in cm³)
            # Actually, we need to be careful with units
            # Let's use the exact formula in consistent units
            g_analytical = (4/3) * np.pi * G * (sphere_radius**3) * density_contrast / (r**2)
        else:
            # Inside the sphere (shouldn't happen in this test)
            g_analytical = (4/3) * np.pi * G * density_contrast * r

        gravity_analytical.append(g_analytical)

    gravity_analytical = np.array(gravity_analytical)

    # Print comparison for inspection
    print("\n=== Sphere Gravity Benchmark ===")
    print(f"Observation Heights: {observation_heights}")
    print(f"Numerical: {gravity_numerical}")
    print(f"Analytical: {gravity_analytical}")
    print(f"Relative Error: {np.abs((gravity_numerical - gravity_analytical) / gravity_analytical) * 100}%")

    # Allow ~5-10% error due to voxelization
    # The sphere is approximated by rectangular voxels, so exact match is not expected
    np.testing.assert_allclose(
        gravity_numerical,
        gravity_analytical,
        rtol=0.15,  # 15% relative tolerance
        err_msg="Gravity calculation deviates significantly from analytical sphere solution"
    )
