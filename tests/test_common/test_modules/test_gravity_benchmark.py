
def test_gravity_sphere_analytical_benchmark():
    """
    Benchmark test comparing gravity calculation against analytical solution for a sphere.
    
    Test geometry (side view):
    
                     z=1200 ●  observation point 5
                            |
                     z=1000 ●  observation point 4
                            |
                     z=800  ●  observation point 3
                            |
                     z=700  ●  observation point 2
                            |
                     z=600  ●  observation point 1
                     -------+------- z=500 (sphere top)
                            |
                         .-"●"-. 
                       .'   |   '.
                      /     |     \    Sphere: R=100m, Δρ=0.5 g/cm³
                     ;    z=500    ;   Center at (500, 500, 500)
                      \     ●     /
                       '.       .'
                         `-._.-'
                     -------+------- z=400 (sphere bottom)
                            |
                            ▼ gz (vertical gravity measured)
    
    All observations at x=500, y=500 (directly above sphere center).
    
    Analytical solution for vertical component:
      g_z = (4/3) * π * G * R³ * Δρ / z²    [for z > R]
    
    where z is the vertical distance from sphere center to observation point.
    
    This validates:
    - Voxelization accuracy (sphere approximated by rectangular voxels)
    - Gravity gradient kernel computation
    - Forward calculation (density × kernel summation)
    
    References:
    - Telford et al. (1990) Applied Geophysics, Section on gravity anomalies
    - https://sites.ualberta.ca/~unsworth/UA-classes/224/notes224/B/224B2-2006.pdf
    """
    import numpy as np
    from gempy_engine.core.data.centered_grid import CenteredGrid
    from gempy_engine.core.data.geophysics_input import GeophysicsInput
    from gempy_engine.modules.geophysics.gravity_gradient import calculate_gravity_gradient

    # Sphere parameters
    sphere_radius = 100  # meters
    sphere_center = np.array([500, 500, 500])  # center of model domain
    density_contrast = 0.5  # g/cm³ (density difference from background)

    # Observation points along a vertical line above the sphere center
    # All at x=500, y=500 (directly above sphere center)
    observation_heights = np.array([600, 700, 800, 1000, 1200])  # z coordinates
    n_observations = len(observation_heights)

    centers = np.column_stack([
            np.full(n_observations, sphere_center[0]),  # x = sphere center
            np.full(n_observations, sphere_center[1]),  # y = sphere center
            observation_heights
    ])

    # Create fine voxelized grid around sphere to approximate it
    voxel_resolution = np.array([100, 100, 100])  # voxels per dimension
    grid_radius = np.array([1500, 1500, 1500])  # capture the sphere

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

    # Calculate numerical gravity
    G = 6.674e-3  # ugal units (cm³·g⁻¹·s⁻²)
    n_voxels_per_device = len(voxel_centers) // n_observations

    gravity_numerical = []
    for i in range(n_observations):
        voxel_slice = slice(i * n_voxels_per_device, (i + 1) * n_voxels_per_device)
        density_distribution = inside_sphere[voxel_slice] * density_contrast
        grav = np.sum(density_distribution * gravity_gradient)
        gravity_numerical.append(grav)

    gravity_numerical = np.array(gravity_numerical)

    # Calculate analytical solution for vertical component
    # For a sphere observed directly above its center:
    # g_z = (4/3) * π * G * R³ * Δρ / z²
    # where z is the distance from sphere center to observation point
    
    gravity_analytical = []
    for height in observation_heights:
        z = height - sphere_center[2]  # vertical distance from sphere center
        
        # Analytical formula for gz (vertical component) of a sphere
        # Valid for observation points outside the sphere (z > R)
        if z > sphere_radius:
            # g_z = (4/3) * π * G * R³ * Δρ / z²
            g_analytical = (4.0/3.0) * np.pi * G * (sphere_radius**3) * density_contrast / (z**2)
        else:
            # Inside the sphere (shouldn't happen in this test)
            # g_z = (4/3) * π * G * Δρ * z
            g_analytical = (4.0/3.0) * np.pi * G * density_contrast * z
        
        gravity_analytical.append(g_analytical)

    gravity_analytical = -np.array(gravity_analytical)

    # Print comparison for inspection
    print("\n=== Sphere Gravity Benchmark (Vertical Component gz) ===")
    print(f"Sphere: R={sphere_radius}m, center={sphere_center}, Δρ={density_contrast} g/cm³")
    print(f"Observation Heights: {observation_heights}")
    print(f"Distances from center: {observation_heights - sphere_center[2]}")
    print(f"Numerical gz:  {gravity_numerical}")
    print(f"Analytical gz: {gravity_analytical}")
    print(f"Absolute Error: {np.abs(gravity_numerical - gravity_analytical)}")
    print(f"Relative Error: {np.abs((gravity_numerical - gravity_analytical) / gravity_analytical) * 100}%")

    # Allow ~10-15% error due to voxelization
    # The sphere is approximated by rectangular voxels, so exact match is not expected
    # Closer observation points have higher relative error due to discretization effects
    np.testing.assert_allclose(
        gravity_numerical,
        gravity_analytical,
        rtol=0.15,  # 15% relative tolerance
        err_msg="Gravity calculation deviates significantly from analytical sphere solution"
    )
    
    # Additional check: verify gravity decreases with distance (physical sanity)
    assert np.all(np.diff(gravity_numerical) < 0), "Gravity should decrease with increasing distance"
    assert np.all(np.diff(gravity_analytical) < 0), "Analytical gravity should decrease with distance"
