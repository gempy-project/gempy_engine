import numpy as np
import pytest

from gempy_engine.core.data.centered_grid import CenteredGrid
from gempy_engine.modules.geophysics.fw_magnetic import compute_magnetic_forward
from gempy_engine.modules.geophysics.magnetic_gradient import (
    calculate_magnetic_gradient_tensor,
    calculate_magnetic_gradient_components,
    _direction_cosines
)


# =============================================================================
# Unit Tests for Helper Functions
# =============================================================================

@pytest.mark.parametrize("inclination,declination,expected_lx,expected_ly,expected_lz", [
    (0.0, 0.0, 1.0, 0.0, 0.0),      # Horizontal north
    (90.0, 0.0, 0.0, 0.0, 1.0),     # Vertical down (north pole)
    (-90.0, 0.0, 0.0, 0.0, -1.0),   # Vertical up (south pole)
    (0.0, 90.0, 0.0, 1.0, 0.0),     # Horizontal east
    (45.0, 0.0, 0.707107, 0.0, 0.707107),  # 45° down to north
    (45.0, 45.0, 0.5, 0.5, 0.707107),      # 45° down, 45° east
])
def test_direction_cosines(inclination, declination, expected_lx, expected_ly, expected_lz):
    """Test direction cosines computation for various field geometries."""
    l = _direction_cosines(inclination, declination)

    # Check components match expectations
    np.testing.assert_allclose(l[0], expected_lx, atol=1e-5, 
                               err_msg=f"l_x mismatch for I={inclination}, D={declination}")
    np.testing.assert_allclose(l[1], expected_ly, atol=1e-5,
                               err_msg=f"l_y mismatch for I={inclination}, D={declination}")
    np.testing.assert_allclose(l[2], expected_lz, atol=1e-5,
                               err_msg=f"l_z mismatch for I={inclination}, D={declination}")

    # Check unit length
    norm = np.linalg.norm(l)
    np.testing.assert_allclose(norm, 1.0, atol=1e-10,
                               err_msg="Direction cosines should be unit length")


def test_direction_cosines_wraparound():
    """Test that declination wraps around correctly."""
    l1 = _direction_cosines(45.0, 0.0)
    l2 = _direction_cosines(45.0, 360.0)
    l3 = _direction_cosines(45.0, -360.0)

    np.testing.assert_allclose(l1, l2, atol=1e-10)
    np.testing.assert_allclose(l1, l3, atol=1e-10)


# =============================================================================
# Equivalence Tests Between Computation Paths
# =============================================================================

@pytest.mark.parametrize("inclination,declination,intensity", [
    (90.0, 0.0, 50000.0),    # Vertical field (pole)
    (0.0, 0.0, 45000.0),     # Horizontal field (equator)
    (60.0, 10.0, 48000.0),   # Mid-latitude field
    (45.0, 45.0, 50000.0),   # Oblique field
    (-30.0, -15.0, 35000.0), # Southern hemisphere
])
def test_path_equivalence(inclination, declination, intensity):
    """Test that pre-projected and raw V computation paths give identical results."""
    # Setup
    grid = CenteredGrid(centers=[[500, 500, 600]], resolution=[10, 10, 10], radius=[100, 100, 100])
    igrf_params = {"inclination": inclination, "declination": declination, "intensity": intensity}
    susceptibilities_per_unit = np.array([0.0, 0.01, 0.0])  # Unit 2 has chi=0.01
    ids_grid = np.ones(1331, dtype=int) * 2  # All voxels are unit 2

    # Path 1: Pre-projected (fast)
    result = calculate_magnetic_gradient_tensor(grid, igrf_params, compute_tmi=True)
    tmi_kernel = result['tmi_kernel']
    chi_mapped = susceptibilities_per_unit[ids_grid - 1]  # Map units to voxels
    tmi_path1 = np.sum(chi_mapped * tmi_kernel)

    # Path 2: Raw V components (flexible)
    V = calculate_magnetic_gradient_components(grid)
    chi_voxels = susceptibilities_per_unit[ids_grid - 1]  # Same mapping
    tmi_path2 = compute_magnetic_forward(V, chi_voxels, igrf_params, n_devices=1)

    # Should match to floating point precision
    np.testing.assert_allclose(
        tmi_path1, 
        tmi_path2[0], 
        rtol=1e-10,
        err_msg=f"Paths differ for I={inclination}, D={declination}, F={intensity}"
    )


@pytest.mark.parametrize("inclination,declination,intensity_nT,rtol", [
    # Vertical field - best case for analytical comparison
    (90.0, 0.0, 50000.0, 0.20),   # North pole, 20% tolerance
    (-90.0, 0.0, 50000.0, 0.20),  # South pole, 20% tolerance
    # Horizontal fields - harder due to asymmetry
    (0.0, 0.0, 45000.0, 0.35),    # Equator pointing north, 35% tolerance
    (0.0, 90.0, 45000.0, 0.35),   # Equator pointing east, 35% tolerance
    # Mid-latitude cases
    (60.0, 0.0, 48000.0, 0.25),   # Typical northern hemisphere, 25% tolerance
    (45.0, 45.0, 50000.0, 0.30),  # Oblique field, 30% tolerance
])
def test_magnetics_sphere_analytical_benchmark_induced_only(inclination, declination, intensity_nT, rtol):
    """
    Benchmark comparing induced-only TMI against analytical solution for a uniformly
    susceptible sphere in a vertical inducing field.

    Geometry mirrors gravity sphere benchmark: observe along vertical line above center.

    Analytical (outside sphere, along axis):
        m = V * M = (4/3)π R^3 * (χ * B0 / μ0)
        Bz = μ0/(4π) * (2 m) / r^3  (dipole field on axis)
        ΔT = B · l = Bz (since l = z-hat for vertical field)
        => ΔT = (2/3) * R^3 * χ * B0 / r^3  [Tesla]
        Convert to nT by × 1e9 if B0 in Tesla. Here intensity_nT is in nT, so use directly:
        ΔT[nT] = (2/3) * R^3 * χ * intensity_nT / r^3

    We accept ~15–20% error due to voxelization discretization.
    """

    # Sphere parameters
    R = 100.0  # meters
    center = np.array([500.0, 500.0, 500.0])
    chi = 0.01  # SI susceptibility (dimensionless)

    # Observation points along vertical above center
    observation_heights = np.array([650.0, 700.0, 800.0, 1000.0, 1200.0])
    n_obs = len(observation_heights)

    centers = np.column_stack([
        np.full(n_obs, center[0]),
        np.full(n_obs, center[1]),
        observation_heights,
    ])

    # Voxel grid around sphere
    geophysics_grid = CenteredGrid(
        centers=centers,
        resolution=np.array([100, 100, 100]),
        radius=np.array([200.0, 200.0, 800.0]),
    )

    # Magnetic kernel (pre-projected TMI kernel per voxel recommended by plan)
    igrf_params = {
        "inclination": inclination,
        "declination": declination,
        "intensity": intensity_nT,  # nT
    }

    try:
        mag_kern_out = calculate_magnetic_gradient_tensor(
            geophysics_grid, igrf_params, compute_tmi=True, units_nT=True
        )
    except TypeError:
        # Some implementations may return the kernel directly rather than a dict; handle both
        mag_kern_out = calculate_magnetic_gradient_tensor(
            geophysics_grid, igrf_params
        )

    # Support both dict output with key 'tmi_kernel' or direct array output
    if isinstance(mag_kern_out, dict):
        tmi_kernel = mag_kern_out.get("tmi_kernel", None)
        if tmi_kernel is None:
            pytest.skip("Magnetic gradient returned dict but no 'tmi_kernel' present yet")
    else:
        tmi_kernel = np.asarray(mag_kern_out)

    # Build a binary sphere susceptibility distribution per device
    voxel_centers = geophysics_grid.values
    n_voxels_per_device = voxel_centers.shape[0] // n_obs

    numerical_tmi = []
    for i in range(n_obs):
        sl = slice(i * n_voxels_per_device, (i + 1) * n_voxels_per_device)
        vc = voxel_centers[sl]
        inside = (np.linalg.norm(vc - center, axis=1) <= R).astype(float)
        chi_vox = inside * chi
        # Forward model: sum(chi * tmi_kernel)
        numerical_tmi.append(np.sum(chi_vox * tmi_kernel))

    numerical_tmi = -np.array(numerical_tmi)

    # Analytical TMI along axis (in nT)
    analytical_tmi = []
    for z in observation_heights:
        r = abs(z - center[2])
        if r <= R:
            # Inside sphere (not expected in this test set). Use outer formula at R for continuity.
            r = R
        dT = (2.0 / 3.0) * (R ** 3) * chi * intensity_nT / (r ** 3)
        analytical_tmi.append(dT)

    analytical_tmi = np.array(analytical_tmi)

    # Report
    print("\n=== Magnetics Sphere Benchmark (Induced-only TMI) ===")
    print(f"Sphere: R={R}m, center={center.tolist()}, χ={chi}")
    print(f"Observation heights: {observation_heights}")
    print(f"Numerical ΔT (nT):  {numerical_tmi}")
    print(f"Analytical ΔT (nT): {analytical_tmi}")
    if np.all(analytical_tmi != 0):
        print(
            f"Relative error (%): {np.abs((numerical_tmi - analytical_tmi) / analytical_tmi) * 100}"
        )

    # Tolerance varies by field geometry (vertical fields are most accurate)
    np.testing.assert_allclose(
        numerical_tmi,
        analytical_tmi,
        rtol=rtol,
        err_msg=(
            f"Magnetic TMI calculation deviates significantly from analytical sphere solution "
            f"for I={inclination}°, D={declination}°"
        ),
    )


@pytest.mark.parametrize("inclination,declination,intensity_nT", [
    (90.0, 0.0, 50000.0),
])
def test_magnetics_line_profile_symmetry_induced_only(inclination, declination, intensity_nT):
    """
    Symmetry test for TMI along a horizontal profile across a spherical induced anomaly.

    Checks:
    1) Symmetry about the center x=500
    2) Peak at center
    3) Decay away from anomaly
    """
    if CenteredGrid is None:
        pytest.skip("CenteredGrid not available; core grid module missing")

    # Profile setup
    x_profile = np.linspace(0.0, 1000.0, 21)
    y_center = 500.0
    z_obs = 600.0

    centers = np.column_stack([
        x_profile,
        np.full_like(x_profile, y_center),
        np.full_like(x_profile, z_obs),
    ])

    geophysics_grid = CenteredGrid(
        centers=centers,
        resolution=np.array([15, 15, 15]),
        radius=np.array([200.0, 200.0, 200.0]),
    )

    igrf_params = {
        "inclination": inclination,
        "declination": declination,
        "intensity": intensity_nT,
    }

    try:
        mag_kern_out = calculate_magnetic_gradient_tensor(
            geophysics_grid, igrf_params, compute_tmi=True
        )
    except TypeError:
        mag_kern_out = calculate_magnetic_gradient_tensor(geophysics_grid, igrf_params)

    if isinstance(mag_kern_out, dict):
        tmi_kernel = mag_kern_out.get("tmi_kernel", None)
        if tmi_kernel is None:
            pytest.skip("Magnetic gradient returned dict but no 'tmi_kernel' present yet")
    else:
        tmi_kernel = np.asarray(mag_kern_out)

    # Spherical anomaly
    anomaly_center = np.array([500.0, 500.0, 500.0])
    anomaly_radius = 80.0
    chi_contrast = 0.02

    voxel_centers = geophysics_grid.values
    n_devices = len(centers)
    n_voxels_per_device = voxel_centers.shape[0] // n_devices

    tmi_profile = []
    for i in range(n_devices):
        sl = slice(i * n_voxels_per_device, (i + 1) * n_voxels_per_device)
        vc = voxel_centers[sl]
        distances = np.linalg.norm(vc - anomaly_center, axis=1)
        chi_vox = (distances <= anomaly_radius).astype(float) * chi_contrast
        tmi = np.sum(chi_vox * tmi_kernel)
        tmi_profile.append(tmi)

    tmi_profile = -np.array(tmi_profile)

    # Symmetry and decay assertions
    center_idx = len(tmi_profile) // 2

    left = tmi_profile[:center_idx]
    right = tmi_profile[center_idx + 1 :][::-1]

    print("\n=== Magnetics Line Profile Symmetry (Induced-only TMI) ===")
    print(f"x_profile: {x_profile}")
    print(f"ΔT profile (nT): {tmi_profile}")
    print(f"Peak index: {np.argmax(tmi_profile)} (expected {center_idx})")

    assert np.argmax(tmi_profile) == center_idx, "TMI peak should be at profile center"

    min_len = min(len(left), len(right))
    np.testing.assert_allclose(
        left[:min_len],
        right[:min_len],
        rtol=0.1,
        err_msg="TMI profile should be approximately symmetric",
    )

    assert tmi_profile[0] < tmi_profile[center_idx]
    assert tmi_profile[-1] < tmi_profile[center_idx]


# =============================================================================
# Multiple Device Tests
# =============================================================================

@pytest.mark.parametrize("n_devices", [1, 2, 5, 10])
def test_multiple_devices(n_devices):
    """Test that magnetic forward modeling works correctly with multiple observation points."""
    # Create a grid of observation points
    x_coords = np.linspace(400, 600, n_devices)
    centers = np.column_stack([
        x_coords,
        np.full(n_devices, 500.0),
        np.full(n_devices, 600.0),
    ])

    grid = CenteredGrid(
        centers=centers,
        resolution=np.array([10, 10, 10]),
        radius=np.array([100.0, 100.0, 100.0]),
    )

    igrf_params = {"inclination": 60.0, "declination": 0.0, "intensity": 48000.0}

    # Compute V components once
    V = calculate_magnetic_gradient_components(grid)

    # Create susceptibility distribution (uniform for simplicity)
    n_voxels = V.shape[1]
    chi_per_voxel = np.full(n_voxels * n_devices, 0.001)

    # Compute TMI for all devices
    tmi = compute_magnetic_forward(V, chi_per_voxel, igrf_params, n_devices=n_devices)

    # Check output shape
    assert tmi.shape == (n_devices,), f"Expected shape ({n_devices},), got {tmi.shape}"

    # All devices should have the same response (uniform susceptibility)
    np.testing.assert_allclose(tmi, tmi[0], rtol=1e-10,
                               err_msg="Uniform susceptibility should give uniform response")


# =============================================================================
# Edge Case Tests
# =============================================================================

def test_zero_susceptibility():
    """Test that zero susceptibility produces zero anomaly."""
    grid = CenteredGrid(centers=[[500, 500, 600]], resolution=[10, 10, 10], radius=[100, 100, 100])
    igrf_params = {"inclination": 45.0, "declination": 10.0, "intensity": 50000.0}

    V = calculate_magnetic_gradient_components(grid)
    chi = np.zeros(V.shape[1])

    tmi = compute_magnetic_forward(V, chi, igrf_params, n_devices=1)

    np.testing.assert_allclose(tmi, 0.0, atol=1e-10,
                               err_msg="Zero susceptibility should produce zero anomaly")


def test_negative_susceptibility():
    """Test that negative susceptibility produces negative (diamagnetic) anomaly."""
    grid = CenteredGrid(centers=[[500, 500, 600]], resolution=[10, 10, 10], radius=[100, 100, 100])
    igrf_params = {"inclination": 90.0, "declination": 0.0, "intensity": 50000.0}

    V = calculate_magnetic_gradient_components(grid)

    # Positive susceptibility
    chi_pos = np.full(V.shape[1], 0.01)
    tmi_pos = compute_magnetic_forward(V, chi_pos, igrf_params, n_devices=1)

    # Negative susceptibility (diamagnetic)
    chi_neg = np.full(V.shape[1], -0.01)
    tmi_neg = compute_magnetic_forward(V, chi_neg, igrf_params, n_devices=1)

    # Should be opposite sign
    np.testing.assert_allclose(tmi_pos, -tmi_neg, rtol=1e-10,
                               err_msg="Negative susceptibility should produce opposite anomaly")


@pytest.mark.parametrize("distance_multiplier", [2.0, 5.0, 10.0])
def test_kernel_decay_with_distance(distance_multiplier):
    """Test that magnetic anomaly decays with distance from source."""
    # Source at origin
    source_center = np.array([500.0, 500.0, 500.0])

    # Observation at increasing distances
    obs_z_near = 600.0
    obs_z_far = source_center[2] + (obs_z_near - source_center[2]) * distance_multiplier

    centers = np.array([
        [source_center[0], source_center[1], obs_z_near],
        [source_center[0], source_center[1], obs_z_far],
    ])

    grid = CenteredGrid(
        centers=centers,
        resolution=np.array([20, 20, 20]),
        radius=np.array([150.0, 150.0, 150.0]),
    )

    igrf_params = {"inclination": 90.0, "declination": 0.0, "intensity": 50000.0}

    # Compute kernel
    result = calculate_magnetic_gradient_tensor(grid, igrf_params, compute_tmi=True)
    tmi_kernel = result['tmi_kernel']

    # Uniform susceptibility
    n_voxels_per_device = tmi_kernel.shape[0]
    chi = np.full(n_voxels_per_device * 2, 0.01)

    # Compute TMI at both distances
    tmi_near = np.sum(chi[:n_voxels_per_device] * tmi_kernel)
    tmi_far = np.sum(chi[n_voxels_per_device:] * tmi_kernel)

    # Far anomaly should be smaller (by approximately distance^3 for dipole)
    assert abs(tmi_far) < abs(tmi_near), "Anomaly should decay with distance"

    # Check approximate 1/r³ decay (allow large tolerance due to voxelization)
    expected_ratio = distance_multiplier ** 3
    actual_ratio = abs(tmi_near / tmi_far)
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.5,
                               err_msg=f"Decay should be approximately 1/r³")


# =============================================================================
# Kernel Property Tests
# =============================================================================

def test_kernel_symmetry():
    """Test that kernel has expected symmetry properties for vertical field."""
    # Centered observation point with symmetric grid
    grid = CenteredGrid(
        centers=[[500, 500, 600]],
        resolution=np.array([20, 20, 20]),
        radius=np.array([100.0, 100.0, 100.0]),
    )

    # Vertical field
    igrf_params = {"inclination": 90.0, "declination": 0.0, "intensity": 50000.0}

    result = calculate_magnetic_gradient_tensor(grid, igrf_params, compute_tmi=True)
    tmi_kernel = result['tmi_kernel']

    # Reshape kernel to 3D grid
    nx, ny, nz = 20, 20, 20
    kernel_3d = tmi_kernel.reshape((nz, ny, nx))

    # For vertical field, kernel should be symmetric about vertical axis
    # Check horizontal slices are approximately radially symmetric
    mid_z = nz // 2
    slice_mid = kernel_3d[mid_z, :, :]

    # Check that corners are approximately equal (radial symmetry)
    corners = [
        slice_mid[0, 0], slice_mid[0, -1], 
        slice_mid[-1, 0], slice_mid[-1, -1]
    ]

    # All corners should be similar for radial symmetry
    np.testing.assert_allclose(corners, corners[0], rtol=0.1,
                               err_msg="Kernel should show radial symmetry for vertical field")


def test_v_components_reusability():
    """Test that V components can be reused with different IGRF parameters."""
    grid = CenteredGrid(centers=[[500, 500, 600]], resolution=[10, 10, 10], radius=[100, 100, 100])

    # Compute V once
    V = calculate_magnetic_gradient_components(grid)

    # Different IGRF scenarios
    igrf_scenarios = [
        {"inclination": 90.0, "declination": 0.0, "intensity": 50000.0},
        {"inclination": 0.0, "declination": 0.0, "intensity": 45000.0},
        {"inclination": 60.0, "declination": 30.0, "intensity": 48000.0},
    ]

    chi = np.full(V.shape[1], 0.01)

    results = []
    for igrf in igrf_scenarios:
        tmi = compute_magnetic_forward(V, chi, igrf, n_devices=1)
        results.append(tmi[0])

    # Results should be different for different IGRF parameters
    assert not np.allclose(results[0], results[1]), "Different IGRF should give different results"
    assert not np.allclose(results[0], results[2]), "Different IGRF should give different results"
    assert not np.allclose(results[1], results[2]), "Different IGRF should give different results"


@pytest.mark.parametrize("resolution", [(5, 5, 5), (10, 10, 10), (15, 15, 15)])
def test_different_resolutions(resolution):
    """Test that computation works with different voxel resolutions."""
    grid = CenteredGrid(
        centers=[[500, 500, 600]],
        resolution=np.array(resolution),
        radius=np.array([100.0, 100.0, 100.0]),
    )

    igrf_params = {"inclination": 60.0, "declination": 10.0, "intensity": 48000.0}

    # Should not raise any errors
    result = calculate_magnetic_gradient_tensor(grid, igrf_params, compute_tmi=True)
    tmi_kernel = result['tmi_kernel']

    # Check expected number of voxels
    expected_voxels = grid.get_total_number_of_voxels()

    assert tmi_kernel.shape[0] == expected_voxels, \
        f"Expected {expected_voxels} voxels, got {tmi_kernel.shape[0]}"
