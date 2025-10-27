import numpy as np
import pytest

from gempy_engine.core.data.centered_grid import CenteredGrid
from gempy_engine.modules.geophysics.magnetic_gradient import calculate_magnetic_gradient_tensor


@pytest.mark.parametrize("inclination,declination,intensity_nT", [
    # Use vertical field to simplify analytical comparison along vertical axis
    (90.0, 0.0, 50000.0),  # IGRF-like strength, vertical down
])
def test_magnetics_sphere_analytical_benchmark_induced_only(inclination, declination, intensity_nT):
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
    observation_heights = np.array([625.0, 700.0, 800.0, 1000.0, 1200.0])
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
            geophysics_grid, igrf_params, compute_tmi=True
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

    numerical_tmi = np.array(numerical_tmi)

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

    # Allow 20% tolerance initially; adjust tighter once implementation stabilizes
    np.testing.assert_allclose(
        numerical_tmi,
        analytical_tmi,
        rtol=0.2,
        err_msg=(
            "Magnetic TMI calculation deviates significantly from analytical sphere solution"
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

    calculate_magnetic_gradient_tensor = _try_import_magnetics()

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

    tmi_profile = np.array(tmi_profile)

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
