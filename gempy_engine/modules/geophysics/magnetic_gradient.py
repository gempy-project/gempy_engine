import numpy as np

from gempy_engine.core.data.centered_grid import CenteredGrid


def _direction_cosines(inclination_deg: float, declination_deg: float) -> np.ndarray:
    """Compute unit vector of Earth's field from inclination/declination.

    Convention:
    - Inclination I: positive downward from horizontal, in degrees [-90, 90]
    - Declination D: clockwise from geographic north toward east, in degrees [-180, 180]

    Returns unit vector l = [lx, ly, lz].
    """
    I = np.deg2rad(inclination_deg)
    D = np.deg2rad(declination_deg)
    cI = np.cos(I)
    sI = np.sin(I)
    cD = np.cos(D)
    sD = np.sin(D)
    # North (x), East (y), Down (z) convention
    l = np.array([cI * cD, cI * sD, sI], dtype=float)
    # Already unit length by construction, but normalize defensively
    n = np.linalg.norm(l)
    if n == 0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return l / n


def calculate_magnetic_gradient_tensor(
    centered_grid: CenteredGrid,
    igrf_params: dict,
    compute_tmi: bool = True,
    units_nT: bool = True,
):
    """
    Compute magnetic kernels for voxelized forward modeling around each observation point.

    This MVP implementation provides a pre-projected Total Magnetic Intensity (TMI) scalar kernel
    per voxel using a point-dipole approximation for each voxel. It mirrors the gravity workflow by
    returning a per-device kernel that can be reused across devices with identical grid geometry.

    Physics (induced-only, per voxel v considered as a dipole at its center):
        m = chi * V * F/μ0 * l
        B(r) = μ0/(4π r^3) * [3 (m·r^) r^ - m]
        ΔT = B · l = (F / (4π r^3)) * V * chi * [3 (l·r^)^2 - 1]

    Therefore, the kernel per unit susceptibility (chi = 1) is:
        k_TMI = (F / (4π)) * V * [3 (l·r^)^2 - 1] / r^3   [same unit as F]

    Where:
        - F is the IGRF total intensity (we accept nT directly if units_nT=True)
        - l is the unit vector of field direction from inclination/declination
        - r is distance from device center to voxel center
        - V is voxel volume

    Args:
        centered_grid: Grid definition with observation centers, resolution, and radii.
        igrf_params: dict with keys {"inclination", "declination", "intensity"}. Intensity in nT if units_nT=True.
        compute_tmi: If True, returns dict with 'tmi_kernel'. Full tensor not implemented in MVP.
        units_nT: If True, outputs kernel in nT per unit susceptibility. If False, outputs in Tesla.

    Returns:
        dict with keys:
          - 'tmi_kernel': np.ndarray, shape (n_voxels_per_device,) when compute_tmi=True
          - 'field_direction': np.ndarray, shape (3,)
          - 'inclination', 'declination', 'intensity'

    Notes:
        - Kernel is computed for the first device and assumed identical for all devices
          (same relative voxel layout), consistent with gravity implementation/usage.
        - Numerical safeguards added to avoid r=0 singularities.
    """
    if not compute_tmi:
        # Placeholder for future full tensor computation
        raise NotImplementedError("Full magnetic gradient tensor computation is not implemented yet.")

    # Extract grid geometry
    centers = np.asarray(centered_grid.centers)
    values = np.asarray(centered_grid.values)
    resolution = np.asarray(centered_grid.resolution, dtype=float)
    radius = np.asarray(centered_grid.radius, dtype=float)

    if centers.ndim != 2 or centers.shape[0] < 1:
        raise ValueError("CenteredGrid.centers must have at least one device center.")

    n_devices = centers.shape[0]
    n_voxels_total = values.shape[0]
    if n_devices <= 0 or n_voxels_total % n_devices != 0:
        raise ValueError("Values array length must be divisible by number of device centers.")
    n_vox_per_device = n_voxels_total // n_devices

    # Slice first device voxel positions and compute relative vectors
    c0 = centers[0]
    vc0 = values[:n_vox_per_device]
    r_vec = vc0 - c0
    r = np.linalg.norm(r_vec, axis=1)

    # Numerical safe-guard: avoid division by zero at device center
    eps = 1e-12
    r_safe = np.maximum(r, eps)
    r_hat = r_vec / r_safe[:, None]

    # Voxel volume (grid symmetric around device): V = (2*rx/nx)*(2*ry/ny)*(2*rz/nz)
    # radius can be scalar or 3-array; ensure 3-array
    if radius.size == 1:
        rx = ry = rz = float(radius)
    else:
        rx, ry, rz = radius.astype(float)
    nx, ny, nz = resolution.astype(float)
    V = (2.0 * rx / nx) * (2.0 * ry / ny) * (2.0 * rz / nz)

    # IGRF direction and intensity
    I = float(igrf_params.get("inclination", 0.0))
    D = float(igrf_params.get("declination", 0.0))
    F = float(igrf_params.get("intensity", 50000.0))  # default ~50,000 nT

    l = _direction_cosines(I, D)

    # Cosine between l and r_hat
    cos_lr = np.clip(r_hat @ l, -1.0, 1.0)

    # Point-dipole TMI kernel per unit susceptibility chi=1
    # k = V * F / (4π) * (3*cos^2 - 1) / r^3
    factor = (3.0 * cos_lr * cos_lr - 1.0) / (r_safe ** 3)
    kernel = (V * F / (4.0 * np.pi)) * factor

    if not units_nT:
        # Convert nT to Tesla if requested
        kernel = kernel * 1e-9

    return {
        "tmi_kernel": kernel.astype(float),
        "field_direction": l,
        "inclination": I,
        "declination": D,
        "intensity": F,
    }
