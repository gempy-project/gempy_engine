import numpy as np

from .magnetic_gradient import _direction_cosines
from ...core.data.geophysics_input import GeophysicsInput, MagneticsInput
from ...core.data.interp_output import InterpOutput
from ...core.backend_tensor import BackendTensor


def map_susceptibilities_to_ids_basic(ids_geophysics_grid, susceptibilities):
    """
    Map per-unit susceptibilities to voxel centers using 1-based geological IDs.
    Matches gravity's basic mapper behavior.
    """
    return susceptibilities[ids_geophysics_grid - 1]


def compute_tmi(geophysics_input: MagneticsInput, root_output: InterpOutput) -> BackendTensor.t:
    """
    Compute induced-only Total Magnetic Intensity (TMI) anomalies (nT) by combining
    precomputed per-voxel TMI kernel with voxel susceptibilities.

    Expectations for Phase 1 (MVP):
    - geophysics_input.mag_kernel is a scalar kernel per voxel (for one device geometry)
      that already includes projection along the IGRF field direction and unit handling
      (nT per unit susceptibility).
    - geophysics_input.susceptibilities is an array of susceptibilities per geologic unit (SI).
    - root_output.ids_geophysics_grid are 1-based IDs mapping each voxel to a geologic unit.

    Returns:
        BackendTensor.t array of shape (n_devices,) with TMI anomaly in nT.
        
    Notes:
        This function works with pre-projected TMI kernels from 
        calculate_magnetic_gradient_tensor(..., compute_tmi=True).
        For more flexibility (e.g., remanent magnetization in Phase 2), 
        consider using the raw V components and compute_magnetic_forward().
    """
    if geophysics_input.mag_kernel is None:
        raise ValueError("GeophysicsInput.mag_kernel is required for magnetic forward modeling.")
    if geophysics_input.susceptibilities is None:
        raise ValueError("GeophysicsInput.susceptibilities is required for magnetic forward modeling.")

    # Kernel for one device geometry
    mag_kernel = BackendTensor.t.array(geophysics_input.mag_kernel)

    # Map susceptibilities to voxel centers according to IDs (1-based indexing)
    chi = map_susceptibilities_to_ids_basic(
        ids_geophysics_grid=root_output.ids_geophysics_grid,
        susceptibilities=BackendTensor.t.array(geophysics_input.susceptibilities)
    )

    # Determine how many devices are present by comparing lengths
    n_devices = chi.shape[0] // mag_kernel.shape[0]

    # Reshape for batch multiply-sum across devices
    mag_kernel = mag_kernel.reshape(1, -1)
    chi = chi.reshape(n_devices, -1)

    # Weighted sum per device
    tmi = BackendTensor.t.sum(chi * mag_kernel, axis=1)
    return tmi


def compute_magnetic_forward(
        V: np.ndarray,
        susceptibilities: np.ndarray,
        igrf_params: dict,
        n_devices: int = 1,
        units_nT: bool = True
) -> np.ndarray:
    """
    Compute Total Magnetic Intensity (TMI) anomalies from precomputed tensor components.

    This follows the legacy implementation workflow: combine geometry-dependent V components
    with susceptibility and IGRF field parameters to compute TMI.

    Args:
        V: np.ndarray of shape (6, n_voxels_per_device) from calculate_magnetic_gradient_components()
        susceptibilities: np.ndarray of shape (n_total_voxels,) - susceptibility per voxel for all devices
        igrf_params: dict with keys {"inclination", "declination", "intensity"} (intensity in nT)
        n_devices: Number of observation devices (default 1)
        units_nT: If True, output in nT; if False, output in Tesla

    Returns:
        dT: np.ndarray of shape (n_devices,) - TMI anomaly at each observation point

    Notes:
        Implements the formula from legacy code:
        - Compute induced magnetization: J = chi * B_ext
        - Compute directional components: Jx, Jy, Jz
        - Apply gradient tensor: Tx, Ty, Tz using V components
        - Project onto field direction: dT = Tx*dir_x + Ty*dir_y + Tz*dir_z
    """
    # Extract IGRF parameters
    incl = float(igrf_params.get("inclination", 0.0))
    decl = float(igrf_params.get("declination", 0.0))
    B_ext = float(igrf_params.get("intensity", 50000.0))  # in nT

    # Convert to Tesla for internal computation
    B_ext_tesla = B_ext * 1e-9

    # Get field direction cosines
    dir_x, dir_y, dir_z = _direction_cosines(incl, decl)

    # Compute induced magnetization [Tesla]
    # J = chi * B_ext (susceptibility is dimensionless)
    J = susceptibilities * B_ext_tesla

    # Compute magnetization components along field direction
    Jx = dir_x * J
    Jy = dir_y * J
    Jz = dir_z * J

    # Tile V for multiple devices (repeat the kernel for each device)
    V_tiled = np.tile(V, (1, n_devices))

    # Compute directional magnetic effect on each voxel using V components
    # This is equation (3.19) from the theory
    # Bx = Jx*V1 + Jy*V2 + Jz*V3
    # By = Jx*V2 + Jy*V4 + Jz*V5  (V2 = V[1] because tensor is symmetric)
    # Bz = Jx*V3 + Jy*V5 + Jz*V6
    Tx = (Jx * V_tiled[0, :] + Jy * V_tiled[1, :] + Jz * V_tiled[2, :]) / (4 * np.pi)
    Ty = (Jx * V_tiled[1, :] + Jy * V_tiled[3, :] + Jz * V_tiled[4, :]) / (4 * np.pi)
    Tz = (Jx * V_tiled[2, :] + Jy * V_tiled[4, :] + Jz * V_tiled[5, :]) / (4 * np.pi)

    # Sum over voxels for each device
    n_voxels_per_device = V.shape[1]
    Tx = np.sum(Tx.reshape((n_devices, n_voxels_per_device)), axis=1)
    Ty = np.sum(Ty.reshape((n_devices, n_voxels_per_device)), axis=1)
    Tz = np.sum(Tz.reshape((n_devices, n_voxels_per_device)), axis=1)

    # Project onto field direction to get TMI
    # "Total field magnetometers can measure only that part of the anomalous field 
    # which is in the direction of the Earth's main field" (SimPEG documentation)
    dT = Tx * dir_x + Ty * dir_y + Tz * dir_z

    if units_nT:
        # Convert to nT
        dT = dT * 1e9

    return dT
