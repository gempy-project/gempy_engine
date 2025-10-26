import numpy as np

from ...core.data.geophysics_input import GeophysicsInput
from ...core.data.interp_output import InterpOutput
from ...core.backend_tensor import BackendTensor


def map_susceptibilities_to_ids_basic(ids_geophysics_grid, susceptibilities):
    """
    Map per-unit susceptibilities to voxel centers using 1-based geological IDs.
    Matches gravity's basic mapper behavior.
    """
    return susceptibilities[ids_geophysics_grid - 1]


def compute_tmi(geophysics_input: GeophysicsInput, root_ouput: InterpOutput) -> BackendTensor.t:
    """
    Compute induced-only Total Magnetic Intensity (TMI) anomalies (nT) by combining
    precomputed per-voxel TMI kernel with voxel susceptibilities.

    Expectations for Phase 1 (MVP):
    - geophysics_input.mag_kernel is a scalar kernel per voxel (for one device geometry)
      that already includes projection along the IGRF field direction and unit handling
      (nT per unit susceptibility).
    - geophysics_input.susceptibilities is an array of susceptibilities per geologic unit (SI).
    - root_ouput.ids_geophysics_grid are 1-based IDs mapping each voxel to a geologic unit.

    Returns:
        BackendTensor.t array of shape (n_devices,) with TMI anomaly in nT.
    """
    if geophysics_input.mag_kernel is None:
        raise ValueError("GeophysicsInput.mag_kernel is required for magnetic forward modeling.")
    if geophysics_input.susceptibilities is None:
        raise ValueError("GeophysicsInput.susceptibilities is required for magnetic forward modeling.")

    # Kernel for one device geometry
    mag_kernel = BackendTensor.t.array(geophysics_input.mag_kernel)

    # Map susceptibilities to voxel centers according to IDs (1-based indexing)
    chi = map_susceptibilities_to_ids_basic(
        ids_geophysics_grid=root_ouput.ids_geophysics_grid,
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
