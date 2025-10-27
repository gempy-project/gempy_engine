import numpy as np

from gempy_engine.core.data.centered_grid import CenteredGrid


def calculate_magnetic_gradient_tensor(
        centered_grid: CenteredGrid,
        igrf_params: dict,
        compute_tmi: bool = True,
        units_nT: bool = True,
):
    """
    Compute magnetic kernels for voxelized forward modeling around each observation point.

    This implementation uses analytical rectangular prism integration (Blakely 1995)
    for accurate magnetic field computation from finite voxels.

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
    """
    if not compute_tmi:
        raise NotImplementedError("Full magnetic gradient tensor computation is not implemented yet.")

    # Extract grid geometry - use kernel_centers, kernel_dxyz_right, kernel_dxyz_left
    grid_values = np.asarray(centered_grid.kernel_centers)
    dxyz_right = np.asarray(centered_grid.kernel_dxyz_right)
    dxyz_left = np.asarray(centered_grid.kernel_dxyz_left)

    if grid_values.ndim != 2 or grid_values.shape[0] < 1:
        raise ValueError("CenteredGrid.kernel_centers must have at least one voxel.")

    # Get voxel center coordinates (observation point is at origin in kernel space)
    s_gr_x = grid_values[:, 0]
    s_gr_y = grid_values[:, 1]
    s_gr_z = -1 * grid_values[:, 2]  # Talwani takes z-axis positive downwards

    # Getting the coordinates of the corners of the voxel
    x_cor = np.stack((s_gr_x - dxyz_left[:, 0], s_gr_x + dxyz_right[:, 0]), axis=1)
    y_cor = np.stack((s_gr_y - dxyz_left[:, 1], s_gr_y + dxyz_right[:, 1]), axis=1)
    z_cor = np.stack((s_gr_z + dxyz_left[:, 2], s_gr_z - dxyz_right[:, 2]), axis=1)

    # Prepare them for vectorial operations
    x_matrix = np.repeat(x_cor, 4, axis=1)
    y_matrix = np.tile(np.repeat(y_cor, 2, axis=1), (1, 2))
    z_matrix = np.tile(z_cor, (1, 4))

    # Distance to each corner
    R = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

    # Add small epsilon to avoid log(0) and division by zero
    eps = 1e-12
    R = np.maximum(R, eps)

    # Gives the sign of each corner (depends on coordinate system)
    s = np.array([-1, 1, 1, -1, 1, -1, -1, 1])

    # Variables V1-6 represent integrals of volume for each voxel
    # These are the 6 independent components of the magnetic gradient tensor
    V1 = np.sum(-1 * s * np.arctan2((y_matrix * z_matrix), (x_matrix * R + eps)), axis=1)
    V2 = np.sum(s * np.log(R + z_matrix + eps), axis=1)
    V3 = np.sum(s * np.log(R + y_matrix + eps), axis=1)
    V4 = np.sum(-1 * s * np.arctan2((x_matrix * z_matrix), (y_matrix * R + eps)), axis=1)
    V5 = np.sum(s * np.log(R + x_matrix + eps), axis=1)
    V6 = np.sum(-1 * s * np.arctan2((x_matrix * y_matrix), (z_matrix * R + eps)), axis=1)

    # IGRF direction and intensity
    I = float(igrf_params.get("inclination", 0.0))
    D = float(igrf_params.get("declination", 0.0))
    F = float(igrf_params.get("intensity", 50000.0))  # default ~50,000 nT

    l = _direction_cosines(I, D)

    # Now combine V1-V6 with field direction to compute TMI kernel
    # The magnetic field components at the observation point due to a voxel with 
    # unit magnetization M = [Mx, My, Mz] are:
    #   Bx = Mx*V1 + My*V2 + Mz*V3
    #   By = Mx*V2 + My*V4 + Mz*V5  
    #   Bz = Mx*V3 + My*V5 + Mz*V6
    #
    # For induced magnetization: M = chi * B0 / mu0, where B0 = F * l
    # So M_x = chi * F * l_x / mu0, etc.
    #
    # TMI anomaly = (Bx*l_x + By*l_y + Bz*l_z)
    #
    # Substituting and simplifying (chi cancels for kernel):
    # TMI_kernel = (F/mu0) * [l_x*(l_x*V1 + l_y*V2 + l_z*V3) + 
    #                         l_y*(l_x*V2 + l_y*V4 + l_z*V5) +
    #                         l_z*(l_x*V3 + l_y*V5 + l_z*V6)]
    #            = (F/mu0) * [l_x^2*V1 + l_y^2*V4 + l_z^2*V6 + 
    #                         2*l_x*l_y*V2 + 2*l_x*l_z*V3 + 2*l_y*l_z*V5]

    tmi_kernel = (
            l[0] * l[0] * V1 +  # l_x^2 * T_xx
            l[1] * l[1] * V4 +  # l_y^2 * T_yy
            l[2] * l[2] * V6 +  # l_z^2 * T_zz
            2 * l[0] * l[1] * V2 +  # 2 * l_x * l_y * T_xy
            2 * l[0] * l[2] * V3 +  # 2 * l_x * l_z * T_xz
            2 * l[1] * l[2] * V5  # 2 * l_y * l_z * T_yz
    )

    # Apply physical constants and field intensity
    # mu_0 / (4*pi) = 1e-7 in SI units
    CM = 1e-7  # Tesla * m / Ampere (this is mu_0/(4*pi))

    # Scale by field intensity
    tmi_kernel = tmi_kernel * F * CM

    if units_nT:
        # Convert to nT
        tmi_kernel = tmi_kernel * 1e9
    # else: already in Tesla

    return {
            "tmi_kernel"     : tmi_kernel.astype(float),
            "field_direction": l,
            "inclination"    : I,
            "declination"    : D,
            "intensity"      : F,
    }