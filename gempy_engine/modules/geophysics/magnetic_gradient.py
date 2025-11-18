import numpy as np

from gempy_engine.core.data.centered_grid import CenteredGrid


def calculate_magnetic_gradient_components(centered_grid: CenteredGrid) -> np.ndarray:
    """
    Calculate the 6 independent magnetic gradient tensor components (V1-V6) for each voxel.
    
    This is the geometry-dependent part that can be precomputed and reused with different
    IGRF parameters or susceptibility values. Follows the legacy implementation approach.
    
    Args:
        centered_grid: Grid definition with observation centers, resolution, and radii.
    
    Returns:
        V: np.ndarray of shape (6, n_voxels_per_device) containing the volume integrals:
           V[0, :] = V1 (related to T_xx)
           V[1, :] = V2 (related to T_xy)
           V[2, :] = V3 (related to T_xz)
           V[3, :] = V4 (related to T_yy)
           V[4, :] = V5 (related to T_yz)
           V[5, :] = V6 (related to T_zz)
    
    Notes:
        These components represent the analytical integration of the magnetic gradient tensor
        over rectangular prism voxels using the formulas from Blakely (1995).
        The sign convention follows Talwani (z-axis positive downwards).
    """

    voxel_centers = centered_grid.kernel_grid_centers
    center_x, center_y, center_z = voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2]

    # Calculate the coordinates of the voxel corners
    left_edges = centered_grid.left_voxel_edges
    right_edges = centered_grid.right_voxel_edges

    x_corners = np.stack((center_x - left_edges[:, 0], center_x + right_edges[:, 0]), axis=1)
    y_corners = np.stack((center_y - left_edges[:, 1], center_y + right_edges[:, 1]), axis=1)
    z_corners = np.stack((center_z - left_edges[:, 2], center_z + right_edges[:, 2]), axis=1)

    # Prepare coordinates for vector operations
    x_matrix = np.repeat(x_corners, 4, axis=1)
    y_matrix = np.tile(np.repeat(y_corners, 2, axis=1), (1, 2))
    z_matrix = np.tile(z_corners, (1, 4))

    # Distance to each corner
    R = np.sqrt(x_matrix ** 2 + y_matrix ** 2 + z_matrix ** 2)

    # Add small epsilon to avoid log(0) and division by zero
    eps = 1e-12
    R = np.maximum(R, eps)

    # Sign pattern for 8 corners (depends on coordinate system)
    s = np.array([-1, 1, 1, -1, 1, -1, -1, 1])

    # Variables V1-6 represent volume integrals for each voxel
    # These are the 6 independent components of the symmetric magnetic gradient tensor
    V1 = np.sum(-1 * s * np.arctan2((y_matrix * z_matrix), (x_matrix * R + eps)), axis=1)
    V2 = np.sum(s * np.log(R + z_matrix + eps), axis=1)
    V3 = np.sum(s * np.log(R + y_matrix + eps), axis=1)
    V4 = np.sum(-1 * s * np.arctan2((x_matrix * z_matrix), (y_matrix * R + eps)), axis=1)
    V5 = np.sum(s * np.log(R + x_matrix + eps), axis=1)
    V6 = np.sum(-1 * s * np.arctan2((x_matrix * y_matrix), (z_matrix * R + eps)), axis=1)

    # Stack into shape (6, n_voxels) matching legacy implementation
    V = np.array([V1, V2, V3, V4, V5, V6])

    return V


def calculate_magnetic_gradient_tensor(
        centered_grid: CenteredGrid,
        igrf_params: dict,
        compute_tmi: bool = True,
        units_nT: bool = True,
) -> dict:
    """
    Compute magnetic kernels for voxelized forward modeling around each observation point.

    This is a convenience wrapper that combines calculate_magnetic_gradient_components()
    and pre-projection for TMI. For maximum flexibility, use the component functions directly.

    Args:
        centered_grid: Grid definition with observation centers, resolution, and radii.
        igrf_params: dict with keys {"inclination", "declination", "intensity"}. Intensity in nT if units_nT=True.
        compute_tmi: If True, returns pre-projected TMI kernel. If False, returns V components.
        units_nT: If True, outputs kernel in nT per unit susceptibility. If False, outputs in Tesla.

    Returns:
        dict with keys:
          - 'tmi_kernel': np.ndarray, shape (n_voxels_per_device,) when compute_tmi=True
          - 'V': np.ndarray, shape (6, n_voxels_per_device) when compute_tmi=False
          - 'field_direction': np.ndarray, shape (3,)
          - 'inclination', 'declination', 'intensity'
    """
    # Compute V components (geometry-dependent only)
    V = calculate_magnetic_gradient_components(centered_grid)

    # Extract IGRF parameters
    I = float(igrf_params.get("inclination", 0.0))
    D = float(igrf_params.get("declination", 0.0))
    F = float(igrf_params.get("intensity", 50000.0))  # in nT

    l = _direction_cosines(I, D)

    result = {
            "field_direction": l,
            "inclination"    : I,
            "declination"    : D,
            "intensity"      : F,
    }

    if compute_tmi:
        # Pre-project V components into TMI kernel for faster forward modeling
        # This combines the V tensor with the field direction ahead of time
        F_tesla = F * 1e-9

        # TMI kernel per unit susceptibility:
        # tmi_kernel = (F / (4*pi)) * [l_x^2*V1 + l_y^2*V4 + l_z^2*V6 + 
        #                               2*l_x*l_y*V2 + 2*l_x*l_z*V3 + 2*l_y*l_z*V5]
        tmi_kernel = (
                l[0] * l[0] * V[0, :] +  # l_x^2 * V1
                l[1] * l[1] * V[3, :] +  # l_y^2 * V4
                l[2] * l[2] * V[5, :] +  # l_z^2 * V6
                2 * l[0] * l[1] * V[1, :] +  # 2*l_x*l_y * V2
                2 * l[0] * l[2] * V[2, :] +  # 2*l_x*l_z * V3
                2 * l[1] * l[2] * V[4, :]  # 2*l_y*l_z * V5
        )

        tmi_kernel = tmi_kernel * F_tesla / (4 * np.pi)

        if units_nT:
            tmi_kernel = tmi_kernel * 1e9

        result["tmi_kernel"] = tmi_kernel.astype(float)
    else:
        # Return raw V components for maximum flexibility
        result["V"] = V

    return result


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
